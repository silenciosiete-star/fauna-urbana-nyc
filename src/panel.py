"""Panel web: stream MJPEG en directo + historial de hitos."""
import csv
import datetime
import queue
import threading
import time
from collections import deque
from pathlib import Path

import plotly.graph_objects as go

import cv2
import numpy as np
import supervision as sv
from dash import ALL, Dash, Input, Output, State, dcc, html
from flask import Response
from loguru import logger

from .base_datos import BaseDatos
from .rastreador import ResultadoTracking
from .simulador import Simulador
from .verificador import Verificador
from .zonas import Zona

_DIR_ENTRENAMIENTO = Path(__file__).parent.parent / "modelos" / "fauna_urbana"

_MAP50_POR_CLASE = {
    "deadpool":         0.995,
    "gorila":           0.995,
    "transformer":      0.990,
    "estatua_libertad": 0.957,
    "sonic":            0.911,
    "spiderman":        0.910,
    "super_mario":      0.900,
    "batman":           0.849,
    "minnie_mouse":     0.823,
    "elmo":             0.765,
    "mickey_mouse":     0.580,
}

_HITOS_SIMULABLES = [
    ("crossover",           "Crossover"),
    ("marvel_vs_dc",        "Marvel vs DC"),
    ("hora_punta",          "Hora Punta"),
    ("conflicto_identidad", "Conflicto de Identidad"),
    ("avistamiento_raro",   "Avistamiento Raro"),
]

_ANCHO_STREAM = 960
_HEATMAP_CAP   = 3.0    # pico post-blur por detección es ~1.8; cap=3 → amarillo a 1 paso, rojo a ~2
_HEATMAP_DECAY = 0.990  # factor de decay por frame (~50% en 3 s a 25 fps)
_HEATMAP_RADIO = 20     # radio fijo del punto de calor en píxeles (igual para todos los personajes)
_TRAIL_LONGITUD = 100   # posiciones por personaje (~4 s a 25 fps)
_TRAIL_GROSOR   = 3     # grosor máximo de la línea de trayectoria
_INTERVALO_REFRESCO_MS = 2000
_VENTANA_FPS = 30
_COLORES_ZONA = {
    "fauna": (0, 200, 0),
    "esquina_norte": (0, 200, 220),
    "esquina_sur": (200, 0, 200),
}
_COLOR_ZONA_DEFAULT = (180, 180, 180)
_COLORES_HITO = {
    "crossover":           "#ff9800",
    "conflicto_identidad": "#e040fb",
    "hora_punta":          "#00bcd4",
    "avistamiento_raro":   "#4caf50",
    "marvel_vs_dc":        "#f44336",
}
_COLOR_HITO_DEFAULT = "#607d8b"
_DIR_ASSETS = str(Path(__file__).parent.parent / "assets")
_FUENTES_EXT = [
    "https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500&display=swap"
]


class Panel:

    def __init__(
        self,
        cola_frames: queue.Queue,
        cola_tracking: queue.Queue,
        zonas: dict[str, Zona],
        base_datos: BaseDatos,
        puerto: int = 8050,
        carpeta_capturas: str = "capturas",
    ):
        self._cola_frames = cola_frames
        self._cola_tracking = cola_tracking
        self._zonas = zonas
        self._base_datos = base_datos
        self._puerto = puerto
        self._carpeta_capturas = Path(carpeta_capturas)
        self._activo = False
        self._pausado = False
        self._simulador: Simulador | None = None
        self._verificador: Verificador | None = None
        self._capturador = None
        self._hilo_frames: threading.Thread | None = None
        self._ultimo_tracking: ResultadoTracking | None = None
        self._ultimo_frame: np.ndarray | None = None
        self._lock_frame = threading.Lock()
        self._tiempos_frame: deque = deque(maxlen=_VENTANA_FPS)
        self._anotador_cajas = sv.BoxAnnotator()
        self._anotador_etiquetas = sv.LabelAnnotator()
        self._heatmap_activo = False
        self._acum_heatmap: np.ndarray | None = None
        self._lock_heatmap = threading.Lock()
        self._tracking_activo = False
        self._trails: dict[int, deque] = {}
        self._lock_trails = threading.Lock()
        self._app = self._crear_app()

    def conectar_simulador(self, simulador: Simulador) -> None:
        self._simulador = simulador
        self._capturador = simulador._capturador

    def conectar_verificador(self, verificador: Verificador) -> None:
        self._verificador = verificador

    def iniciar(self) -> None:
        self._activo = True
        self._hilo_frames = threading.Thread(target=self._bucle_frames, daemon=True)
        self._hilo_frames.start()
        hilo_dash = threading.Thread(
            target=lambda: self._app.run(host="0.0.0.0", port=self._puerto, debug=False),
            daemon=True,
        )
        hilo_dash.start()
        logger.info(f"Panel web iniciado en http://localhost:{self._puerto}")

    def detener(self) -> None:
        self._activo = False
        if self._hilo_frames:
            self._hilo_frames.join(timeout=2)
        logger.info("Panel web detenido")

    # ------------------------------------------------------------------

    def _bucle_frames(self) -> None:
        while self._activo:
            while True:
                try:
                    self._ultimo_tracking = self._cola_tracking.get_nowait()
                except queue.Empty:
                    break

            # Durante simulación usar el frame del dataset en lugar del stream
            if self._simulador and self._simulador.frame_override is not None:
                frame = self._simulador.frame_override
            else:
                frame = None
                try:
                    while True:
                        frame = self._cola_frames.get_nowait()
                except queue.Empty:
                    pass
                if frame is None:
                    try:
                        frame = self._cola_frames.get(timeout=0.05)
                    except queue.Empty:
                        continue

            self._tiempos_frame.append(time.monotonic())
            simulando = self._simulador.simulando if self._simulador else None
            frame_anotado = self._anotar_frame(frame, simulando)
            alto = int(frame_anotado.shape[0] * _ANCHO_STREAM / frame_anotado.shape[1])
            frame_redim = cv2.resize(frame_anotado, (_ANCHO_STREAM, alto))
            with self._lock_frame:
                self._ultimo_frame = frame_redim

    def _generar_mjpeg(self):
        ultimo_buffer: bytes | None = None
        while True:
            simulando = self._simulador is not None and self._simulador.simulando is not None
            if not self._pausado or simulando:
                with self._lock_frame:
                    frame = self._ultimo_frame
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        ultimo_buffer = buf.tobytes()
            if ultimo_buffer:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + ultimo_buffer
                    + b"\r\n"
                )
            time.sleep(1 / 25)

    def _crear_app(self) -> Dash:
        app = Dash(
            __name__,
            title="Fauna Urbana NYC",
            assets_folder=_DIR_ASSETS,
            external_stylesheets=_FUENTES_EXT,
        )

        @app.server.route("/stream")
        def stream():
            return Response(
                self._generar_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.server.route("/modelo-img/<nombre>")
        def servir_imagen_modelo(nombre):
            from flask import send_from_directory
            return send_from_directory(str(_DIR_ENTRENAMIENTO), nombre)

        @app.server.route("/captura-img/<nombre>")
        def servir_captura(nombre):
            from flask import send_from_directory
            return send_from_directory(str(self._carpeta_capturas.resolve()), nombre)

        _estilo_drawer_base = {
            "position": "fixed", "top": 0, "width": "380px", "height": "100vh",
            "background": "#0d0d1a", "borderLeft": "1px solid #2a2a3e",
            "zIndex": 1000, "padding": "24px 20px",
            "boxShadow": "-4px 0 20px rgba(0,0,0,0.5)",
            "transition": "right 0.3s ease", "overflowY": "auto",
        }

        app.layout = html.Div(
            style={"minHeight": "100vh", "padding": "20px 24px"},
            children=[
                dcc.Store(id="hito-seleccionado"),
                dcc.Store(id="menu-abierto", data=False),
                dcc.Store(id="menu-posicion", data="abajo"),
                # ── Header ──────────────────────────────────────────
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "20px"},
                    children=[
                        html.Span("Fauna Urbana NYC", className="titulo"),
                        html.Div(id="live-badge", className="live-badge",
                                 children=[html.Div(className="live-dot"), "LIVE"]),
                    ],
                ),
                # ── Stats BD ─────────────────────────────────────────
                html.Div(id="barra-stats", style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),
                dcc.Interval(id="intervalo", interval=_INTERVALO_REFRESCO_MS, n_intervals=0),
                # ── Contenido principal ──────────────────────────────
                html.Div(
                    style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
                    children=[
                        # Columna stream + controles
                        html.Div(
                            style={"flex": "1.6"},
                            children=[
                                html.Div(
                                    className="stream-wrapper",
                                    children=[html.Img(src="/stream")],
                                ),
                                html.Div(
                                    className="controles",
                                    children=[
                                        html.Button("⏸  Pausar", id="btn-pausa", className="btn-control", n_clicks=0),
                                        html.Button("📸  Captura", id="btn-captura", className="btn-control", n_clicks=0),
                                        html.Div(
                                            style={"position": "relative", "display": "inline-block"},
                                            children=[
                                                html.Button("🎭  Simular ▾", id="btn-simular-toggle", className="btn-control", n_clicks=0),
                                                html.Div(
                                                    id="menu-simular",
                                                    style={"display": "none", "position": "absolute",
                                                           "top": "calc(100% + 6px)", "left": 0,
                                                           "background": "#13132a", "border": "1px solid #2a2a4e",
                                                           "borderRadius": "8px", "zIndex": 200,
                                                           "minWidth": "220px", "padding": "6px 0",
                                                           "boxShadow": "0 8px 24px rgba(0,0,0,0.6)"},
                                                    children=[
                                                        html.Button(
                                                            etiqueta,
                                                            id=f"btn-sim-{tipo}",
                                                            className="btn-sim-item",
                                                            n_clicks=0,
                                                            style={
                                                                "display": "block", "width": "100%",
                                                                "textAlign": "left", "padding": "9px 16px",
                                                                "border": "none", "background": "transparent",
                                                                "color": "#c0c0e0", "cursor": "pointer",
                                                                "fontSize": "0.88em", "letterSpacing": "0.03em",
                                                                "borderLeft": f"3px solid {_COLORES_HITO.get(tipo, _COLOR_HITO_DEFAULT)}",
                                                            },
                                                        )
                                                        for tipo, etiqueta in _HITOS_SIMULABLES
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Button("📊  Modelo", id="btn-modelo", className="btn-control", n_clicks=0),
                                        html.Button("🌡  Calor", id="btn-calor", className="btn-control", n_clicks=0),
                                        html.Button("🛤  Trayectorias", id="btn-trayectorias", className="btn-control", n_clicks=0),
                                        html.Span(id="msg-captura", className="msg-control"),
                                        html.Span(id="msg-simular", className="msg-control"),
                                    ],
                                ),
                            ],
                        ),
                        # Columna hitos
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Div("Hitos recientes", className="seccion-titulo"),
                                html.Div(id="lista-hitos", style={"overflowY": "auto", "maxHeight": "460px"}),
                            ],
                        ),
                    ],
                ),
                # ── Backdrop menú simular ────────────────────────────
                html.Div(
                    id="menu-backdrop",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 199},
                ),
                # ── Backdrop (cierra el drawer al hacer click fuera) ──
                html.Div(
                    id="drawer-backdrop",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 999},
                ),
                # ── Drawer lateral derecho: detalle hito ────────────
                html.Div(
                    id="cajita-detalle",
                    style={**_estilo_drawer_base, "right": "-400px"},
                    children=[
                        html.Div(id="cajita-detalle-titulo", style={"marginBottom": "16px"}),
                        html.Div(id="cajita-detalle-cuerpo"),
                    ],
                ),
                # ── Backdrop modelo ──────────────────────────────────
                html.Div(
                    id="drawer-backdrop-modelo",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 999},
                ),
                # ── Drawer lateral derecho: modelo ───────────────────
                html.Div(
                    id="cajita-modelo",
                    style={**_estilo_drawer_base, "right": "-800px", "width": "780px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "4px"},
                            children=[
                                html.Div("MODELO — YOLO26s fine-tuned",
                                         style={"fontFamily": "Rajdhani, sans-serif",
                                                "fontSize": "1.1em", "fontWeight": "700",
                                                "color": "#00bcd4"}),
                                html.Button("✕", id="btn-cerrar-modelo", n_clicks=0,
                                            style={"background": "none", "border": "none",
                                                   "color": "#555", "cursor": "pointer",
                                                   "fontSize": "1.2em", "lineHeight": "1"}),
                            ],
                        ),
                        html.Div(id="cajita-modelo-cuerpo"),
                    ],
                ),
            ],
        )

        # ── Callbacks ────────────────────────────────────────────────

        app.clientside_callback(
            """
            function(n_clicks) {
                if (!n_clicks) return 'abajo';
                var btn = document.getElementById('btn-simular-toggle');
                if (!btn) return 'abajo';
                var rect = btn.getBoundingClientRect();
                return (window.innerHeight - rect.bottom) >= 220 ? 'abajo' : 'arriba';
            }
            """,
            Output("menu-posicion", "data"),
            Input("btn-simular-toggle", "n_clicks"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output("lista-hitos", "children"),
            Output("barra-stats", "children"),
            Output("live-badge", "children"),
            Input("intervalo", "n_intervals"),
        )
        def actualizar_hitos(_):
            hitos = self._base_datos.hitos_recientes(limite=100)
            confirmados = sum(1 for h in hitos if h["confirmado"])
            descartados = len(hitos) - confirmados
            barra = [
                _stat_card(str(len(hitos)),  "hitos",       "#e0e0f0"),
                _stat_card(str(confirmados), "confirmados", "#00e676"),
                _stat_card(str(descartados), "descartados", "#ff5252"),
            ]
            simulando = self._simulador.simulando if self._simulador else None
            if simulando:
                badge = [html.Div(className="live-dot",
                                  style={"background": "#ff9800", "animationDuration": "0.6s"}),
                         f"SIM · {simulando.replace('_', ' ').upper()}"]
            elif self._pausado:
                badge = [html.Div(className="live-dot",
                                  style={"background": "#607d8b", "animationPlayState": "paused"}),
                         "PAUSADO"]
            else:
                badge = [html.Div(className="live-dot"), "LIVE"]
            elementos = []
            if self._verificador:
                for p in self._verificador.hitos_en_proceso():
                    color = _COLORES_HITO.get(p["tipo"], _COLOR_HITO_DEFAULT)
                    elementos.append(html.Div(
                        className="hito-card",
                        style={"borderLeftColor": color, "opacity": "0.75"},
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                children=[
                                    html.Span("⏳", style={"fontSize": "0.9em"}),
                                    html.Span(p["tipo"].replace("_", " ").upper(),
                                              className="hito-tipo",
                                              style={"color": color}),
                                    html.Span("verificando...",
                                              style={"marginLeft": "auto", "fontSize": "0.72em",
                                                     "color": "#888", "fontStyle": "italic"}),
                                ],
                            ),
                            html.Div(p["descripcion"], className="hito-mensaje"),
                        ],
                    ))
            if not hitos and not elementos:
                return html.P("Sin hitos registrados aún.", style={"color": "#404060", "fontSize": "0.85em"}), barra, badge
            for h in hitos[:30]:
                ts = datetime.datetime.fromtimestamp(h["marca_tiempo"]).strftime("%d/%m %H:%M")
                color_borde = _COLORES_HITO.get(h["tipo"], _COLOR_HITO_DEFAULT)
                color_estado = "#00e676" if h["confirmado"] else "#ff5252"
                estado_txt = "✓" if h["confirmado"] else "✗"
                elementos.append(html.Div(
                    id={"type": "hito-card", "index": h["id"]},
                    className="hito-card hito-card-clickable",
                    style={"borderLeftColor": color_borde, "cursor": "pointer"},
                    n_clicks=0,
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center"},
                            children=[
                                html.Span(ts, className="hito-ts"),
                                html.Span(estado_txt, className="hito-estado", style={"color": color_estado}),
                                html.Span(h["tipo"].replace("_", " "), className="hito-tipo"),
                                html.Span("ver detalle →", style={"marginLeft": "auto", "fontSize": "0.75em", "color": "#555"}),
                            ],
                        ),
                        html.Div(h["mensaje"] or h["descripcion"], className="hito-mensaje"),
                    ],
                ))
            return elementos, barra, badge

        @app.callback(
            Output("btn-calor", "children"),
            Output("btn-calor", "className"),
            Input("btn-calor", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_calor(_):
            self._heatmap_activo = not self._heatmap_activo
            if self._heatmap_activo:
                return "🌡  Calor", "btn-control activo"
            return "🌡  Calor", "btn-control"

        @app.callback(
            Output("btn-trayectorias", "children"),
            Output("btn-trayectorias", "className"),
            Input("btn-trayectorias", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_trayectorias(_):
            self._tracking_activo = not self._tracking_activo
            if self._tracking_activo:
                return "🛤  Trayectorias", "btn-control activo"
            return "🛤  Trayectorias", "btn-control"

        @app.callback(
            Output("btn-pausa", "children"),
            Output("btn-pausa", "className"),
            Input("btn-pausa", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_pausa(_):
            self._pausado = not self._pausado
            if self._capturador:
                self._capturador.pausado = self._pausado
            if self._simulador:
                gestor = self._simulador._gestor
                gestor.en_pausa = self._pausado
                if self._pausado:
                    for t in gestor._consecutivos:
                        gestor._consecutivos[t] = 0
            if self._pausado:
                return "▶  Reanudar", "btn-control activo"
            return "⏸  Pausar", "btn-control"

        @app.callback(
            Output("msg-captura", "children"),
            Input("btn-captura", "n_clicks"),
            prevent_initial_call=True,
        )
        def guardar_captura(_):
            with self._lock_frame:
                frame = self._ultimo_frame
            if frame is None:
                return "Sin frame disponible"
            self._carpeta_capturas.mkdir(parents=True, exist_ok=True)
            nombre = datetime.datetime.now().strftime("panel_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(str(self._carpeta_capturas / nombre), frame)
            logger.info(f"Captura guardada: {nombre}")
            return f"✓ {nombre}"

        @app.callback(
            Output("menu-abierto", "data"),
            Input("btn-simular-toggle", "n_clicks"),
            [Input(f"btn-sim-{tipo}", "n_clicks") for tipo, _ in _HITOS_SIMULABLES],
            Input("menu-backdrop", "n_clicks"),
            prevent_initial_call=True,
        )
        def actualizar_menu_abierto(n_toggle, *_resto):
            from dash import ctx
            if not ctx.triggered_id:
                return False
            if ctx.triggered_id == "btn-simular-toggle":
                return (n_toggle or 0) % 2 == 1
            return False

        @app.callback(
            Output("menu-backdrop", "style"),
            Input("menu-abierto", "data"),
        )
        def toggle_menu_backdrop(abierto):
            base = {"position": "fixed", "top": 0, "left": 0,
                    "width": "100vw", "height": "100vh", "zIndex": 199}
            return {**base, "display": "block" if abierto else "none"}

        @app.callback(
            Output("menu-simular", "style"),
            Input("menu-abierto", "data"),
            Input("menu-posicion", "data"),
        )
        def renderizar_menu(abierto, posicion):
            base = {
                "position": "absolute", "left": 0,
                "background": "#13132a", "border": "1px solid #2a2a4e",
                "borderRadius": "8px", "zIndex": 200, "minWidth": "220px",
                "padding": "6px 0", "boxShadow": "0 8px 24px rgba(0,0,0,0.6)",
                "display": "block" if abierto else "none",
            }
            if (posicion or "abajo") == "arriba":
                base["bottom"] = "calc(100% + 6px)"
            else:
                base["top"] = "calc(100% + 6px)"
            return base

        @app.callback(
            Output("msg-simular", "children"),
            [Input(f"btn-sim-{tipo}", "n_clicks") for tipo, _ in _HITOS_SIMULABLES],
            prevent_initial_call=True,
        )
        def lanzar_simulacion(*_):
            from dash import ctx
            if not ctx.triggered_id or self._simulador is None:
                return ""
            tipo = ctx.triggered_id.replace("btn-sim-", "")
            self._simulador.simular(tipo)
            etiqueta = next((e for t, e in _HITOS_SIMULABLES if t == tipo), tipo)
            logger.info(f"Simulación lanzada: {tipo}")
            return ""

        @app.callback(
            Output("hito-seleccionado", "data"),
            Input({"type": "hito-card", "index": ALL}, "n_clicks"),
            Input("drawer-backdrop", "n_clicks"),
            prevent_initial_call=True,
        )
        def seleccionar_hito(n_clicks_list, _backdrop):
            from dash import ctx, no_update
            if not ctx.triggered_id:
                return no_update
            if ctx.triggered_id == "drawer-backdrop":
                return None
            # Ignorar disparos por re-render (n_clicks == 0, no es un click real)
            if not ctx.triggered or ctx.triggered[0].get("value", 0) == 0:
                return no_update
            hito_id = ctx.triggered_id["index"]
            hitos = self._base_datos.hitos_recientes(limite=100)
            hito = next((h for h in hitos if h["id"] == hito_id), None)
            if hito is None:
                return no_update
            return {k: hito[k] for k in ("tipo", "descripcion", "razonamiento", "mensaje",
                                          "confirmado", "marca_tiempo", "marca_tiempo_deteccion",
                                          "acciones", "ruta_frame")}

        @app.callback(
            Output("cajita-detalle", "style"),
            Output("cajita-detalle-titulo", "children"),
            Output("cajita-detalle-cuerpo", "children"),
            Output("drawer-backdrop", "style"),
            Input("hito-seleccionado", "data"),
        )
        def mostrar_detalle(data):
            estilo_base = {
                "position": "fixed", "top": 0, "width": "380px", "height": "100vh",
                "background": "#0d0d1a", "borderLeft": "1px solid #2a2a3e",
                "zIndex": 1000, "padding": "24px 20px",
                "boxShadow": "-4px 0 20px rgba(0,0,0,0.5)",
                "transition": "right 0.3s ease", "overflowY": "auto",
            }
            backdrop_oculto = {"display": "none", "position": "fixed", "top": 0, "left": 0,
                               "width": "100vw", "height": "100vh", "zIndex": 999}
            backdrop_visible = {**backdrop_oculto, "display": "block"}
            if not data:
                return {**estilo_base, "right": "-400px"}, [], [], backdrop_oculto
            color = _COLORES_HITO.get(data["tipo"], _COLOR_HITO_DEFAULT)
            confirmado = data["confirmado"]

            def _ts(ts):
                if not ts:
                    return "—"
                return datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S")

            ts_deteccion = _ts(data.get("marca_tiempo_deteccion"))
            ts_verificacion = _ts(data.get("marca_tiempo"))

            titulo = [
                html.Div(data["tipo"].replace("_", " ").upper(),
                         style={"fontFamily": "Rajdhani, sans-serif", "fontSize": "1.2em",
                                "fontWeight": "700", "color": color}),
                html.Div("✓ CONFIRMADO" if confirmado else "✗ DESCARTADO",
                         style={"fontSize": "0.72em", "marginTop": "3px",
                                "color": "#00e676" if confirmado else "#ff5252"}),
            ]

            _s_label = {"fontSize": "0.66em", "color": "#555", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "marginBottom": "4px"}
            _s_bloque = {"marginBottom": "18px"}

            ruta_frame = data.get("ruta_frame")
            img_bloque = []
            if ruta_frame:
                nombre_archivo = Path(ruta_frame).name
                img_bloque = [html.Img(
                    src=f"/captura-img/{nombre_archivo}",
                    style={"width": "100%", "borderRadius": "6px", "display": "block",
                           "marginBottom": "16px", "border": f"1px solid {color}44"},
                )]

            cuerpo = [
                html.Hr(style={"borderColor": "#2a2a3e", "margin": "0 0 16px 0"}),
                *img_bloque,

                # — Bloque 1: Detección ——————————————————————————
                html.Div(style=_s_bloque, children=[
                    html.Div("Detección", style=_s_label),
                    html.Div(ts_deteccion,
                             style={"fontSize": "0.78em", "color": "#607080",
                                    "marginBottom": "6px", "fontVariantNumeric": "tabular-nums"}),
                    html.Div(data["descripcion"],
                             style={"fontSize": "0.88em", "color": "#b0b0d0", "lineHeight": "1.5"}),
                ]),

                # — Bloque 2: Verificación ————————————————————————
                html.Div(style=_s_bloque, children=[
                    html.Div("Verificación", style=_s_label),
                    html.Div(ts_verificacion,
                             style={"fontSize": "0.78em", "color": "#607080",
                                    "marginBottom": "6px", "fontVariantNumeric": "tabular-nums"}),
                    *(
                        [html.Div(data["mensaje"],
                                  style={"fontSize": "0.88em", "color": "#e0e0ff",
                                         "fontStyle": "italic", "padding": "10px 12px",
                                         "background": "#12122a", "borderRadius": "6px",
                                         "borderLeft": f"3px solid {color}", "lineHeight": "1.5"})]
                        if confirmado and data.get("mensaje")
                        else [html.Div("Hito descartado — no confirmado visualmente.",
                                       style={"fontSize": "0.85em", "color": "#607080",
                                              "fontStyle": "italic"})]
                    ),
                ]),
            ]

            # — Bloque 3: Pensamiento de Gemma ——————————————————————
            razonamiento = (data.get("razonamiento") or "").strip()
            if razonamiento:
                cuerpo.append(html.Div(style=_s_bloque, children=[
                    html.Div("Pensamiento de Gemma", style=_s_label),
                    html.Div(razonamiento,
                             style={"fontSize": "0.82em", "color": "#888", "lineHeight": "1.6",
                                    "padding": "8px 10px", "background": "#0a0a18",
                                    "borderRadius": "4px", "borderLeft": "2px solid #2a2a4e"}),
                ]))

            # — Bloque 4: Acciones disparadas —————————————————————
            _ETIQUETAS_ACCION = {
                "email":    ("📧", "Email",    "#ff9800"),
                "telegram": ("💬", "Telegram", "#29b6f6"),
                "captura":  ("📸", "Captura",  "#78909c"),
            }
            acciones_raw = data.get("acciones") or ""
            acciones_lista = [a for a in acciones_raw.split(",") if a]
            if acciones_lista:
                cuerpo.append(html.Div(style=_s_bloque, children=[
                    html.Div("Acciones", style=_s_label),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "2px"},
                        children=[
                            html.Span(
                                f"{ico} {etq}",
                                style={
                                    "fontSize": "0.78em", "padding": "3px 10px",
                                    "borderRadius": "12px", "color": col,
                                    "background": f"{col}18",
                                    "border": f"1px solid {col}44",
                                    "letterSpacing": "0.03em",
                                },
                            )
                            for a in acciones_lista
                            for ico, etq, col in [_ETIQUETAS_ACCION.get(a, ("•", a, "#607d8b"))]
                        ],
                    ),
                ]))

            return {**estilo_base, "right": "0px"}, titulo, cuerpo, backdrop_visible

        _estilo_modelo_base = {
            "position": "fixed", "top": 0, "width": "780px", "height": "100vh",
            "background": "#0d0d1a", "borderLeft": "1px solid #2a2a3e",
            "zIndex": 1000, "padding": "24px 20px",
            "boxShadow": "-4px 0 20px rgba(0,0,0,0.5)",
            "transition": "right 0.3s ease", "overflowY": "auto",
        }

        @app.callback(
            Output("cajita-modelo", "style"),
            Output("cajita-modelo-cuerpo", "children"),
            Output("drawer-backdrop-modelo", "style"),
            Input("btn-modelo", "n_clicks"),
            Input("btn-cerrar-modelo", "n_clicks"),
            Input("drawer-backdrop-modelo", "n_clicks"),
            State("cajita-modelo", "style"),
            prevent_initial_call=True,
        )
        def toggle_cajita_modelo(n_btn, _cerrar, _backdrop, estilo_actual):
            from dash import ctx, no_update
            backdrop_oculto = {"display": "none", "position": "fixed", "top": 0, "left": 0,
                               "width": "100vw", "height": "100vh", "zIndex": 999}
            backdrop_visible = {**backdrop_oculto, "display": "block"}
            abierto = estilo_actual is not None and estilo_actual.get("right") == "0px"
            if ctx.triggered_id == "btn-modelo" and not abierto:
                return ({**_estilo_modelo_base, "right": "0px"},
                        _construir_contenido_modelo(),
                        backdrop_visible)
            return {**_estilo_modelo_base, "right": "-800px"}, no_update, backdrop_oculto

        return app

    # ------------------------------------------------------------------

    def _fps(self) -> float:
        if len(self._tiempos_frame) < 2:
            return 0.0
        transcurrido = self._tiempos_frame[-1] - self._tiempos_frame[0]
        return (len(self._tiempos_frame) - 1) / transcurrido if transcurrido > 0 else 0.0

    def _anotar_frame(self, frame: np.ndarray, simulando: str | None = None) -> np.ndarray:
        frame = frame.copy()

        for nombre, zona in self._zonas.items():
            if nombre == "fauna":
                continue
            color = _COLORES_ZONA.get(nombre, _COLOR_ZONA_DEFAULT)
            pts = zona.poligono.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x, y = zona.poligono.polygon[0]
            cv2.putText(frame, nombre, (int(x) + 6, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detecciones = self._ultimo_tracking.detecciones if self._ultimo_tracking is not None else None

        h, w = frame.shape[:2]
        with self._lock_heatmap:
            if self._acum_heatmap is not None:
                self._acum_heatmap *= _HEATMAP_DECAY

        if detecciones is not None and len(detecciones) > 0:
            with self._lock_heatmap:
                if self._acum_heatmap is None or self._acum_heatmap.shape != (h, w):
                    self._acum_heatmap = np.zeros((h, w), dtype=np.float32)
                for bx0, by0, bx1, by1 in detecciones.xyxy:
                    cx = int((bx0 + bx1) / 2)
                    cy = int(by1)  # suelo donde pisa el personaje
                    cv2.circle(self._acum_heatmap, (cx, cy), _HEATMAP_RADIO, 2.0, -1)

        if detecciones is not None and len(detecciones) > 0 and detecciones.tracker_id is not None:
            with self._lock_trails:
                for i, tid in enumerate(detecciones.tracker_id):
                    tid_int = int(tid)
                    bx0, by0, bx1, by1 = detecciones.xyxy[i]
                    cx = int((bx0 + bx1) / 2)
                    cy = int(by1)
                    if tid_int not in self._trails:
                        self._trails[tid_int] = deque(maxlen=_TRAIL_LONGITUD)
                    self._trails[tid_int].append((cx, cy))

        if self._heatmap_activo:
            with self._lock_heatmap:
                acum = self._acum_heatmap.copy() if self._acum_heatmap is not None else None
            if acum is not None:
                suavizado = cv2.GaussianBlur(acum, (61, 61), 0)
                norm = np.clip(suavizado / _HEATMAP_CAP * 255, 0, 255).astype(np.uint8)
                coloreado = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.6, coloreado, 0.4, 0)

        if self._tracking_activo:
            with self._lock_trails:
                trails_copia = {tid: list(pts) for tid, pts in self._trails.items() if len(pts) >= 2}
            for tid, pts in trails_copia.items():
                color = _color_por_id(tid)
                n = len(pts)
                for i in range(1, n):
                    alpha = i / n
                    color_fade = tuple(int(c * alpha) for c in color)
                    grosor = max(1, int(_TRAIL_GROSOR * (0.4 + 0.6 * alpha)))
                    cv2.line(frame, pts[i - 1], pts[i], color_fade, grosor, cv2.LINE_AA)

        if detecciones is not None and len(detecciones) > 0:
            nombres_clase = detecciones.data.get("class_name", np.array([]))
            tracker_ids = detecciones.tracker_id
            etiquetas = []
            for i, conf in enumerate(detecciones.confidence):
                nombre = nombres_clase[i] if i < len(nombres_clase) else str(detecciones.class_id[i])
                tid = f" #{int(tracker_ids[i])}" if tracker_ids is not None else ""
                etiquetas.append(f"{nombre} {conf:.0%}{tid}")
            frame = self._anotador_cajas.annotate(scene=frame, detections=detecciones)
            frame = self._anotador_etiquetas.annotate(scene=frame, detections=detecciones, labels=etiquetas)

        if simulando:
            self._dibujar_banner_simulacion(frame, simulando)

        self._dibujar_panel_stats(frame, detecciones)
        return frame

    def _dibujar_banner_simulacion(self, frame: np.ndarray, tipo: str) -> None:
        texto = f"SIMULACION: {tipo.replace('_', ' ').upper()}"
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, y0 = (w - tw) // 2 - 12, h - 50
        x1, y1 = x0 + tw + 24, y0 + th + 16
        region = frame[y0:y1, x0:x1]
        fondo = np.zeros_like(region)
        fondo[:] = (0, 120, 220)
        cv2.addWeighted(fondo, 0.75, region, 0.25, 0, region)
        frame[y0:y1, x0:x1] = region
        cv2.putText(frame, texto, (x0 + 12, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def _dibujar_panel_stats(self, frame: np.ndarray, detecciones: sv.Detections | None) -> None:
        conteo: dict[str, int] = {}
        confianza_media = 0.0
        if detecciones is not None and len(detecciones) > 0:
            nombres = detecciones.data.get("class_name", np.array([]))
            for nombre in nombres:
                conteo[nombre] = conteo.get(nombre, 0) + 1
            confianza_media = float(np.mean(detecciones.confidence))

        lineas = [f"FPS: {self._fps():.1f}", f"Objetos: {sum(conteo.values())}"]
        for clase, n in sorted(conteo.items()):
            lineas.append(f"  {clase}: {n}")
        if confianza_media > 0:
            lineas.append(f"Conf. media: {confianza_media:.0%}")

        alto_linea, margen, padding, ancho_panel = 22, 10, 8, 180
        alto_panel = len(lineas) * alto_linea + padding * 2
        x0, y0 = margen, margen
        x1, y1 = x0 + ancho_panel, y0 + alto_panel

        region = frame[y0:y1, x0:x1]
        fondo = np.zeros_like(region)
        cv2.addWeighted(fondo, 0.6, region, 0.4, 0, region)
        frame[y0:y1, x0:x1] = region

        for i, linea in enumerate(lineas):
            y_texto = y0 + padding + (i + 1) * alto_linea - 4
            cv2.putText(frame, linea, (x0 + padding, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------

def _color_por_id(tracker_id: int) -> tuple[int, int, int]:
    h = (tracker_id * 47) % 180
    color_hsv = np.uint8([[[h, 220, 220]]])
    bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _cargar_resultados_csv() -> dict[str, list]:
    ruta = _DIR_ENTRENAMIENTO / "results.csv"
    if not ruta.exists():
        return {}
    datos: dict[str, list] = {}
    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            for k, v in fila.items():
                try:
                    val: float | None = float(v)
                except (ValueError, TypeError):
                    val = None
                datos.setdefault(k.strip(), []).append(val)
    return datos


def _construir_contenido_modelo() -> list:
    datos = _cargar_resultados_csv()
    epocas = datos.get("epoch", [])
    n_epocas = len(epocas)

    mAP50_final    = (datos.get("metrics/mAP50(B)",    [None])[-1] or 0)
    mAP5095_final  = (datos.get("metrics/mAP50-95(B)", [None])[-1] or 0)
    precision_final = (datos.get("metrics/precision(B)", [None])[-1] or 0)
    recall_final    = (datos.get("metrics/recall(B)",    [None])[-1] or 0)

    _estilo_grafica = dict(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#13132a",
        margin=dict(l=42, r=10, t=28, b=28),
        font=dict(color="#c0c0e0", size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1e1e32", zerolinecolor="#1e1e32", title="época"),
        yaxis=dict(gridcolor="#1e1e32", zerolinecolor="#1e1e32"),
    )

    fig_metricas = go.Figure()
    fig_metricas.add_trace(go.Scatter(
        x=epocas, y=datos.get("metrics/mAP50(B)", []),
        name="mAP50", line=dict(color="#00e676", width=2)))
    fig_metricas.add_trace(go.Scatter(
        x=epocas, y=datos.get("metrics/precision(B)", []),
        name="Precisión", line=dict(color="#ff9800", width=1.5, dash="dot")))
    fig_metricas.add_trace(go.Scatter(
        x=epocas, y=datos.get("metrics/recall(B)", []),
        name="Recall", line=dict(color="#f44336", width=1.5, dash="dot")))
    fig_metricas.update_layout(**_estilo_grafica,
                               title=dict(text="Métricas por época", font=dict(size=11)))
    fig_metricas.update_yaxes(range=[0, 1.05])

    fig_losses = go.Figure()
    fig_losses.add_trace(go.Scatter(
        x=epocas, y=datos.get("train/cls_loss", []),
        name="cls train", line=dict(color="#00bcd4", width=2)))
    fig_losses.add_trace(go.Scatter(
        x=epocas, y=datos.get("val/cls_loss", []),
        name="cls val", line=dict(color="#e040fb", width=2)))
    fig_losses.add_trace(go.Scatter(
        x=epocas, y=datos.get("train/box_loss", []),
        name="box train", line=dict(color="#00bcd4", width=1.5, dash="dot")))
    fig_losses.add_trace(go.Scatter(
        x=epocas, y=datos.get("val/box_loss", []),
        name="box val", line=dict(color="#e040fb", width=1.5, dash="dot")))
    fig_losses.update_layout(**_estilo_grafica,
                             title=dict(text="Pérdidas train vs val", font=dict(size=11)))

    clases_ord = sorted(_MAP50_POR_CLASE.items(), key=lambda x: x[1])
    nombres_c = [c for c, _ in clases_ord]
    valores_c = [v for _, v in clases_ord]
    colores_c = ["#00e676" if v >= 0.9 else "#ff9800" if v >= 0.75 else "#f44336"
                 for v in valores_c]
    fig_clases = go.Figure(go.Bar(
        x=valores_c, y=nombres_c, orientation="h",
        marker=dict(color=colores_c),
        text=[f"{v:.3f}" for v in valores_c],
        textposition="outside",
        textfont=dict(size=9, color="#c0c0e0"),
    ))
    fig_clases.update_layout(**_estilo_grafica,
                             title=dict(text="mAP50 por clase", font=dict(size=11)),
                             height=290)
    fig_clases.update_layout(margin=dict(l=110, r=40, t=28, b=28))
    fig_clases.update_xaxes(range=[0, 1.08])

    _s_label = {"fontSize": "0.62em", "color": "#666", "textTransform": "uppercase",
                "letterSpacing": "0.07em", "marginBottom": "2px"}
    _s_seccion = {"fontSize": "0.68em", "color": "#555", "textTransform": "uppercase",
                  "letterSpacing": "0.08em", "marginBottom": "8px"}

    return [
        html.Hr(style={"borderColor": "#2a2a3e", "margin": "0 0 12px 0"}),
        html.Div(
            style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                   "fontSize": "0.72em", "color": "#666", "marginBottom": "14px"},
            children=[
                html.Span(f"yolo26s · {n_epocas}/100 épocas"),
                html.Span("imgsz 1280 · batch 8"),
                html.Span("11 clases · 499 imágenes"),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                   "gap": "8px", "marginBottom": "20px"},
            children=[
                _mini_card("mAP50",    f"{mAP50_final:.3f}",    "#00e676"),
                _mini_card("mAP50-95", f"{mAP5095_final:.3f}",  "#00bcd4"),
                _mini_card("Precisión", f"{precision_final:.3f}", "#ff9800"),
                _mini_card("Recall",   f"{recall_final:.3f}",   "#f44336"),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px",
                   "marginBottom": "14px"},
            children=[
                dcc.Graph(figure=fig_metricas, config={"displayModeBar": False},
                          style={"height": "240px"}),
                dcc.Graph(figure=fig_losses, config={"displayModeBar": False},
                          style={"height": "240px"}),
            ],
        ),
        dcc.Graph(figure=fig_clases, config={"displayModeBar": False},
                  style={"height": "300px", "marginBottom": "16px"}),
        html.Div("Matriz de confusión normalizada", style=_s_seccion),
        html.Img(src="/modelo-img/confusion_matrix_normalized.png",
                 style={"width": "100%", "borderRadius": "4px",
                        "border": "1px solid #2a2a3e"}),
    ]


def _mini_card(etiqueta: str, valor: str, color: str) -> html.Div:
    return html.Div(
        style={"background": "#13132a", "borderRadius": "6px",
               "padding": "8px 4px", "textAlign": "center"},
        children=[
            html.Div(etiqueta, style={"fontSize": "0.62em", "color": "#666",
                                      "textTransform": "uppercase", "letterSpacing": "0.07em"}),
            html.Div(valor, style={"fontSize": "1.1em", "fontFamily": "Rajdhani, sans-serif",
                                   "fontWeight": "700", "color": color}),
        ],
    )


def _stat_card(numero: str, etiqueta: str, color: str) -> html.Div:
    return html.Div(
        className="stat-card",
        children=[
            html.Div(etiqueta, className="stat-label"),
            html.Div(numero, className="stat-numero", style={"color": color}),
        ],
    )


