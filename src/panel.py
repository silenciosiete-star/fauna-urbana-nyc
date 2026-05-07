"""Panel web: stream MJPEG en directo + historial de hitos."""
import datetime
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dash import ALL, Dash, Input, Output, dcc, html
from flask import Response
from loguru import logger

from .base_datos import BaseDatos
from .rastreador import ResultadoTracking
from .simulador import Simulador
from .verificador import Verificador
from .zonas import Zona

_HITOS_SIMULABLES = [
    ("avengers_assemble", "Avengers Assemble"),
    ("marvel_vs_dc",      "Marvel vs DC"),
    ("hora_punta",        "Hora Punta"),
    ("conflicto_identidad", "Conflicto de Identidad"),
    ("avistamiento_raro", "Avistamiento Raro"),
]

_ANCHO_STREAM = 960
_INTERVALO_REFRESCO_MS = 2000
_VENTANA_FPS = 30
_COLORES_ZONA = {
    "fauna": (0, 200, 0),
    "esquina_norte": (0, 200, 220),
    "esquina_sur": (200, 0, 200),
}
_COLOR_ZONA_DEFAULT = (180, 180, 180)
_COLORES_HITO = {
    "avengers_assemble": "#ff9800",
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
        self._hilo_frames: threading.Thread | None = None
        self._ultimo_tracking: ResultadoTracking | None = None
        self._ultimo_frame: np.ndarray | None = None
        self._lock_frame = threading.Lock()
        self._tiempos_frame: deque = deque(maxlen=_VENTANA_FPS)
        self._anotador_cajas = sv.BoxAnnotator()
        self._anotador_etiquetas = sv.LabelAnnotator()
        self._app = self._crear_app()

    def conectar_simulador(self, simulador: Simulador) -> None:
        self._simulador = simulador

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
            if not self._pausado:
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
                # ── Header ──────────────────────────────────────────
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "20px"},
                    children=[
                        html.Span("Fauna Urbana NYC", className="titulo"),
                        html.Div(className="live-badge", children=[html.Div(className="live-dot"), "LIVE"]),
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
                # ── Backdrop (cierra el drawer al hacer click fuera) ──
                html.Div(
                    id="drawer-backdrop",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 999},
                ),
                # ── Drawer lateral derecho ───────────────────────────
                html.Div(
                    id="cajita-detalle",
                    style={**_estilo_drawer_base, "right": "-400px"},
                    children=[
                        html.Div(id="cajita-detalle-titulo", style={"marginBottom": "16px"}),
                        html.Div(id="cajita-detalle-cuerpo"),
                    ],
                ),
            ],
        )

        # ── Callbacks ────────────────────────────────────────────────

        @app.callback(
            Output("lista-hitos", "children"),
            Output("barra-stats", "children"),
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
                return html.P("Sin hitos registrados aún.", style={"color": "#404060", "fontSize": "0.85em"}), barra
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
            return elementos, barra

        @app.callback(
            Output("btn-pausa", "children"),
            Output("btn-pausa", "className"),
            Input("btn-pausa", "n_clicks"),
            prevent_initial_call=True,
        )
        def toggle_pausa(_):
            self._pausado = not self._pausado
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
            prevent_initial_call=True,
        )
        def actualizar_menu_abierto(_, *n_sims):
            from dash import ctx
            if not ctx.triggered_id:
                return False
            if ctx.triggered_id == "btn-simular-toggle":
                return not bool(_ and _ % 2 == 1) if _ is None else _ % 2 == 1
            # Cualquier botón de simulación cierra el menú
            return False

        @app.callback(
            Output("menu-simular", "style"),
            Input("menu-abierto", "data"),
        )
        def renderizar_menu(abierto):
            base = {
                "position": "absolute", "top": "calc(100% + 6px)", "left": 0,
                "background": "#13132a", "border": "1px solid #2a2a4e",
                "borderRadius": "8px", "zIndex": 200, "minWidth": "220px",
                "padding": "6px 0", "boxShadow": "0 8px 24px rgba(0,0,0,0.6)",
            }
            return {**base, "display": "block" if abierto else "none"}

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
            return f"▶ {etiqueta}"

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
            return {k: hito[k] for k in ("tipo", "descripcion", "razonamiento", "mensaje", "confirmado")}

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
            titulo = [
                html.Div(data["tipo"].replace("_", " ").upper(),
                         style={"fontFamily": "Rajdhani, sans-serif", "fontSize": "1.2em",
                                "fontWeight": "700", "color": color}),
                html.Div("✓ CONFIRMADO" if data["confirmado"] else "✗ SIN CONFIRMAR",
                         style={"fontSize": "0.72em", "marginTop": "3px",
                                "color": "#00e676" if data["confirmado"] else "#ff5252"}),
            ]
            cuerpo = [
                html.Hr(style={"borderColor": "#2a2a3e", "margin": "0 0 16px 0"}),
                html.Div("Condición detectada",
                         style={"fontSize": "0.68em", "color": "#555", "textTransform": "uppercase",
                                "letterSpacing": "0.08em", "marginBottom": "5px"}),
                html.Div(data["descripcion"],
                         style={"fontSize": "0.88em", "color": "#b0b0d0", "marginBottom": "16px"}),
            ]
            if data["confirmado"] and data.get("mensaje"):
                cuerpo += [
                    html.Div("Mensaje Gemma",
                             style={"fontSize": "0.68em", "color": "#555", "textTransform": "uppercase",
                                    "letterSpacing": "0.08em", "marginBottom": "5px"}),
                    html.Div(data["mensaje"],
                             style={"fontSize": "0.88em", "color": "#e0e0ff", "fontStyle": "italic",
                                    "padding": "10px 12px", "background": "#12122a", "borderRadius": "6px",
                                    "borderLeft": f"3px solid {color}", "marginBottom": "16px",
                                    "lineHeight": "1.5"}),
                ]
            razonamiento = data.get("razonamiento") or ""
            if razonamiento and razonamiento not in ("FALSO_POSITIVO", ""):
                label = "Razonamiento Gemma" if data["confirmado"] else "Error de verificación"
                cuerpo += [
                    html.Div(label,
                             style={"fontSize": "0.68em", "color": "#555", "textTransform": "uppercase",
                                    "letterSpacing": "0.08em", "marginBottom": "5px"}),
                    html.Div(razonamiento,
                             style={"fontSize": "0.82em", "color": "#888", "lineHeight": "1.6"}),
                ]
            return {**estilo_base, "right": "0px"}, titulo, cuerpo, backdrop_visible

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

def _stat_card(numero: str, etiqueta: str, color: str) -> html.Div:
    return html.Div(
        className="stat-card",
        children=[
            html.Div(etiqueta, className="stat-label"),
            html.Div(numero, className="stat-numero", style={"color": color}),
        ],
    )
