"""Panel web: stream MJPEG en directo + historial de hitos."""
import base64
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
        self._notificador = None
        self._gestor_eventos = None
        self._frame_editor: np.ndarray | None = None
        self._dim_frame_original: tuple[int, int] = (1920, 1080)
        self._lock_editor = threading.Lock()
        self._zonas_custom: dict | None = None
        self._formas_editor: list = []
        self._app = self._crear_app()

    def conectar_simulador(self, simulador: Simulador) -> None:
        self._simulador = simulador
        self._capturador = simulador._capturador

    def conectar_verificador(self, verificador: Verificador) -> None:
        self._verificador = verificador

    def conectar_notificador(self, notificador) -> None:
        self._notificador = notificador

    def conectar_gestor_eventos(self, gestor) -> None:
        self._gestor_eventos = gestor

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
            # Guardar frame crudo (sin anotar) para el editor de zonas
            if self._simulador is None or self._simulador.frame_override is None:
                alto_ed = int(frame.shape[0] * _ANCHO_STREAM / frame.shape[1])
                frame_ed = cv2.resize(frame, (_ANCHO_STREAM, alto_ed))
                with self._lock_editor:
                    self._frame_editor = frame_ed
                    self._dim_frame_original = (frame.shape[1], frame.shape[0])
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

        @app.server.route("/audio/<nombre>")
        def servir_audio(nombre):
            from flask import send_from_directory
            return send_from_directory(str(self._carpeta_capturas.resolve()), nombre)

        @app.server.route("/frame-zona")
        def servir_frame_zona():
            with self._lock_editor:
                frame = self._frame_editor
            if frame is None:
                return Response(status=204)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return Response(buf.tobytes(), mimetype="image/jpeg")

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
                dcc.Store(id="audio-url-hito", data=None),
                dcc.Store(id="cola-audio-nuevos", data=[]),
                dcc.Store(id="zonas-custom-activas", data=False),
                dcc.Store(id="zona-editor-formas", data=[]),
                dcc.Store(id="zona-editor-info", data=None),
                dcc.Store(id="editor-init-cmd", data=None),
                dcc.Store(id="galeria-abierta", data=False),
                dcc.Store(id="galeria-filtro", data={"categoria": "todas", "hito": None}),
                dcc.Store(id="galeria-img-ampliada", data=None),
                html.Audio(id="audio-tts", style={"display": "none"}, preload="auto"),
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
                                        html.Button("🖼  Galería", id="btn-galeria", className="btn-control", n_clicks=0),
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
                                        html.Button("↗  Trayectorias", id="btn-trayectorias", className="btn-control", n_clicks=0),
                                        html.Button("✏️  Zona Custom", id="btn-zonas", className="btn-control", n_clicks=0),
                                        html.Span(id="msg-captura", style={"display": "none"}),
                                        html.Span(id="msg-simular", className="msg-control"),
                                    ],
                                ),
                            ],
                        ),
                        # Columna hitos
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "marginBottom": "0"},
                                    children=[
                                        html.Div("Hitos recientes", className="seccion-titulo",
                                                 style={"marginBottom": "0", "flex": "1"}),
                                        html.Button("🔊", id="btn-audio-auto", n_clicks=0,
                                                    style={"background": "none", "border": "none",
                                                           "cursor": "pointer", "fontSize": "1.1em",
                                                           "opacity": "1", "padding": "0 4px",
                                                           "lineHeight": "1", "title": "Audio automático"}),
                                    ],
                                ),
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
                        html.Button(
                            "🔊 Leer",
                            id="btn-leer-voz",
                            n_clicks=0,
                            style={"display": "none", "marginTop": "12px", "width": "100%",
                                   "padding": "8px", "background": "#7c4dff22",
                                   "border": "1px solid #7c4dff66", "borderRadius": "6px",
                                   "color": "#b39ddb", "cursor": "pointer", "fontSize": "0.85em"},
                        ),
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
                # ── Backdrop zonas ───────────────────────────────────
                html.Div(
                    id="drawer-backdrop-zonas",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 999},
                ),
                # ── Drawer lateral derecho: editor de zonas ──────────
                html.Div(
                    id="cajita-zonas",
                    style={**_estilo_drawer_base, "right": "-800px", "width": "780px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "8px"},
                            children=[
                                html.Div("ZONAS — Conflicto de Identidad",
                                         style={"fontFamily": "Rajdhani, sans-serif",
                                                "fontSize": "1.1em", "fontWeight": "700",
                                                "color": "#e040fb"}),
                                html.Button("✕", id="btn-cerrar-zonas", n_clicks=0,
                                            style={"background": "none", "border": "none",
                                                   "color": "#555", "cursor": "pointer",
                                                   "fontSize": "1.2em", "lineHeight": "1"}),
                            ],
                        ),
                        html.Div(
                            "Dibuja hasta 2 rectángulos sobre el frame para definir las zonas. "
                            "El hito «Conflicto de Identidad» solo se disparará cuando dos "
                            "personajes iguales coincidan dentro de la misma zona.",
                            style={"fontSize": "0.72em", "color": "#4a4a7a",
                                   "marginBottom": "10px", "lineHeight": "1.5"},
                        ),
                        html.Div(id="zona-contador",
                                 children="0/2 zonas dibujadas",
                                 style={"fontSize": "0.75em", "fontWeight": "600",
                                        "color": "#607d8b", "marginBottom": "8px"}),
                        html.Div(
                            id="editor-contenedor",
                            style={"position": "relative", "lineHeight": "0",
                                   "background": "#07070f", "borderRadius": "4px",
                                   "overflow": "hidden", "minHeight": "200px"},
                            children=[
                                html.Img(id="editor-img", src="",
                                         style={"display": "block", "width": "100%",
                                                "userSelect": "none", "pointerEvents": "none"}),
                                html.Canvas(id="editor-canvas", width=960, height=540,
                                            style={"position": "absolute", "top": "0", "left": "0",
                                                   "width": "100%", "height": "100%",
                                                   "cursor": "crosshair"}),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginTop": "10px",
                                   "alignItems": "center"},
                            children=[
                                html.Button("✓ Confirmar zonas", id="btn-confirmar-zonas",
                                            className="btn-control activo", n_clicks=0),
                                html.Button("Por defecto", id="btn-reset-zonas",
                                            className="btn-control", n_clicks=0,
                                            title="Restaurar zonas del fichero config.yaml y cerrar"),
                                html.Button("✕ Limpiar", id="btn-limpiar-zonas",
                                            className="btn-control", n_clicks=0,
                                            title="Borrar rectángulos dibujados"),
                                html.Span(id="msg-zonas", className="msg-control"),
                            ],
                        ),
                    ],
                ),
                # ── Backdrop galería ─────────────────────────────────
                html.Div(
                    id="drawer-backdrop-galeria",
                    n_clicks=0,
                    style={"display": "none", "position": "fixed", "top": 0, "left": 0,
                           "width": "100vw", "height": "100vh", "zIndex": 999},
                ),
                # ── Drawer lateral derecho: galería de capturas ──────
                html.Div(
                    id="cajita-galeria",
                    style={**_estilo_drawer_base, "right": "-820px", "width": "800px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "12px"},
                            children=[
                                html.Div("GALERÍA DE CAPTURAS",
                                         style={"fontFamily": "Rajdhani, sans-serif",
                                                "fontSize": "1.1em", "fontWeight": "700",
                                                "color": "#00bcd4"}),
                                html.Button("✕", id="btn-cerrar-galeria", n_clicks=0,
                                            style={"background": "none", "border": "none",
                                                   "color": "#555", "cursor": "pointer",
                                                   "fontSize": "1.2em", "lineHeight": "1"}),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "6px", "marginBottom": "8px"},
                            children=[
                                html.Button("Todas", id="btn-filtro-galeria-todas",
                                            className="btn-control activo", n_clicks=0),
                                html.Button("Manuales", id="btn-filtro-galeria-manual",
                                            className="btn-control", n_clicks=0),
                                html.Button("Automáticas", id="btn-filtro-galeria-auto",
                                            className="btn-control", n_clicks=0),
                            ],
                        ),
                        html.Div(
                            id="galeria-filtros-hito",
                            style={"display": "none", "gap": "6px", "flexWrap": "wrap",
                                   "marginBottom": "10px"},
                            children=[
                                html.Button("Todos", id="btn-filtro-galeria-hito-todos",
                                            className="btn-control activo", n_clicks=0),
                                *[
                                    html.Button(
                                        etiqueta,
                                        id=f"btn-filtro-galeria-hito-{tipo}",
                                        className="btn-control",
                                        n_clicks=0,
                                        style={"borderLeft": f"3px solid {_COLORES_HITO.get(tipo, _COLOR_HITO_DEFAULT)}"},
                                    )
                                    for tipo, etiqueta in _HITOS_SIMULABLES
                                ],
                            ],
                        ),
                        html.Div(id="galeria-contenido"),
                    ],
                ),
                # ── Lightbox pantalla completa ───────────────────────
                html.Div(
                    id="galeria-lightbox",
                    style={"display": "none", "position": "fixed", "inset": "0",
                           "zIndex": 2000},
                    children=[
                        html.Div(
                            id="galeria-lightbox-backdrop",
                            n_clicks=0,
                            style={"position": "absolute", "inset": "0",
                                   "background": "rgba(0,0,0,0.92)", "cursor": "zoom-out"},
                        ),
                        html.Button(
                            "✕", id="btn-cerrar-lightbox", n_clicks=0,
                            style={"position": "absolute", "top": "20px", "right": "24px",
                                   "background": "none", "border": "none",
                                   "color": "#888", "cursor": "pointer",
                                   "fontSize": "1.5em", "lineHeight": "1", "zIndex": 2001},
                        ),
                        html.Div(
                            style={"position": "absolute", "inset": "0", "display": "flex",
                                   "flexDirection": "column", "alignItems": "center",
                                   "justifyContent": "center", "padding": "60px 40px",
                                   "pointerEvents": "none"},
                            children=[
                                html.Img(
                                    id="lightbox-img", src="",
                                    style={"maxWidth": "100%", "maxHeight": "80vh",
                                           "objectFit": "contain", "borderRadius": "6px",
                                           "display": "block", "pointerEvents": "all"},
                                ),
                                html.Div(id="lightbox-info",
                                         style={"marginTop": "14px", "textAlign": "center",
                                                "pointerEvents": "none"}),
                            ],
                        ),
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

            def _tarjeta_proceso(p: dict, etiqueta: str) -> html.Div:
                color = _COLORES_HITO.get(p["tipo"], _COLOR_HITO_DEFAULT)
                return html.Div(
                    className="hito-card",
                    style={"borderLeftColor": color, "opacity": "0.75"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Span("⏳", style={"fontSize": "0.9em"}),
                                html.Span(p["tipo"].replace("_", " ").upper(),
                                          className="hito-tipo", style={"color": color}),
                                html.Span(etiqueta,
                                          style={"marginLeft": "auto", "fontSize": "0.72em",
                                                 "color": "#888", "fontStyle": "italic"}),
                            ],
                        ),
                        html.Div(p.get("descripcion", ""), className="hito-mensaje"),
                    ],
                )

            if self._notificador:
                for p in self._notificador.hitos_preparando():
                    elementos.append(_tarjeta_proceso(p, "notificando..."))
            if self._verificador:
                for p in self._verificador.hitos_en_proceso():
                    elementos.append(_tarjeta_proceso(p, "verificando..."))
            if not hitos and not elementos:
                return html.P("Sin hitos registrados aún.", style={"color": "#404060", "fontSize": "0.85em"}), barra, badge
            for h in hitos[:30]:
                ts = datetime.datetime.fromtimestamp(h["marca_tiempo"]).strftime("%d/%m %H:%M")
                color_borde = _COLORES_HITO.get(h["tipo"], _COLOR_HITO_DEFAULT)
                color_estado = "#00e676" if h["confirmado"] else "#ff5252"
                estado_txt = "✓" if h["confirmado"] else "✗"
                icono_audio = []
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
                                *icono_audio,
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
                return "↗  Trayectorias", "btn-control activo"
            return "↗  Trayectorias", "btn-control"

        app.clientside_callback(
            """
            function(n_clicks) {
                var activo = (window._ttsAutoplay !== false);
                if (n_clicks) {
                    window._ttsAutoplay = !activo;
                    activo = window._ttsAutoplay;
                }
                return [activo ? '🔊' : '🔇', {'background': 'none', 'border': 'none',
                    'cursor': 'pointer', 'fontSize': '1.1em', 'padding': '0 4px', 'lineHeight': '1'}];
            }
            """,
            Output("btn-audio-auto", "children"),
            Output("btn-audio-auto", "style"),
            Input("btn-audio-auto", "n_clicks"),
            prevent_initial_call=False,
        )

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
            return ""

        app.clientside_callback(
            """
            function(n_clicks) {
                if (!n_clicks) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
                setTimeout(function() {
                    var btn = document.getElementById('btn-captura');
                    if (btn) {
                        btn.textContent = '📸  Captura';
                        btn.className = 'btn-control';
                    }
                }, 2000);
                return ['✓ Guardada', 'btn-control activo'];
            }
            """,
            Output("btn-captura", "children"),
            Output("btn-captura", "className"),
            Input("btn-captura", "n_clicks"),
            prevent_initial_call=True,
        )

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
                                          "acciones", "errores", "ruta_frame")}

        @app.callback(
            Output("cajita-detalle", "style"),
            Output("cajita-detalle-titulo", "children"),
            Output("cajita-detalle-cuerpo", "children"),
            Output("drawer-backdrop", "style"),
            Output("btn-leer-voz", "style"),
            Output("audio-url-hito", "data"),
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
            _btn_leer_oculto = {"display": "none"}
            _btn_leer_visible = {
                "display": "block", "marginTop": "12px", "width": "100%",
                "padding": "8px", "background": "#7c4dff22",
                "border": "1px solid #7c4dff66", "borderRadius": "6px",
                "color": "#b39ddb", "cursor": "pointer", "fontSize": "0.85em",
            }
            if not data:
                return {**estilo_base, "right": "-400px"}, [], [], backdrop_oculto, _btn_leer_oculto, None
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
                    id="detalle-frame-img",
                    src=f"/captura-img/{nombre_archivo}",
                    n_clicks=0,
                    style={"width": "100%", "borderRadius": "6px", "display": "block",
                           "marginBottom": "16px", "border": f"1px solid {color}44",
                           "cursor": "zoom-in"},
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
                "voz":      ("🔊", "Voz",      "#7c4dff"),
            }
            acciones_raw = data.get("acciones") or ""
            acciones_lista = [a for a in acciones_raw.split(",") if a]
            errores_lista = [a for a in (data.get("errores") or "").split(",") if a]
            if acciones_lista:
                def _badge_accion(a: str) -> html.Span:
                    ico, etq, col = _ETIQUETAS_ACCION.get(a, ("•", a, "#607d8b"))
                    if a in errores_lista:
                        return html.Span(
                            f"✗ {etq}",
                            title="Error al ejecutar esta acción",
                            style={
                                "fontSize": "0.78em", "padding": "3px 10px",
                                "borderRadius": "12px", "color": "#ff5252",
                                "background": "#ff525218",
                                "border": "1px solid #ff525244",
                                "letterSpacing": "0.03em",
                            },
                        )
                    return html.Span(
                        f"{ico} {etq}",
                        style={
                            "fontSize": "0.78em", "padding": "3px 10px",
                            "borderRadius": "12px", "color": col,
                            "background": f"{col}18",
                            "border": f"1px solid {col}44",
                            "letterSpacing": "0.03em",
                        },
                    )
                cuerpo.append(html.Div(style=_s_bloque, children=[
                    html.Div("Acciones", style=_s_label),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "2px"},
                        children=[_badge_accion(a) for a in acciones_lista],
                    ),
                ]))

            acciones_lista = [a for a in (data.get("acciones") or "").split(",") if a]
            audio_url_hito = (
                f"/audio/audio_{int(data['marca_tiempo'] * 1000)}.wav"
                if confirmado and "voz" in acciones_lista else None
            )
            btn_leer_style = _btn_leer_visible if audio_url_hito else _btn_leer_oculto
            return {**estilo_base, "right": "0px"}, titulo, cuerpo, backdrop_visible, btn_leer_style, audio_url_hito

        # Botón "Leer": reproduce el audio del hito abierto (gesto de usuario → autoplay permitido).
        # Toggle: si ya suena, lo para; si no, lo inicia.
        app.clientside_callback(
            """
            function(n_clicks, audio_url) {
                if (!n_clicks || !audio_url) return window.dash_clientside.no_update;
                var audio = document.getElementById('audio-tts');
                var btn   = document.getElementById('btn-leer-voz');
                if (!audio || !btn) return window.dash_clientside.no_update;
                if (!audio.paused) {
                    audio.pause();
                    audio.currentTime = 0;
                    btn.textContent = '🔊 Leer';
                    window._ttsTs = null;
                    return window.dash_clientside.no_update;
                }
                audio.src = audio_url + '?cb2=' + Date.now();
                btn.textContent = '⏳ Reproduciendo...';
                audio.addEventListener('canplay', function() {
                    audio.play().catch(function() {
                        btn.textContent = '🔊 Leer';
                        window._ttsTs = null;
                    });
                }, { once: true });
                audio.addEventListener('ended', function() {
                    btn.textContent = '🔊 Leer';
                    window._ttsTs = null;
                }, { once: true });
                audio.load();
                return window.dash_clientside.no_update;
            }
            """,
            Output("btn-leer-voz", "children"),
            Input("btn-leer-voz", "n_clicks"),
            State("audio-url-hito", "data"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output("cola-audio-nuevos", "data"),
            Input("intervalo", "n_intervals"),
            prevent_initial_call=True,
        )
        def actualizar_cola_audio(_):
            from dash import no_update
            if not self._notificador:
                return no_update
            nuevos = self._notificador.vaciar_cola_audio()
            if not nuevos:
                return no_update
            return [f"/audio/{n}" for n in nuevos]

        # Cola de reproducción automática.
        # Input: nuevas URLs del servidor. Output: audio-tts.src (sin conflicto).
        # window._ttsPlayNext se define una sola vez y es reutilizable desde el evento ended.
        app.clientside_callback(
            """
            function(nuevas_urls) {
                if (!window._audioQueue) window._audioQueue = [];

                if (!window._ttsPlayNext) {
                    window._ttsPlayNext = function() {
                        var audio = document.getElementById('audio-tts');
                        if (!audio || audio._ttsPlaying) return;
                        if (window._ttsAutoplay === false) return;
                        if (window._audioQueue.length === 0) return;
                        var url = window._audioQueue.shift();
                        var m = url.match(/audio_(\\d+)\\.wav/);
                        window._ttsTs = m ? parseInt(m[1]) / 1000.0 : null;
                        audio._ttsPlaying = true;
                        audio.src = url + '?cb=' + Date.now();
                        audio.addEventListener('canplay', function() {
                            audio.play().catch(function() {
                                audio._ttsPlaying = false;
                                window._ttsTs = null;
                                window._ttsPlayNext();
                            });
                        }, { once: true });
                        audio.addEventListener('ended', function() {
                            audio._ttsPlaying = false;
                            window._ttsTs = null;
                            window._ttsPlayNext();
                        }, { once: true });
                        audio.addEventListener('error', function() {
                            audio._ttsPlaying = false;
                            window._ttsTs = null;
                            window._ttsPlayNext();
                        }, { once: true });
                        audio.load();
                    };
                }

                if (nuevas_urls && nuevas_urls.length > 0) {
                    nuevas_urls.forEach(function(url) { window._audioQueue.push(url); });
                    window._ttsPlayNext();
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output("audio-tts", "src"),
            Input("cola-audio-nuevos", "data"),
            prevent_initial_call=True,
        )


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

        _estilo_zonas_base = {
            "position": "fixed", "top": 0, "width": "780px", "height": "100vh",
            "background": "#0d0d1a", "borderLeft": "1px solid #2a2a3e",
            "zIndex": 1000, "padding": "24px 20px",
            "boxShadow": "-4px 0 20px rgba(0,0,0,0.5)",
            "transition": "right 0.3s ease", "overflowY": "auto",
        }

        @app.callback(
            Output("cajita-zonas", "style"),
            Output("editor-img", "src"),
            Output("editor-init-cmd", "data"),
            Output("drawer-backdrop-zonas", "style"),
            Output("zona-editor-formas", "data"),
            Output("msg-zonas", "children"),
            Output("zonas-custom-activas", "data"),
            Output("zona-editor-info", "data"),
            Input("btn-zonas", "n_clicks"),
            Input("btn-cerrar-zonas", "n_clicks"),
            Input("drawer-backdrop-zonas", "n_clicks"),
            Input("btn-confirmar-zonas", "n_clicks"),
            Input("btn-reset-zonas", "n_clicks"),
            Input("btn-limpiar-zonas", "n_clicks"),
            State("cajita-zonas", "style"),
            State("zona-editor-formas", "data"),
            State("zona-editor-info", "data"),
            prevent_initial_call=True,
        )
        def gestionar_zonas(n_btn, _cerrar, _backdrop, n_conf, n_reset, n_limp,
                            estilo_actual, formas_actuales, info_editor):
            from dash import ctx, no_update

            backdrop_oculto  = {"display": "none", "position": "fixed", "top": 0, "left": 0,
                                "width": "100vw", "height": "100vh", "zIndex": 999}
            backdrop_visible = {**backdrop_oculto, "display": "block"}
            estilo_cerrado   = {**_estilo_zonas_base, "right": "-800px"}
            estilo_abierto   = {**_estilo_zonas_base, "right": "0px"}
            nu = no_update

            tid = ctx.triggered_id

            # ── Abrir drawer ─────────────────────────────────────────
            if tid == "btn-zonas":
                abierto = estilo_actual is not None and estilo_actual.get("right") == "0px"
                if abierto:
                    return estilo_cerrado, nu, nu, backdrop_oculto, nu, nu, nu, nu
                with self._lock_editor:
                    frame_ed = self._frame_editor
                if frame_ed is not None:
                    h_e, w_e = frame_ed.shape[:2]
                    _, buf = cv2.imencode(".jpg", frame_ed, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpeg_b64 = base64.b64encode(buf).decode()
                else:
                    w_e, h_e = 960, 540
                    jpeg_b64 = ""
                data_url = f"data:image/jpeg;base64,{jpeg_b64}" if jpeg_b64 else ""
                info     = {"w_e": w_e, "h_e": h_e}
                formas_init = self._formas_editor if self._zonas_custom is not None else []
                cmd = {"shapes": formas_init, "t": time.time()}
                return (estilo_abierto, data_url, cmd,
                        backdrop_visible, formas_init, "", nu, info)

            # ── Cerrar drawer ────────────────────────────────────────
            if tid in ("btn-cerrar-zonas", "drawer-backdrop-zonas"):
                return estilo_cerrado, nu, nu, backdrop_oculto, nu, "", nu, nu

            # ── Limpiar formas ───────────────────────────────────────
            if tid == "btn-limpiar-zonas":
                cmd = {"shapes": [], "t": time.time()}
                return nu, nu, cmd, nu, [], "", nu, nu

            # ── "Por defecto" — restaurar YAML y cerrar ──────────────
            if tid == "btn-reset-zonas":
                if self._gestor_eventos:
                    self._gestor_eventos.actualizar_zonas_conflicto(None)
                self._zonas_custom  = None
                self._formas_editor = []
                cmd = {"shapes": [], "t": time.time()}
                return estilo_cerrado, nu, cmd, backdrop_oculto, [], "", False, nu

            # ── Confirmar zonas ──────────────────────────────────────
            if tid == "btn-confirmar-zonas":
                formas = [s for s in (formas_actuales or []) if "x0" in s][:2]
                if not formas:
                    return nu, nu, nu, nu, nu, "Sin rectángulos dibujados", nu, nu

                info = info_editor or {}
                w_e  = info.get("w_e", 960)
                h_e  = info.get("h_e", 540)
                with self._lock_editor:
                    orig_w, orig_h = self._dim_frame_original

                escala_x = orig_w / w_e
                escala_y = orig_h / h_e
                clases_detectables = list(_MAP50_POR_CLASE.keys())
                nuevas_zonas: dict = {}
                for i, s in enumerate(formas):
                    x0o = int(min(s["x0"], s["x1"]) * escala_x)
                    x1o = int(max(s["x0"], s["x1"]) * escala_x)
                    y0o = int(min(s["y0"], s["y1"]) * escala_y)
                    y1o = int(max(s["y0"], s["y1"]) * escala_y)
                    nombre = f"zona_custom_{i + 1}"
                    puntos = np.array(
                        [[x0o, y0o], [x1o, y0o], [x1o, y1o], [x0o, y1o]], dtype=np.int32
                    )
                    nuevas_zonas[nombre] = Zona(
                        nombre=nombre,
                        clases_detectables=clases_detectables,
                        poligono=sv.PolygonZone(polygon=puntos),
                    )
                if self._gestor_eventos:
                    self._gestor_eventos.actualizar_zonas_conflicto(nuevas_zonas)
                self._zonas_custom  = nuevas_zonas
                self._formas_editor = formas
                return (estilo_cerrado, nu, nu, backdrop_oculto,
                        nu, f"✓ {len(nuevas_zonas)} zonas activas", True, nu)

            return nu, nu, nu, nu, nu, nu, nu, nu

        # ── Inicializar canvas cuando cambia el comando ───────────────
        app.clientside_callback(
            """
            function(cmd) {
                if (!cmd) return window.dash_clientside.no_update;
                var img    = document.getElementById('editor-img');
                var canvas = document.getElementById('editor-canvas');
                if (!img || !canvas) return window.dash_clientside.no_update;
                function doInit() {
                    canvas.width  = img.naturalWidth  || 960;
                    canvas.height = img.naturalHeight || 540;
                    if (window.zeInit) window.zeInit('editor-canvas', cmd.shapes || []);
                }
                if (img.complete && img.naturalWidth > 0) { doInit(); }
                else { img.onload = doInit; }
                return window.dash_clientside.no_update;
            }
            """,
            Output("editor-canvas", "title"),
            Input("editor-init-cmd", "data"),
            prevent_initial_call=True,
        )

        app.clientside_callback(
            """
            function(n_clicks) {
                if (!n_clicks) return window.dash_clientside.no_update;
                var img = document.getElementById('detalle-frame-img');
                if (!img || !img.src) return window.dash_clientside.no_update;
                var nombre = img.src.split('/').pop();
                return {nombre: nombre, t: Date.now()};
            }
            """,
            Output("galeria-img-ampliada", "data", allow_duplicate=True),
            Input("detalle-frame-img", "n_clicks"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output("btn-zonas", "className"),
            Output("btn-zonas", "children"),
            Input("zonas-custom-activas", "data"),
            prevent_initial_call=False,
        )
        def actualizar_btn_zonas(activas):
            if activas:
                return "btn-control activo", "✏️  Zona Custom"
            return "btn-control", "✏️  Zona Custom"

        @app.callback(
            Output("galeria-img-ampliada", "data"),
            Input({"type": "galeria-card", "index": ALL}, "n_clicks"),
            prevent_initial_call=True,
        )
        def seleccionar_img_galeria(n_clicks_list):
            from dash import ctx, no_update
            if not ctx.triggered or ctx.triggered[0].get("value", 0) == 0:
                return no_update
            tid = ctx.triggered_id
            if not isinstance(tid, dict) or tid.get("type") != "galeria-card":
                return no_update
            return {"nombre": tid["index"], "t": time.time()}

        @app.callback(
            Output("galeria-lightbox", "style"),
            Output("lightbox-img", "src"),
            Output("lightbox-info", "children"),
            Input("galeria-img-ampliada", "data"),
            Input("galeria-lightbox-backdrop", "n_clicks"),
            Input("btn-cerrar-lightbox", "n_clicks"),
            prevent_initial_call=True,
        )
        def mostrar_lightbox(datos, _backdrop, _cerrar):
            from dash import ctx, no_update as nu
            estilo_oculto  = {"display": "none",  "position": "fixed", "inset": "0", "zIndex": 2000}
            estilo_visible = {"display": "block", "position": "fixed", "inset": "0", "zIndex": 2000}
            if ctx.triggered_id in ("galeria-lightbox-backdrop", "btn-cerrar-lightbox"):
                return estilo_oculto, nu, nu
            if not datos or not datos.get("nombre"):
                return estilo_oculto, nu, nu
            nombre = datos["nombre"]
            tipo   = None
            ts_str = ""
            hito = next(
                (h for h in self._base_datos.hitos_recientes(limite=500)
                 if h.get("ruta_frame") and Path(h["ruta_frame"]).name == nombre),
                None,
            )
            if hito:
                tipo   = hito["tipo"]
                ts_str = datetime.datetime.fromtimestamp(hito["marca_tiempo"]).strftime("%d/%m/%Y %H:%M:%S")
            else:
                ruta = self._carpeta_capturas / nombre
                if ruta.exists():
                    ts_str = datetime.datetime.fromtimestamp(ruta.stat().st_mtime).strftime("%d/%m/%Y %H:%M:%S")
            color      = _COLORES_HITO.get(tipo, _COLOR_HITO_DEFAULT) if tipo else "#607d8b"
            tipo_label = tipo.replace("_", " ").upper() if tipo else "MANUAL"
            info = html.Div(children=[
                html.Span(tipo_label,
                          style={"color": color, "fontFamily": "Rajdhani, sans-serif",
                                 "fontWeight": "700", "letterSpacing": "0.05em",
                                 "marginRight": "12px", "fontSize": "0.9em"}),
                html.Span(ts_str, style={"fontSize": "0.82em", "color": "#505070"}),
            ])
            return estilo_visible, f"/captura-img/{nombre}", info

        _estilo_galeria_base = {**_estilo_drawer_base, "width": "800px"}

        @app.callback(
            Output("cajita-galeria", "style"),
            Output("drawer-backdrop-galeria", "style"),
            Output("galeria-abierta", "data"),
            Input("btn-galeria", "n_clicks"),
            Input("btn-cerrar-galeria", "n_clicks"),
            Input("drawer-backdrop-galeria", "n_clicks"),
            State("cajita-galeria", "style"),
            prevent_initial_call=True,
        )
        def gestionar_galeria_estado(n_btn, _cerrar, _backdrop, estilo_actual):
            from dash import ctx
            bd_oculto  = {"display": "none", "position": "fixed", "top": 0, "left": 0,
                          "width": "100vw", "height": "100vh", "zIndex": 999}
            bd_visible = {**bd_oculto, "display": "block"}
            if ctx.triggered_id == "btn-galeria":
                abierto = estilo_actual is not None and estilo_actual.get("right") == "0px"
                if abierto:
                    return {**_estilo_galeria_base, "right": "-820px"}, bd_oculto, False
                return {**_estilo_galeria_base, "right": "0px"}, bd_visible, True
            return {**_estilo_galeria_base, "right": "-820px"}, bd_oculto, False

        @app.callback(
            Output("galeria-contenido", "children"),
            Output("galeria-filtros-hito", "style"),
            Output("btn-filtro-galeria-todas", "className"),
            Output("btn-filtro-galeria-manual", "className"),
            Output("btn-filtro-galeria-auto", "className"),
            Output("btn-filtro-galeria-hito-todos", "className"),
            *[Output(f"btn-filtro-galeria-hito-{tipo}", "className") for tipo, _ in _HITOS_SIMULABLES],
            Output("galeria-filtro", "data"),
            Input("galeria-abierta", "data"),
            Input("btn-filtro-galeria-todas", "n_clicks"),
            Input("btn-filtro-galeria-manual", "n_clicks"),
            Input("btn-filtro-galeria-auto", "n_clicks"),
            Input("btn-filtro-galeria-hito-todos", "n_clicks"),
            *[Input(f"btn-filtro-galeria-hito-{tipo}", "n_clicks") for tipo, _ in _HITOS_SIMULABLES],
            State("galeria-filtro", "data"),
            prevent_initial_call=True,
        )
        def renderizar_galeria(abierta, *_rest):
            from dash import ctx, no_update
            nu = no_update
            filtro = _rest[-1] or {"categoria": "todas", "hito": None}

            if ctx.triggered_id == "galeria-abierta" and not abierta:
                return (nu,) * (7 + len(_HITOS_SIMULABLES))

            tid = ctx.triggered_id
            if tid == "btn-filtro-galeria-todas":
                filtro = {"categoria": "todas", "hito": None}
            elif tid == "btn-filtro-galeria-manual":
                filtro = {"categoria": "manual", "hito": None}
            elif tid == "btn-filtro-galeria-auto":
                filtro = {"categoria": "auto", "hito": None}
            elif isinstance(tid, str) and tid.startswith("btn-filtro-galeria-hito-"):
                hito_tipo = tid.replace("btn-filtro-galeria-hito-", "")
                filtro = {"categoria": "auto",
                          "hito": None if hito_tipo == "todos" else hito_tipo}

            cat  = filtro["categoria"]
            hito = filtro["hito"]

            estilo_hito_row = {
                "display": "flex" if cat == "auto" else "none",
                "gap": "6px", "flexWrap": "wrap", "marginBottom": "10px",
            }
            cn_todas  = "btn-control activo" if cat == "todas"  else "btn-control"
            cn_manual = "btn-control activo" if cat == "manual" else "btn-control"
            cn_auto   = "btn-control activo" if cat == "auto"   else "btn-control"
            cn_h_todos = "btn-control activo" if cat == "auto" and hito is None else "btn-control"
            cn_hitos = [
                "btn-control activo" if cat == "auto" and hito == tipo else "btn-control"
                for tipo, _ in _HITOS_SIMULABLES
            ]

            contenido = _construir_galeria_contenido(self, filtro)
            return (contenido, estilo_hito_row,
                    cn_todas, cn_manual, cn_auto, cn_h_todos,
                    *cn_hitos, filtro)

        return app

    # ------------------------------------------------------------------

    def _fps(self) -> float:
        if len(self._tiempos_frame) < 2:
            return 0.0
        transcurrido = self._tiempos_frame[-1] - self._tiempos_frame[0]
        return (len(self._tiempos_frame) - 1) / transcurrido if transcurrido > 0 else 0.0

    def _anotar_frame(self, frame: np.ndarray, simulando: str | None = None) -> np.ndarray:
        frame = frame.copy()

        zonas_display = self._zonas_custom if self._zonas_custom is not None else self._zonas
        for nombre, zona in zonas_display.items():
            if nombre == "fauna":
                continue
            if self._zonas_custom is not None:
                # zona_custom_1 → color norte, zona_custom_2 → color sur
                color = _COLORES_ZONA.get(
                    "esquina_norte" if nombre == "zona_custom_1" else "esquina_sur",
                    _COLOR_ZONA_DEFAULT,
                )
            else:
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

def _construir_galeria_contenido(panel: "Panel", filtro: dict) -> html.Div:
    categoria  = filtro.get("categoria", "todas")
    hito_filtro = filtro.get("hito")
    capturas: list[dict] = []

    if categoria in ("todas", "manual"):
        carpeta = panel._carpeta_capturas
        if carpeta.exists():
            for f in sorted(carpeta.glob("panel_*.jpg"),
                            key=lambda p: p.stat().st_mtime, reverse=True):
                ts = f.stat().st_mtime
                capturas.append({
                    "nombre": f.name,
                    "tipo":   None,
                    "ts":     ts,
                    "ts_str": datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M"),
                })

    if categoria in ("todas", "auto"):
        for h in panel._base_datos.hitos_recientes(limite=500):
            if not h.get("ruta_frame") or not h.get("confirmado"):
                continue
            if hito_filtro and h["tipo"] != hito_filtro:
                continue
            nombre = Path(h["ruta_frame"]).name
            ts     = h["marca_tiempo"]
            capturas.append({
                "nombre": nombre,
                "tipo":   h["tipo"],
                "ts":     ts,
                "ts_str": datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M"),
            })

    capturas.sort(key=lambda x: x["ts"], reverse=True)

    if not capturas:
        return html.P("Sin capturas disponibles.",
                      style={"color": "#404060", "fontSize": "0.85em", "padding": "20px 0"})

    cards = []
    for c in capturas:
        color      = _COLORES_HITO.get(c["tipo"], _COLOR_HITO_DEFAULT) if c["tipo"] else "#607d8b"
        tipo_label = c["tipo"].replace("_", " ").upper() if c["tipo"] else "MANUAL"
        cards.append(html.Div(
            id={"type": "galeria-card", "index": c["nombre"]},
            n_clicks=0,
            style={"background": "#0b0b18", "borderRadius": "6px",
                   "overflow": "hidden", "border": "1px solid #1a1a2e",
                   "cursor": "zoom-in"},
            children=[
                html.Img(
                    src=f"/captura-img/{c['nombre']}",
                    style={"width": "100%", "display": "block",
                           "aspectRatio": "16/9", "objectFit": "cover"},
                ),
                html.Div(
                    style={"padding": "6px 8px"},
                    children=[
                        html.Div(tipo_label,
                                 style={"fontSize": "0.68em", "color": color,
                                        "fontFamily": "Rajdhani, sans-serif",
                                        "fontWeight": "700", "letterSpacing": "0.05em",
                                        "marginBottom": "2px"}),
                        html.Div(c["ts_str"],
                                 style={"fontSize": "0.62em", "color": "#404060",
                                        "fontVariantNumeric": "tabular-nums"}),
                    ],
                ),
            ],
        ))

    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
               "gap": "8px", "paddingBottom": "8px"},
        children=cards,
    )


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
        margin=dict(l=42, r=10, t=10, b=28),
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
    fig_metricas.update_layout(**_estilo_grafica)
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
    fig_losses.update_layout(**_estilo_grafica)

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
    fig_clases.update_layout(**_estilo_grafica, height=290)
    fig_clases.update_layout(margin=dict(l=110, r=40, t=10, b=28))
    fig_clases.update_xaxes(range=[0, 1.08])

    _ayuda_metricas = (
        "mAP50 (verde): fracción de instancias detectadas correctamente con IoU ≥ 0.5. "
        "Es el indicador principal del entrenamiento.\n"
        "Precisión (naranja): detecciones correctas / total de detecciones realizadas.\n"
        "Recall (rojo): objetos detectados / total de objetos reales presentes."
    )
    _ayuda_losses = (
        "Sólido = entrenamiento · Punteado = validación.\n"
        "cls: pérdida de clasificación (¿qué personaje es?).\n"
        "box: pérdida de localización (¿dónde está exactamente?).\n"
        "Si train y val divergen mucho → sobreajuste."
    )
    _ayuda_clases = (
        "mAP50 por clase sobre el set de test (93 imágenes).\n"
        "Verde ≥ 0.90 · Naranja ≥ 0.75 · Rojo < 0.75.\n"
        "Un valor bajo indica confusión con otras clases o pocas imágenes de entrenamiento."
    )
    _ayuda_matriz = (
        "Cada fila = clase real · Cada columna = clase predicha.\n"
        "La diagonal perfecta sería 1.0 en todas las celdas.\n"
        "bkg FP: fondo predicho como personaje (falsa alarma).\n"
        "bkg FN: personaje presente no detectado (pérdida)."
    )

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
                html.Div([
                    _encabezado_seccion_modelo("Métricas por época", _ayuda_metricas),
                    dcc.Graph(figure=fig_metricas, config={"displayModeBar": False},
                              style={"height": "240px"}),
                ]),
                html.Div([
                    _encabezado_seccion_modelo("Pérdidas train vs val", _ayuda_losses, izquierda=True),
                    dcc.Graph(figure=fig_losses, config={"displayModeBar": False},
                              style={"height": "240px"}),
                ]),
            ],
        ),
        _encabezado_seccion_modelo("mAP50 por clase", _ayuda_clases),
        dcc.Graph(figure=fig_clases, config={"displayModeBar": False},
                  style={"height": "300px", "marginBottom": "16px"}),
        _encabezado_seccion_modelo("Matriz de confusión normalizada", _ayuda_matriz),
        html.Img(src="/modelo-img/confusion_matrix_normalized.png",
                 style={"width": "100%", "borderRadius": "4px",
                        "border": "1px solid #2a2a3e", "marginBottom": "14px"}),
        _conclusiones_modelo(),
    ]


def _encabezado_seccion_modelo(titulo: str, ayuda: str, izquierda: bool = False) -> html.Div:
    clase = "info-icono info-icono--izquierda" if izquierda else "info-icono"
    return html.Div(
        style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
        children=[
            html.Span(titulo, style={
                "fontSize": "0.68em", "color": "#555",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
            }),
            html.Span(
                ["i", html.Span(ayuda, className="info-popup")],
                className=clase,
            ),
        ],
    )


def _conclusion_item(icono: str, color: str, texto: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "8px", "marginBottom": "8px",
               "alignItems": "flex-start"},
        children=[
            html.Span(icono, style={"color": color, "fontSize": "0.78em",
                                    "fontWeight": "700", "flexShrink": "0",
                                    "marginTop": "1px"}),
            html.Span(texto, style={"fontSize": "0.72em", "color": "#7070a0",
                                    "lineHeight": "1.5"}),
        ],
    )


def _conclusiones_modelo() -> html.Div:
    return html.Div(
        style={"background": "#10101f", "borderRadius": "6px",
               "border": "1px solid #1a1a32", "padding": "12px 14px"},
        children=[
            html.Div("Conclusiones del modelo", style={
                "fontSize": "0.65em", "color": "#555", "textTransform": "uppercase",
                "letterSpacing": "0.08em", "marginBottom": "10px",
            }),
            _conclusion_item("✓", "#00e676",
                "Gorila, Deadpool y Transformer superan mAP50 = 0.99. Sin ambigüedad "
                "en producción con estas clases."),
            _conclusion_item("✓", "#00e676",
                "Estatua de la Libertad, Sonic y Spider-Man por encima de 0.91 — "
                "detección sólida en condiciones reales."),
            _conclusion_item("⚠", "#ff9800",
                "Batman (0.849) y Minnie Mouse (0.823) con algo más de error, "
                "probablemente por variación de iluminación y pocas muestras "
                "representativas en el dataset."),
            _conclusion_item("✗", "#f44336",
                "Mickey Mouse es la clase más débil (mAP50 0.58, recall 0.37): "
                "alta confusión con Minnie por similitud de orejas. Mejorable "
                "añadiendo más imágenes de Mickey con traje rojo visible."),
            _conclusion_item("→", "#7070c0",
                "Sin sobreajuste visible: train y val convergen en las curvas de "
                "pérdida. El modelo generaliza bien con solo 499 imágenes."),
        ],
    )


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


