"""Panel web: stream MJPEG en directo + historial de hitos."""
import datetime
import queue
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import supervision as sv
from dash import Dash, Input, Output, State, dcc, html
from flask import Response
from loguru import logger

from .base_datos import BaseDatos
from .rastreador import ResultadoTracking
from .zonas import Zona

_ANCHO_STREAM = 960
_INTERVALO_REFRESCO_MS = 5000
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
        url_stream: str = "",
        puerto: int = 8050,
        carpeta_capturas: str = "capturas",
    ):
        self._cola_frames = cola_frames
        self._cola_tracking = cola_tracking
        self._zonas = zonas
        self._base_datos = base_datos
        self._puerto = puerto
        self._carpeta_capturas = Path(carpeta_capturas)
        video_id = parse_qs(urlparse(url_stream).query).get("v", [""])[0]
        self._url_embed = f"https://www.youtube.com/embed/{video_id}" if video_id else ""
        self._activo = False
        self._pausado = False
        self._hilo_frames: threading.Thread | None = None
        self._ultimo_tracking: ResultadoTracking | None = None
        self._ultimo_frame: np.ndarray | None = None
        self._lock_frame = threading.Lock()
        self._tiempos_frame: deque = deque(maxlen=_VENTANA_FPS)
        self._anotador_cajas = sv.BoxAnnotator()
        self._anotador_etiquetas = sv.LabelAnnotator()
        self._app = self._crear_app()

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
            try:
                frame: np.ndarray = self._cola_frames.get(timeout=0.05)
            except queue.Empty:
                continue
            self._tiempos_frame.append(time.monotonic())
            frame_anotado = self._anotar_frame(frame)
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

        app.layout = html.Div(
            style={"minHeight": "100vh", "padding": "20px 24px"},
            children=[
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
                                    children=[
                                        html.Img(src="/stream"),
                                        html.Div(
                                            className="stream-live-badge",
                                            children=[html.Div(className="live-badge", children=[html.Div(className="live-dot"), "LIVE"])],
                                        ),
                                    ],
                                ),
                                # Controles
                                html.Div(
                                    className="controles",
                                    children=[
                                        html.Button("⏸  Pausar", id="btn-pausa", className="btn-control", n_clicks=0),
                                        html.Button("📸  Captura", id="btn-captura", className="btn-control", n_clicks=0),
                                        html.Span(id="msg-captura", className="msg-control"),
                                    ],
                                ),
                                # Player de audio
                                *([html.Div(
                                    className="audio-wrapper",
                                    children=[
                                        html.Div("Audio del stream", className="seccion-titulo", style={"marginBottom": "8px"}),
                                        html.Iframe(
                                            src=self._url_embed,
                                            className="audio-iframe",
                                            allow="autoplay",
                                        ),
                                        html.P(
                                            "Pulsa ▶ en el player para activar el audio. El vídeo del stream anotado sigue siendo el de arriba.",
                                            className="audio-nota",
                                        ),
                                    ],
                                )] if self._url_embed else []),
                            ],
                        ),
                        # Columna hitos
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Div("Hitos recientes", className="seccion-titulo"),
                                html.Div(id="lista-hitos", style={"overflowY": "auto", "maxHeight": "520px"}),
                            ],
                        ),
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
            if not hitos:
                return html.P("Sin hitos registrados aún.", style={"color": "#404060", "fontSize": "0.85em"}), barra
            elementos = []
            for h in hitos[:30]:
                ts = datetime.datetime.fromtimestamp(h["marca_tiempo"]).strftime("%d/%m %H:%M")
                color_borde = _COLORES_HITO.get(h["tipo"], _COLOR_HITO_DEFAULT)
                color_estado = "#00e676" if h["confirmado"] else "#ff5252"
                estado_txt = "✓" if h["confirmado"] else "✗"
                elementos.append(html.Div(
                    className="hito-card",
                    style={"borderLeftColor": color_borde},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center"},
                            children=[
                                html.Span(ts, className="hito-ts"),
                                html.Span(estado_txt, className="hito-estado", style={"color": color_estado}),
                                html.Span(h["tipo"].replace("_", " "), className="hito-tipo"),
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

        return app

    # ------------------------------------------------------------------

    def _fps(self) -> float:
        if len(self._tiempos_frame) < 2:
            return 0.0
        transcurrido = self._tiempos_frame[-1] - self._tiempos_frame[0]
        return (len(self._tiempos_frame) - 1) / transcurrido if transcurrido > 0 else 0.0

    def _anotar_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()

        for nombre, zona in self._zonas.items():
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

        self._dibujar_panel_stats(frame, detecciones)
        return frame

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
