"""Hilo de visualización. Muestra el stream en directo con bboxes, IDs de tracking y límites de zona."""
import os
import queue
import signal
import threading
import time
from collections import deque

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from .rastreador import ResultadoTracking
from .zonas import Zona

_ANCHO_DISPLAY = 960
_COLORES_ZONA = {
    "fauna": (0, 200, 0),
    "esquina_norte": (0, 200, 220),
    "esquina_sur": (200, 0, 200),
}
_COLOR_ZONA_DEFAULT = (180, 180, 180)
_VENTANA_FPS = 30  # Frames para calcular el FPS medio


class Visualizador:

    def __init__(
        self,
        cola_frames: queue.Queue,
        cola_tracking: queue.Queue,
        zonas: dict[str, Zona],
    ):
        self._cola_frames = cola_frames
        self._cola_tracking = cola_tracking
        self._zonas = zonas
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._anotador_cajas = sv.BoxAnnotator()
        self._anotador_etiquetas = sv.LabelAnnotator()
        self._ultimo_tracking: ResultadoTracking | None = None
        self._tiempos_frame: deque = deque(maxlen=_VENTANA_FPS)

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_display, daemon=True)
        self._hilo.start()
        logger.info("Visualizador iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=5)
        cv2.destroyAllWindows()
        logger.info("Visualizador detenido")

    # ------------------------------------------------------------------

    def _bucle_display(self) -> None:
        cv2.namedWindow("Fauna Urbana NYC", cv2.WINDOW_NORMAL)
        dimensiones_logueadas = False
        while self._activo:
            while True:
                try:
                    self._ultimo_tracking = self._cola_tracking.get_nowait()
                except queue.Empty:
                    break

            try:
                frame: np.ndarray = self._cola_frames.get(timeout=0.05)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._activo = False
                continue

            if not dimensiones_logueadas:
                h, w = frame.shape[:2]
                logger.info(f"Tamaño del frame: {w}×{h} px — ajusta config.yaml si las zonas no coinciden")
                dimensiones_logueadas = True

            self._tiempos_frame.append(time.monotonic())

            frame = self._anotar_frame(frame)
            alto = int(frame.shape[0] * _ANCHO_DISPLAY / frame.shape[1])
            cv2.imshow("Fauna Urbana NYC", cv2.resize(frame, (_ANCHO_DISPLAY, alto)))

            tecla = cv2.waitKey(1) & 0xFF
            ventana_cerrada = cv2.getWindowProperty("Fauna Urbana NYC", cv2.WND_PROP_VISIBLE) < 1
            if tecla == ord("q") or ventana_cerrada:
                self._activo = False
                os.kill(os.getpid(), signal.SIGINT)
                break

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

        lineas = [f"FPS: {self._fps():.1f}"]
        lineas.append(f"Objetos: {sum(conteo.values())}")
        for clase, n in sorted(conteo.items()):
            lineas.append(f"  {clase}: {n}")
        if confianza_media > 0:
            lineas.append(f"Conf. media: {confianza_media:.0%}")

        alto_linea = 22
        margen = 10
        padding = 8
        ancho_panel = 180
        alto_panel = len(lineas) * alto_linea + padding * 2

        # Esquina superior izquierda (zona sin acción del stream)
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
