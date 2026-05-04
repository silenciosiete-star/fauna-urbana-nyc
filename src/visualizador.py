"""Hilo de visualización. Muestra el stream en directo con bboxes, IDs de tracking y límites de zona."""
import queue
import threading

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
        dimensiones_logueadas = False
        while self._activo:
            # Actualizar detecciones si el rastreador tiene resultados nuevos
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
                break

            if not dimensiones_logueadas:
                h, w = frame.shape[:2]
                logger.info(f"Tamaño del frame: {w}×{h} px — ajusta config.yaml si las zonas no coinciden")
                dimensiones_logueadas = True

            frame = self._anotar_frame(frame)
            alto = int(frame.shape[0] * _ANCHO_DISPLAY / frame.shape[1])
            cv2.imshow("Fauna Urbana NYC", cv2.resize(frame, (_ANCHO_DISPLAY, alto)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._activo = False
                break

    def _anotar_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()

        for nombre, zona in self._zonas.items():
            color = _COLORES_ZONA.get(nombre, _COLOR_ZONA_DEFAULT)
            pts = zona.poligono.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x, y = zona.poligono.polygon[0]
            cv2.putText(frame, nombre, (int(x) + 6, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self._ultimo_tracking is None or len(self._ultimo_tracking.detecciones) == 0:
            return frame

        detecciones = self._ultimo_tracking.detecciones
        nombres_clase = detecciones.data.get("class_name", np.array([]))
        tracker_ids = detecciones.tracker_id

        etiquetas = []
        for i, conf in enumerate(detecciones.confidence):
            nombre = nombres_clase[i] if i < len(nombres_clase) else str(detecciones.class_id[i])
            tid = f" #{int(tracker_ids[i])}" if tracker_ids is not None else ""
            etiquetas.append(f"{nombre} {conf:.0%}{tid}")

        frame = self._anotador_cajas.annotate(scene=frame, detections=detecciones)
        frame = self._anotador_etiquetas.annotate(scene=frame, detections=detecciones, labels=etiquetas)
        return frame
