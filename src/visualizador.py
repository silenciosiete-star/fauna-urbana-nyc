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
    "trafico": (255, 140, 0),
    "fauna": (0, 200, 0),
    "esquina_norte": (0, 200, 220),
    "zona_foto": (200, 0, 200),
}
_COLOR_ZONA_DEFAULT = (180, 180, 180)


class Visualizador:

    def __init__(self, cola_entrada: queue.Queue, zonas: dict[str, Zona]):
        self.cola_entrada = cola_entrada
        self._zonas = zonas
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._anotador_cajas = sv.BoxAnnotator()
        self._anotador_etiquetas = sv.LabelAnnotator()

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
        while self._activo:
            try:
                resultado: ResultadoTracking = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            frame = self._anotar(resultado)
            alto = int(frame.shape[0] * _ANCHO_DISPLAY / frame.shape[1])
            frame_display = cv2.resize(frame, (_ANCHO_DISPLAY, alto))

            cv2.imshow("Fauna Urbana NYC", frame_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._activo = False
                break

    def _anotar(self, resultado: ResultadoTracking) -> np.ndarray:
        frame = resultado.frame.copy()

        for nombre, zona in self._zonas.items():
            color = _COLORES_ZONA.get(nombre, _COLOR_ZONA_DEFAULT)
            pts = zona.poligono.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x, y = zona.poligono.polygon[0]
            cv2.putText(frame, nombre, (int(x) + 6, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(resultado.detecciones) == 0:
            return frame

        nombres_clase = resultado.detecciones.data.get("class_name", np.array([]))
        tracker_ids = resultado.detecciones.tracker_id

        etiquetas = []
        for i, conf in enumerate(resultado.detecciones.confidence):
            nombre = nombres_clase[i] if i < len(nombres_clase) else str(resultado.detecciones.class_id[i])
            tid = f" #{int(tracker_ids[i])}" if tracker_ids is not None else ""
            etiquetas.append(f"{nombre} {conf:.0%}{tid}")

        frame = self._anotador_cajas.annotate(scene=frame, detections=resultado.detecciones)
        frame = self._anotador_etiquetas.annotate(
            scene=frame, detections=resultado.detecciones, labels=etiquetas
        )
        return frame
