"""Hilo de tracking. Lee ResultadoDeteccion del detector, asigna IDs persistentes con ByteTrack."""
import queue
import threading
from dataclasses import dataclass, field
import time

import numpy as np
import supervision as sv
from loguru import logger

from .detector import ResultadoDeteccion


@dataclass
class ResultadoTracking:
    frame: np.ndarray
    detecciones: sv.Detections  # con tracker_id asignado
    marca_tiempo: float = field(default_factory=time.time)


class Rastreador:

    def __init__(self, cola_entrada: queue.Queue):
        self.cola_entrada = cola_entrada
        self.cola_salida: queue.Queue = queue.Queue(maxsize=5)
        self.cola_display: queue.Queue = queue.Queue(maxsize=5)
        self._tracker = sv.ByteTrack()
        self._activo = False
        self._hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_tracking, daemon=True)
        self._hilo.start()
        logger.info("Rastreador iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Rastreador detenido")

    # ------------------------------------------------------------------

    def _bucle_tracking(self) -> None:
        while self._activo:
            try:
                resultado: ResultadoDeteccion = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            detecciones_con_id = self._tracker.update_with_detections(resultado.detecciones)

            logger.debug(
                f"Tracking: {len(detecciones_con_id)} objetos activos "
                f"(IDs: {detecciones_con_id.tracker_id.tolist() if detecciones_con_id.tracker_id is not None else []})"
            )

            resultado_tracking = ResultadoTracking(
                frame=resultado.frame,
                detecciones=detecciones_con_id,
            )

            for cola in (self.cola_salida, self.cola_display):
                if cola.full():
                    try:
                        cola.get_nowait()
                    except queue.Empty:
                        pass
                cola.put(resultado_tracking)
