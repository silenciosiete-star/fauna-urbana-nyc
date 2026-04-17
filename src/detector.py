"""Inferencia YOLO cada N frames. Lee la cola de captura, escribe resultados."""
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv
from loguru import logger
from ultralytics import YOLO

_MODELO_GENERICO = "yolo11n.pt"


@dataclass
class ResultadoDeteccion:
    frame: np.ndarray
    detecciones: sv.Detections
    marca_tiempo: float = field(default_factory=time.time)


class Detector:

    def __init__(
        self,
        cola_entrada: queue.Queue,
        ruta_modelo: str,
        frames_por_inferencia: int = 4,
    ):
        self.cola_entrada = cola_entrada
        self.cola_salida: queue.Queue = queue.Queue(maxsize=5)
        self.frames_por_inferencia = frames_por_inferencia
        self._modelo = self._cargar_modelo(ruta_modelo)
        self._activo = False
        self._hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_deteccion, daemon=True)
        self._hilo.start()
        logger.info(f"Detector iniciado (1 de cada {self.frames_por_inferencia} frames)")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Detector detenido")

    # ------------------------------------------------------------------

    def _cargar_modelo(self, ruta: str) -> YOLO:
        if not Path(ruta).exists():
            logger.warning(
                f"Modelo '{ruta}' no encontrado. Usando modelo genérico '{_MODELO_GENERICO}'."
            )
            return YOLO(_MODELO_GENERICO)
        logger.info(f"Cargando modelo: {ruta}")
        return YOLO(ruta)

    def _bucle_deteccion(self) -> None:
        contador = 0
        while self._activo:
            try:
                frame = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            contador += 1
            if contador % self.frames_por_inferencia != 0:
                continue

            resultado = self._modelo(frame, verbose=False)[0]
            detecciones = sv.Detections.from_ultralytics(resultado)

            logger.debug(f"Frame {contador}: {len(detecciones)} detecciones")

            # Descartar resultado obsoleto si el consumidor no da abasto
            if self.cola_salida.full():
                try:
                    self.cola_salida.get_nowait()
                except queue.Empty:
                    pass

            self.cola_salida.put(ResultadoDeteccion(frame=frame, detecciones=detecciones))
