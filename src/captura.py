"""Hilo de captura del stream de YouTube. Expone una cola de frames."""
import queue
import threading
import time

import cv2
import yt_dlp
from loguru import logger

_INTERVALO_RECONEXION_HLS_S = 300  # Renovar URL del stream cada 5 min (expira)

_SEGUNDOS_RECONEXION = 5
_MAX_FRAMES_COLA = 10
_TIMEOUT_OPEN_MS = 8000   # ms para abrir el stream antes de declarar fallo
_TIMEOUT_READ_MS = 8000   # ms para leer un frame antes de reconectar


class CapturadorStream:

    def __init__(self, url: str):
        self.url = url
        self.cola: queue.Queue = queue.Queue(maxsize=_MAX_FRAMES_COLA)
        self.cola_display: queue.Queue = queue.Queue(maxsize=_MAX_FRAMES_COLA)
        self.pausado: bool = False
        self._activo = False
        self._hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_captura, daemon=True)
        self._hilo.start()
        logger.info("Capturador iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Capturador detenido")

    # ------------------------------------------------------------------

    def _obtener_url_directa(self) -> str:
        opciones = {
            "format": "best[height>=1080]/best[height>=720]/best",
            "quiet": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }
        with yt_dlp.YoutubeDL(opciones) as ydl:
            info = ydl.extract_info(self.url, download=False)
            return info["url"]

    def _bucle_captura(self) -> None:
        while self._activo:
            try:
                logger.info("Obteniendo URL del stream...")
                url_directa = self._obtener_url_directa()
                cap = cv2.VideoCapture(url_directa, cv2.CAP_FFMPEG)
                # Timeouts a nivel de FFmpeg para evitar que cap.read()
                # se quede bloqueado indefinidamente si la red cae.
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _TIMEOUT_OPEN_MS)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _TIMEOUT_READ_MS)

                if not cap.isOpened():
                    raise RuntimeError("No se pudo abrir el stream")

                logger.info("Stream abierto. Capturando frames...")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                intervalo = 1.0 / fps
                t_renovar = time.monotonic() + _INTERVALO_RECONEXION_HLS_S

                while self._activo:
                    if time.monotonic() >= t_renovar:
                        logger.info("Renovando URL del stream HLS...")
                        break

                    t_inicio = time.monotonic()
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("Error leyendo frame, reconectando...")
                        break

                    if not self.pausado:
                        for cola in (self.cola, self.cola_display):
                            if cola.full():
                                try:
                                    cola.get_nowait()
                                except queue.Empty:
                                    pass
                            cola.put(frame)

                    # Dormir solo lo que falta para completar el intervalo real-time
                    pausa = intervalo - (time.monotonic() - t_inicio)
                    if pausa > 0:
                        time.sleep(pausa)

                cap.release()

            except Exception as error:
                logger.error(f"Error en captura: {error}")

            if self._activo:
                logger.info(f"Reconectando en {_SEGUNDOS_RECONEXION} segundos...")
                time.sleep(_SEGUNDOS_RECONEXION)
