"""Hilo de captura del stream de YouTube. Expone una cola de frames."""
import queue
import threading
import time

import cv2
import yt_dlp
from loguru import logger

_SEGUNDOS_RECONEXION = 5
_MAX_FRAMES_COLA = 10


class CapturadorStream:

    def __init__(self, url: str):
        self.url = url
        self.cola: queue.Queue = queue.Queue(maxsize=_MAX_FRAMES_COLA)
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
            "format": "best[ext=mp4]/bestvideo[ext=mp4]/best",
            "quiet": True,
            "js_runtimes": ["node"],
        }
        with yt_dlp.YoutubeDL(opciones) as ydl:
            info = ydl.extract_info(self.url, download=False)
            return info["url"]

    def _bucle_captura(self) -> None:
        while self._activo:
            try:
                logger.info("Obteniendo URL del stream...")
                url_directa = self._obtener_url_directa()
                cap = cv2.VideoCapture(url_directa)

                if not cap.isOpened():
                    raise RuntimeError("No se pudo abrir el stream")

                logger.info("Stream abierto. Capturando frames...")

                while self._activo:
                    ok, frame = cap.read()
                    if not ok:
                        logger.warning("Error leyendo frame, reconectando...")
                        break

                    # Descartar el frame más antiguo si la cola está llena
                    # para no acumular frames obsoletos
                    if self.cola.full():
                        try:
                            self.cola.get_nowait()
                        except queue.Empty:
                            pass

                    self.cola.put(frame)

                cap.release()

            except Exception as error:
                logger.error(f"Error en captura: {error}")

            if self._activo:
                logger.info(f"Reconectando en {_SEGUNDOS_RECONEXION} segundos...")
                time.sleep(_SEGUNDOS_RECONEXION)
