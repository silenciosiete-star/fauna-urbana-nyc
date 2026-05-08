"""Despacha hitos verificados: guarda frame, registra en BD, envía Telegram y TTS."""
import datetime
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
from loguru import logger

from .base_datos import BaseDatos
from .verificador import HitoVerificado

if TYPE_CHECKING:
    from .bot_telegram import BotTelegram


class Notificador:

    def __init__(
        self,
        cola_entrada: queue.Queue,
        base_datos: BaseDatos,
        config_notificaciones: dict,
        config_capturas: dict,
        bot_telegram: "BotTelegram | None" = None,
    ):
        self.cola_entrada = cola_entrada
        self._bd = base_datos
        self._cfg_notif = config_notificaciones
        self._carpeta_capturas = Path(config_capturas.get("carpeta", "capturas/"))
        self._guardar_frame = config_capturas.get("guardar_en_hito", True)
        self._bot_telegram = bot_telegram
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._pipeline_tts = None  # carga lazy de KPipeline en el primer hito
        self.cb_tts: Callable[[float | None], None] | None = None

    def iniciar(self) -> None:
        self._activo = True
        self._carpeta_capturas.mkdir(parents=True, exist_ok=True)
        self._hilo = threading.Thread(target=self._bucle_notificaciones, daemon=True)
        self._hilo.start()
        logger.info("Notificador iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Notificador detenido")

    # ------------------------------------------------------------------

    def _bucle_notificaciones(self) -> None:
        while self._activo:
            try:
                hito: HitoVerificado = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            ruta_frame = self._guardar_captura(hito) if self._guardar_frame else None

            acciones: list[str] = []
            if ruta_frame:
                acciones.append("captura")
            if hito.confirmado:
                if self._cfg_notif.get("email", {}).get("activo", False):
                    acciones.append("email")
                if self._bot_telegram:
                    acciones.append("telegram")
                if hito.texto_tts:
                    acciones.append("voz")

            self._bd.registrar_hito(hito, ruta_frame, acciones)

            if not hito.confirmado:
                continue

            logger.info(f"Hito confirmado: {hito.tipo} — {hito.mensaje}")

            if self._bot_telegram:
                self._bot_telegram.enviar_hito(hito)

            if hito.texto_tts:
                self._reproducir_tts(hito)

    def _guardar_captura(self, hito: HitoVerificado) -> str | None:
        try:
            marca = datetime.datetime.fromtimestamp(hito.marca_tiempo).strftime("%Y%m%d_%H%M%S")
            ruta = self._carpeta_capturas / f"{hito.tipo}_{marca}.jpg"
            cv2.imwrite(str(ruta), hito.frame)
            logger.debug(f"Frame guardado: {ruta}")
            return str(ruta)
        except Exception as error:
            logger.error(f"Error guardando frame: {error}")
            return None

    def _reproducir_tts(self, hito: HitoVerificado) -> None:
        try:
            from kokoro import KPipeline
            import sounddevice as sd

            if self._pipeline_tts is None:
                logger.info("Cargando pipeline Kokoro TTS (primera vez)...")
                self._pipeline_tts = KPipeline(lang_code="e")
                logger.info("Kokoro TTS listo")

            if self.cb_tts:
                self.cb_tts(hito.marca_tiempo)

            for _, _, audio in self._pipeline_tts(hito.texto_tts, voice="ef_dora"):
                sd.play(audio, samplerate=24000)
                sd.wait()

        except Exception as error:
            logger.error(f"Error en TTS: {error}")
        finally:
            if self.cb_tts:
                self.cb_tts(None)
