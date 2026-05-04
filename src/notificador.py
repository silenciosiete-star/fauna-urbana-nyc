"""Despacha hitos verificados: guarda frame, registra en BD, envía Telegram y TTS."""
import asyncio
import datetime
import os
import queue
import threading
from pathlib import Path

import cv2
from loguru import logger

from .base_datos import BaseDatos
from .verificador import HitoVerificado


class Notificador:

    def __init__(
        self,
        cola_entrada: queue.Queue,
        base_datos: BaseDatos,
        config_notificaciones: dict,
        config_capturas: dict,
    ):
        self.cola_entrada = cola_entrada
        self._bd = base_datos
        self._cfg_notif = config_notificaciones
        self._carpeta_capturas = Path(config_capturas.get("carpeta", "capturas/"))
        self._guardar_frame = config_capturas.get("guardar_en_hito", True)
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._bot = None

    def iniciar(self) -> None:
        self._activo = True
        self._carpeta_capturas.mkdir(parents=True, exist_ok=True)
        if self._cfg_notif.get("telegram", {}).get("activo", False):
            from telegram import Bot
            self._bot = Bot(token=os.getenv("TELEGRAM_TOKEN", ""))
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
            self._bd.registrar_hito(hito, ruta_frame)

            if not hito.confirmado:
                continue

            logger.info(f"Hito confirmado: {hito.tipo} — {hito.mensaje}")

            if self._cfg_notif.get("telegram", {}).get("activo", False):
                self._enviar_telegram(hito)

            if self._cfg_notif.get("tts", {}).get("activo", False):
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

    def _enviar_telegram(self, hito: HitoVerificado) -> None:
        try:
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            asyncio.run(self._bot.send_message(chat_id=chat_id, text=hito.mensaje))
            logger.debug("Mensaje Telegram enviado")
        except Exception as error:
            logger.error(f"Error enviando Telegram: {error}")

    def _reproducir_tts(self, hito: HitoVerificado) -> None:
        try:
            import pyttsx3
            velocidad = self._cfg_notif.get("tts", {}).get("velocidad", 150)
            motor = pyttsx3.init()
            motor.setProperty("rate", velocidad)
            motor.say(hito.mensaje)
            motor.runAndWait()
        except Exception as error:
            logger.error(f"Error en TTS: {error}")
