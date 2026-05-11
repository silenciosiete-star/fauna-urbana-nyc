"""Despacha hitos verificados: guarda frame, registra en BD, envía Telegram y TTS."""
import datetime
import queue
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from loguru import logger

from .base_datos import BaseDatos
from .verificador import HitoVerificado

_AUDIO_MAX_EDAD_S = 60 * 60          # eliminar audios TTS de más de 1 h
_INTERVALO_LIMPIEZA_S = 10 * 60      # ejecutar limpieza cada 10 min
_TELEGRAM_TIMEOUT_S = 10             # tiempo máximo de espera por Telegram

# Sustituye palabras inglesas por su pronunciación aproximada en español
# para que espeak-ng (backend de Kokoro) las vocalice correctamente.
_FONET: list[tuple[str, str]] = [
    ("Times Square",  "Taims Skuér"),
    ("Spider-Man",    "Espáiderman"),
    ("Spiderman",     "Espáiderman"),
    ("Deadpool",      "Dédpul"),
    ("Batman",        "Bátman"),
    ("Minnie Mouse",  "Mini Máus"),
    ("Mickey Mouse",  "Miki Máus"),
    ("Super Mario",   "Súper Mario"),
    ("Transformer",   "Transfórmer"),
    ("Sonic",         "Sónic"),
    ("Elmo",          "Élmo"),
]


def _normalizar_tts(texto: str) -> str:
    for original, fonet in _FONET:
        texto = re.sub(re.escape(original), fonet, texto, flags=re.IGNORECASE)
    return texto

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
        self._pipeline_tts = None
        self._lock_pipeline_tts = threading.Lock()
        self._cola_audio: list[str] = []
        self._lock_cola_audio = threading.Lock()
        self._preparando: list[dict] = []
        self._lock_preparando = threading.Lock()

    def iniciar(self) -> None:
        self._activo = True
        self._carpeta_capturas.mkdir(parents=True, exist_ok=True)
        self._hilo = threading.Thread(target=self._bucle_notificaciones, daemon=True)
        self._hilo.start()
        if self._cfg_notif.get("tts", {}).get("activo", False):
            threading.Thread(target=self._precargar_tts, daemon=True).start()
        threading.Thread(target=self._bucle_limpieza, daemon=True).start()
        logger.info("Notificador iniciado")

    def _precargar_tts(self) -> None:
        try:
            from kokoro import KPipeline
            with self._lock_pipeline_tts:
                if self._pipeline_tts is not None:
                    return
                logger.info("Pre-cargando pipeline Kokoro TTS...")
                self._pipeline_tts = KPipeline(lang_code="e")
                logger.info("Kokoro TTS listo")
        except Exception as error:
            logger.error(f"Error pre-cargando TTS: {error}")

    def _bucle_limpieza(self) -> None:
        while self._activo:
            self._limpiar_audios_viejos()
            for _ in range(_INTERVALO_LIMPIEZA_S):
                if not self._activo:
                    return
                time.sleep(1)

    def _limpiar_audios_viejos(self) -> None:
        if not self._carpeta_capturas.exists():
            return
        umbral = time.time() - _AUDIO_MAX_EDAD_S
        eliminados = 0
        for ruta in self._carpeta_capturas.glob("audio_*.wav"):
            try:
                if ruta.stat().st_mtime < umbral:
                    ruta.unlink()
                    eliminados += 1
            except OSError:
                pass
        if eliminados:
            logger.debug(f"Limpieza TTS: {eliminados} audios eliminados")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Notificador detenido")

    def vaciar_cola_audio(self) -> list[str]:
        with self._lock_cola_audio:
            pendientes = list(self._cola_audio)
            self._cola_audio.clear()
            return pendientes

    def hitos_preparando(self) -> list[dict]:
        with self._lock_preparando:
            return list(self._preparando)

    # ------------------------------------------------------------------

    def _bucle_notificaciones(self) -> None:
        while self._activo:
            try:
                hito: HitoVerificado = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            entrada = {"tipo": hito.tipo, "descripcion": hito.descripcion}
            with self._lock_preparando:
                self._preparando.append(entrada)
            try:
                self._procesar_hito(hito)
            finally:
                with self._lock_preparando:
                    self._preparando = [h for h in self._preparando if h is not entrada]

    def _procesar_hito(self, hito: HitoVerificado) -> None:
        ruta_frame = self._guardar_captura(hito) if self._guardar_frame else None

        tts_activo = self._cfg_notif.get("tts", {}).get("activo", False)

        acciones: list[str] = []
        errores: list[str] = []
        if ruta_frame:
            acciones.append("captura")
        if hito.confirmado:
            if self._cfg_notif.get("email", {}).get("activo", False):
                acciones.append("email")
            if self._bot_telegram:
                acciones.append("telegram")
            if tts_activo and hito.mensaje:
                acciones.append("voz")

        if not hito.confirmado:
            self._bd.registrar_hito(hito, ruta_frame, acciones)
            return

        logger.info(f"Hito confirmado: {hito.tipo} — {hito.mensaje}")

        # Telegram corre asíncrono en su propio loop; lanzamos sin bloquear
        # para que TTS pueda generarse en paralelo en este hilo.
        futuro_tg = None
        if self._bot_telegram:
            futuro_tg = self._bot_telegram.enviar_hito(hito)
            if futuro_tg is None:
                logger.warning("Telegram no enviado — bot inactivo (¿falta TELEGRAM_TOKEN?)")
                errores.append("telegram")

        if tts_activo and hito.mensaje:
            # TTS antes de escribir en BD: hito y audio llegan al panel a la vez
            if not self._generar_tts(hito.mensaje, hito.marca_tiempo):
                errores.append("voz")

        if futuro_tg is not None:
            try:
                futuro_tg.result(timeout=_TELEGRAM_TIMEOUT_S)
            except Exception as error:
                logger.error(f"Error enviando Telegram: {error}")
                errores.append("telegram")

        self._bd.registrar_hito(hito, ruta_frame, acciones, errores)

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

    def _generar_tts(self, texto: str, marca_tiempo: float) -> bool:
        try:
            import numpy as np
            import soundfile as sf
            from kokoro import KPipeline

            if self._pipeline_tts is None:
                with self._lock_pipeline_tts:
                    if self._pipeline_tts is None:
                        logger.info("Cargando pipeline Kokoro TTS (primera vez)...")
                        self._pipeline_tts = KPipeline(lang_code="e")
                        logger.info("Kokoro TTS listo")

            chunks = []
            for _, _, audio in self._pipeline_tts(_normalizar_tts(texto), voice="ef_dora"):
                chunks.append(audio)

            if not chunks:
                logger.warning("TTS generó 0 chunks de audio")
                return False

            audio_completo = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            nombre = f"audio_{int(marca_tiempo * 1000)}.wav"
            ruta = self._carpeta_capturas / nombre
            sf.write(str(ruta), audio_completo, 24000)
            with self._lock_cola_audio:
                self._cola_audio.append(nombre)
            logger.debug(f"Audio TTS encolado: {ruta}")
            return True
        except Exception as error:
            logger.error(f"Error generando TTS: {error}")
            return False
