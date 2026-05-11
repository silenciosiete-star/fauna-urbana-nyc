"""Bot de Telegram unificado: notificaciones push de hitos + comandos interactivos."""
import asyncio
import datetime
import io
import os
import threading
from collections import Counter

import cv2
from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from .base_datos import BaseDatos
from .rastreador import Rastreador
from .verificador import HitoVerificado


class BotTelegram:

    def __init__(self, base_datos: BaseDatos, rastreador: Rastreador):
        self._bd = base_datos
        self._rastreador = rastreador
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._app: Application | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def iniciar(self) -> None:
        if not os.getenv("TELEGRAM_TOKEN", ""):
            logger.warning("TELEGRAM_TOKEN no configurado — bot de Telegram desactivado")
            return
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_bot, daemon=True)
        self._hilo.start()
        logger.info("Bot de Telegram iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=15)
        logger.info("Bot de Telegram detenido")

    def enviar_hito(self, hito: HitoVerificado):
        """Envía el hito por Telegram. Devuelve un Future para poder esperar el resultado."""
        if not self._activo or self._loop is None or not hito.confirmado:
            return None
        return asyncio.run_coroutine_threadsafe(self._enviar_hito_async(hito), self._loop)

    # ------------------------------------------------------------------

    async def _enviar_hito_async(self, hito: HitoVerificado) -> None:
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not chat_id:
            raise RuntimeError("TELEGRAM_CHAT_ID no configurado")
        if not self._app:
            raise RuntimeError("App de Telegram no inicializada")
        tipo_label = hito.tipo.replace("_", " ").upper()
        texto = f"*{tipo_label}*\n{hito.mensaje}"
        _, buf = cv2.imencode(".jpg", hito.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        foto = io.BytesIO(buf.tobytes())
        foto.name = "hito.jpg"
        await self._app.bot.send_photo(
            chat_id=chat_id,
            photo=foto,
            caption=texto,
            parse_mode="Markdown",
        )
        logger.debug(f"Notificación Telegram enviada: {hito.tipo}")

    async def _cmd_donde(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        resultado = self._rastreador.ultimo_resultado
        if resultado is None or len(resultado.detecciones) == 0:
            await update.message.reply_text("No hay personajes visibles ahora mismo.")
            return
        nombres = list(resultado.detecciones.data.get("class_name", []))
        conteo = Counter(n for n in nombres if n)
        if not conteo:
            await update.message.reply_text("No hay personajes visibles ahora mismo.")
            return
        ts = datetime.datetime.fromtimestamp(resultado.marca_tiempo).strftime("%H:%M:%S")
        lineas = [f"• {c.replace('_', ' ')}: ×{n}" for c, n in sorted(conteo.items())]
        await update.message.reply_text(f"Personajes en pantalla ({ts}):\n" + "\n".join(lineas))

    async def _cmd_cuantos(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        hitos = self._bd.hitos_recientes(limite=100)
        confirmados = sum(1 for h in hitos if h["confirmado"])
        descartados = len(hitos) - confirmados
        await update.message.reply_text(
            f"Últimos {len(hitos)} hitos registrados:\n"
            f"• Confirmados: {confirmados}\n"
            f"• Descartados: {descartados}"
        )

    async def _cmd_captura(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        resultado = self._rastreador.ultimo_resultado
        if resultado is None:
            await update.message.reply_text("No hay frame disponible todavía.")
            return
        try:
            _, buf = cv2.imencode(".jpg", resultado.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            foto = io.BytesIO(buf.tobytes())
            foto.name = "captura.jpg"
            ts = datetime.datetime.fromtimestamp(resultado.marca_tiempo).strftime("%d/%m/%Y %H:%M:%S")
            await update.message.reply_photo(photo=foto, caption=f"Times Square · {ts}")
        except Exception as error:
            logger.error(f"Error enviando captura: {error}")
            await update.message.reply_text("Error generando la captura.")

    async def _cmd_estado(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        hitos = self._bd.hitos_recientes(limite=5)
        if not hitos:
            await update.message.reply_text("Sin hitos registrados todavía.")
            return
        lineas = []
        for h in hitos:
            ts = datetime.datetime.fromtimestamp(h["marca_tiempo"]).strftime("%H:%M:%S")
            estado = "✓" if h["confirmado"] else "✗"
            lineas.append(f"{estado} [{ts}] {h['tipo'].replace('_', ' ')}")
        await update.message.reply_text("Últimos hitos:\n" + "\n".join(lineas))

    # ------------------------------------------------------------------

    def _bucle_bot(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._correr_app())

    async def _correr_app(self) -> None:
        token = os.getenv("TELEGRAM_TOKEN", "")
        self._app = Application.builder().token(token).build()
        self._app.add_handler(CommandHandler("donde", self._cmd_donde))
        self._app.add_handler(CommandHandler("cuantos", self._cmd_cuantos))
        self._app.add_handler(CommandHandler("captura", self._cmd_captura))
        self._app.add_handler(CommandHandler("estado", self._cmd_estado))

        async with self._app:
            await self._app.start()
            await self._app.updater.start_polling()
            while self._activo:
                await asyncio.sleep(0.5)
            await self._app.updater.stop()
            await self._app.stop()
