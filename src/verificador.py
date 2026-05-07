"""Verifica hitos con Gemma: confirma la condición, genera el mensaje y envía email si procede."""
import base64
import json
import os
import queue
import threading
from dataclasses import dataclass, field
import time

import cv2
import httpx
import numpy as np
from loguru import logger

from .eventos import HitoPotencial

_HERRAMIENTA_EMAIL = {
    "type": "function",
    "function": {
        "name": "enviar_email",
        "description": "Envía un email de alerta cuando el hito es genuino y merece notificación.",
        "parameters": {
            "type": "object",
            "properties": {
                "asunto": {
                    "type": "string",
                    "description": "Asunto conciso del email (sin emojis)",
                },
                "cuerpo": {
                    "type": "string",
                    "description": "Cuerpo del email: máximo 2 frases con tono jocoso describiendo el hito",
                },
            },
            "required": ["asunto", "cuerpo"],
        },
    },
}

_CONTEXTO = (
    "Vigilas una cámara fija de Times Square. El sistema YOLO rastrea personajes disfrazados "
    "de calle: spiderman, deadpool, batman, mickey_mouse, minnie_mouse, sonic, super_mario, "
    "elmo, estatua_libertad (persona con disfraz), gorila (traje de gorila), transformer. "
    "YOLO ya los ha localizado con bounding boxes — tú solo confirmas o descartas visualmente."
)

_PROMPT_PLANTILLA = (
    "{contexto}\n\n"
    "YOLO ha disparado el hito «{tipo}»: {descripcion}.\n"
    "Objetos que YOLO tiene localizados en este frame: {detecciones_str}.\n"
    "La imagen muestra el frame anotado con sus bounding boxes.\n\n"
    "¿Lo confirmas visualmente? Responde directo, sin razonar en voz alta:\n"
    "• Sí → llama a enviar_email (asunto conciso, cuerpo jocoso ≤ 2 frases).\n"
    "• No → responde exactamente: FALSO_POSITIVO"
)


@dataclass
class HitoVerificado:
    tipo: str
    frame: np.ndarray
    descripcion: str
    confirmado: bool
    razonamiento: str
    mensaje: str | None  # None si es FALSO_POSITIVO
    marca_tiempo: float = field(default_factory=time.time)


class Verificador:

    def __init__(
        self,
        cola_entrada: queue.Queue,
        config_gemma: dict,
        config_notificaciones: dict,
    ):
        self.cola_entrada = cola_entrada
        self.cola_salida: queue.Queue = queue.Queue(maxsize=10)
        self._config = config_gemma
        self._proveedor = os.getenv("GEMMA_PROVEEDOR", config_gemma.get("proveedor", "huggingface"))
        self._timeout = config_gemma.get("timeout_segundos", 15)
        self._email_activo = config_notificaciones.get("email", {}).get("activo", False)
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._en_proceso: list[dict] = []
        self._lock_proceso = threading.Lock()

    def hitos_en_proceso(self) -> list[dict]:
        with self._lock_proceso:
            return list(self._en_proceso)

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_verificacion, daemon=True)
        self._hilo.start()
        logger.info(f"Verificador iniciado (proveedor: {self._proveedor})")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=15)
        logger.info("Verificador detenido")

    # ------------------------------------------------------------------

    def _bucle_verificacion(self) -> None:
        while self._activo:
            try:
                hito: HitoPotencial = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            try:
                verificado = self._verificar(hito)
            except Exception as error:
                logger.error(f"Error verificando hito '{hito.tipo}': {error}")
                verificado = HitoVerificado(
                    tipo=hito.tipo,
                    frame=hito.frame,
                    descripcion=hito.descripcion,
                    confirmado=False,
                    razonamiento=f"Error de verificación: {error}",
                    mensaje=None,
                )

            if self.cola_salida.full():
                try:
                    self.cola_salida.get_nowait()
                except queue.Empty:
                    pass
            self.cola_salida.put(verificado)

    def _verificar(self, hito: HitoPotencial) -> HitoVerificado:
        entrada = {"tipo": hito.tipo, "descripcion": hito.descripcion}
        with self._lock_proceso:
            self._en_proceso.append(entrada)
        try:
            return self._verificar_interno(hito)
        finally:
            with self._lock_proceso:
                self._en_proceso = [h for h in self._en_proceso if h is not entrada]

    def _verificar_interno(self, hito: HitoPotencial) -> HitoVerificado:
        frame_b64 = _codificar_frame(hito.frame)
        prompt = _PROMPT_PLANTILLA.format(
            contexto=_CONTEXTO,
            tipo=hito.tipo.replace("_", " "),
            descripcion=hito.descripcion,
            detecciones_str=hito.detecciones_str or "desconocidos",
        )

        if self._proveedor == "ollama":
            contenido, llamada = self._llamar_ollama(prompt, frame_b64)
        elif self._proveedor == "google":
            contenido, llamada = self._llamar_google(prompt, frame_b64)
        else:
            contenido, llamada = self._llamar_huggingface(prompt, frame_b64)

        if llamada and llamada["nombre"] == "enviar_email":
            args = llamada["argumentos"]
            if self._email_activo:
                self._enviar_email(args["asunto"], args["cuerpo"])
            logger.info(f"Hito '{hito.tipo}' → CONFIRMADO")
            logger.debug(f"Mensaje Gemma: {args['cuerpo']}")
            return HitoVerificado(
                tipo=hito.tipo,
                frame=hito.frame,
                descripcion=hito.descripcion,
                confirmado=True,
                razonamiento=contenido or "Hito confirmado por Gemma.",
                mensaje=args["cuerpo"],
            )

        logger.info(f"Hito '{hito.tipo}' → FALSO POSITIVO")
        return HitoVerificado(
            tipo=hito.tipo,
            frame=hito.frame,
            descripcion=hito.descripcion,
            confirmado=False,
            razonamiento=contenido or "FALSO_POSITIVO",
            mensaje=None,
        )

    def _llamar_huggingface(self, prompt: str, frame_b64: str) -> tuple[str, dict | None]:
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        modelo = self._config["modelo_nombre"]
        url = "https://api-inference.huggingface.co/v1/chat/completions"

        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "tools": [_HERRAMIENTA_EMAIL],
            "tool_choice": "auto",
            "max_tokens": 300,
        }
        cabeceras = {"Authorization": f"Bearer {token}"}

        respuesta = httpx.post(url, json=payload, headers=cabeceras, timeout=self._timeout)
        respuesta.raise_for_status()
        mensaje = respuesta.json()["choices"][0]["message"]

        contenido = mensaje.get("content") or ""
        llamada = None
        tool_calls = mensaje.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            llamada = {
                "nombre": tc["function"]["name"],
                "argumentos": json.loads(tc["function"]["arguments"]),
            }

        return contenido, llamada

    def _llamar_ollama(self, prompt: str, frame_b64: str) -> tuple[str, dict | None]:
        url_base = os.getenv("OLLAMA_URL", "http://192.168.0.135:11434")
        modelo = self._config.get("ollama_modelo", self._config["modelo_nombre"].split("/")[-1])

        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [frame_b64],
                }
            ],
            "tools": [_HERRAMIENTA_EMAIL],
            "stream": False,
        }

        respuesta = httpx.post(f"{url_base}/api/chat", json=payload, timeout=self._timeout)
        respuesta.raise_for_status()
        mensaje = respuesta.json()["message"]

        contenido = mensaje.get("content") or ""
        llamada = None
        tool_calls = mensaje.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            # Ollama devuelve argumentos ya como dict, no como string JSON
            args = tc["function"]["arguments"]
            llamada = {
                "nombre": tc["function"]["name"],
                "argumentos": args if isinstance(args, dict) else json.loads(args),
            }

        return contenido, llamada

    def _llamar_google(self, prompt: str, frame_b64: str) -> tuple[str, dict | None]:
        clave = os.getenv("GEMINI_API_KEY", "")
        modelo = self._config.get("gemini_modelo", "models/gemma-4-26b-a4b-it")
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "tools": [_HERRAMIENTA_EMAIL],
            "tool_choice": "auto",
            "max_tokens": 300,
        }
        cabeceras = {"Authorization": f"Bearer {clave}"}

        respuesta = httpx.post(url, json=payload, headers=cabeceras, timeout=self._timeout)
        respuesta.raise_for_status()
        mensaje = respuesta.json()["choices"][0]["message"]

        contenido = mensaje.get("content") or ""
        llamada = None
        tool_calls = mensaje.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            llamada = {
                "nombre": tc["function"]["name"],
                "argumentos": json.loads(tc["function"]["arguments"]),
            }

        return contenido, llamada

    def _enviar_email(self, asunto: str, cuerpo: str) -> None:
        url = os.getenv("GAS_EMAIL_URL", "")
        if not url:
            logger.warning("GAS_EMAIL_URL no configurada — email no enviado")
            return
        respuesta = httpx.post(url, json={"asunto": asunto, "cuerpo": cuerpo},
                               timeout=10, follow_redirects=True)
        respuesta.raise_for_status()
        logger.debug(f"Email enviado: {asunto}")


# ------------------------------------------------------------------
# Utilidades de módulo

def _codificar_frame(frame: np.ndarray) -> str:
    alto, ancho = frame.shape[:2]
    if ancho > 960:
        factor = 960 / ancho
        frame = cv2.resize(frame, (960, int(alto * factor)))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode("utf-8")
