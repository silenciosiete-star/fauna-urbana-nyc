"""Verifica hitos con Gemma: confirma la condición y genera el mensaje jocoso."""
import base64
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

_FALSO_POSITIVO = "FALSO_POSITIVO"

_PROMPT_PLANTILLA = """Eres el vigilante sarcástico de Times Square.
El sistema ha detectado un posible hito: {descripcion}.

1. ¿Se cumple realmente la condición en la imagen? Razona brevemente.
2. Si se cumple, escribe un mensaje de alerta divertido describiendo lo que está pasando (máximo 2 frases, tono jocoso).
   Si NO se cumple, responde únicamente: FALSO_POSITIVO"""


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

    def __init__(self, cola_entrada: queue.Queue, config_gemma: dict):
        self.cola_entrada = cola_entrada
        self.cola_salida: queue.Queue = queue.Queue(maxsize=10)
        self._config = config_gemma
        self._proveedor = os.getenv("GEMMA_PROVEEDOR", config_gemma.get("proveedor", "huggingface"))
        self._timeout = config_gemma.get("timeout_segundos", 15)
        self._activo = False
        self._hilo: threading.Thread | None = None

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
                continue

            if self.cola_salida.full():
                try:
                    self.cola_salida.get_nowait()
                except queue.Empty:
                    pass
            self.cola_salida.put(verificado)

    def _verificar(self, hito: HitoPotencial) -> HitoVerificado:
        frame_b64 = _codificar_frame(hito.frame)
        prompt = _PROMPT_PLANTILLA.format(descripcion=hito.descripcion)

        if self._proveedor == "ollama":
            respuesta_raw = self._llamar_ollama(prompt, frame_b64)
        else:
            respuesta_raw = self._llamar_huggingface(prompt, frame_b64)

        return _parsear_respuesta(hito, respuesta_raw)

    def _llamar_huggingface(self, prompt: str, frame_b64: str) -> str:
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        modelo = self._config["modelo_nombre"]
        url = f"https://api-inference.huggingface.co/models/{modelo}/v1/chat/completions"

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
            "max_tokens": 300,
        }
        cabeceras = {"Authorization": f"Bearer {token}"}

        respuesta = httpx.post(url, json=payload, headers=cabeceras, timeout=self._timeout)
        respuesta.raise_for_status()
        return respuesta.json()["choices"][0]["message"]["content"]

    def _llamar_ollama(self, prompt: str, frame_b64: str) -> str:
        url_base = os.getenv("OLLAMA_URL", "http://192.168.0.135:11434")
        modelo = self._config["modelo_nombre"].split("/")[-1]  # "gemma-3-27b-it" sin prefijo HF

        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [frame_b64],
                }
            ],
            "stream": False,
        }

        respuesta = httpx.post(
            f"{url_base}/api/chat", json=payload, timeout=self._timeout
        )
        respuesta.raise_for_status()
        return respuesta.json()["message"]["content"]


# ------------------------------------------------------------------
# Utilidades de módulo

def _codificar_frame(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def _parsear_respuesta(hito: HitoPotencial, respuesta: str) -> HitoVerificado:
    confirmado = _FALSO_POSITIVO not in respuesta.upper()

    if confirmado:
        lineas = [l.strip() for l in respuesta.strip().splitlines() if l.strip()]
        # Separar razonamiento (punto 1) del mensaje jocoso (punto 2)
        # Gemma suele estructurarlo con numeración; tomamos la última línea como mensaje
        razonamiento = " ".join(lineas[:-1]) if len(lineas) > 1 else lineas[0]
        mensaje = lineas[-1] if len(lineas) > 1 else lineas[0]
    else:
        razonamiento = _FALSO_POSITIVO
        mensaje = None

    estado = "CONFIRMADO" if confirmado else "FALSO POSITIVO"
    logger.info(f"Hito '{hito.tipo}' → {estado}")
    if confirmado:
        logger.debug(f"Mensaje Gemma: {mensaje}")

    return HitoVerificado(
        tipo=hito.tipo,
        frame=hito.frame,
        descripcion=hito.descripcion,
        confirmado=confirmado,
        razonamiento=razonamiento,
        mensaje=mensaje,
    )
