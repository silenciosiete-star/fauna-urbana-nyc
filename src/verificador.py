"""Verifica hitos con Gemma: confirma la condición, genera el mensaje y envía email si procede."""
import base64
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time

import cv2
import httpx
import numpy as np
from loguru import logger

from .eventos import HitoPotencial

_COLORES_HITO = {
    "crossover":           "#ff9800",
    "conflicto_identidad": "#e040fb",
    "hora_punta":          "#00bcd4",
    "avistamiento_raro":   "#4caf50",
    "marvel_vs_dc":        "#f44336",
}
_COLOR_HITO_DEFAULT = "#607d8b"

_HERRAMIENTA_VOZ = {
    "type": "function",
    "function": {
        "name": "sintetizar_voz",
        "description": "Sintetiza en voz alta el mensaje de alerta delegando en el modelo de TTS local.",
        "parameters": {
            "type": "object",
            "properties": {
                "texto": {
                    "type": "string",
                    "description": "Texto a vocalizar: máximo 2 frases, tono jocoso, en español",
                },
            },
            "required": ["texto"],
        },
    },
}

_HERRAMIENTA_DESCARTAR = {
    "type": "function",
    "function": {
        "name": "descartar_hito",
        "description": "Descarta el hito porque la condición no se cumple visualmente.",
        "parameters": {
            "type": "object",
            "properties": {
                "razon": {
                    "type": "string",
                    "description": "Explicación breve de por qué la condición no se cumple",
                },
            },
            "required": ["razon"],
        },
    },
}

_HERRAMIENTA_EMAIL = {
    "type": "function",
    "function": {
        "name": "enviar_email",
        "description": "Envía un email de alerta cuando el hito es genuino y merece notificación.",
        "parameters": {
            "type": "object",
            "properties": {
                "razonamiento": {
                    "type": "string",
                    "description": "1-2 frases describiendo lo que ves en la imagen y por qué la condición se cumple",
                },
                "asunto": {
                    "type": "string",
                    "description": "Asunto conciso del email (sin emojis)",
                },
                "cuerpo": {
                    "type": "string",
                    "description": "Cuerpo del email: máximo 2 frases con tono jocoso describiendo el hito",
                },
            },
            "required": ["razonamiento", "asunto", "cuerpo"],
        },
    },
}

_CONTEXTO = (
    "Vigilas una cámara fija de Times Square. YOLO detecta personajes disfrazados: "
    "spiderman, deadpool, batman, mickey_mouse, minnie_mouse, sonic, super_mario, "
    "elmo, estatua_libertad, gorila, transformer. "
    "La imagen que recibes tiene los bounding boxes de YOLO ya dibujados con sus etiquetas."
)

_PROMPT_PLANTILLA = (
    "{contexto}\n\n"
    "YOLO ha disparado el hito «{tipo}»: {descripcion}.\n"
    "Clases detectadas en este frame: {detecciones_str}.\n\n"
    "Mira los bounding boxes en la imagen y verifica que la condición del hito se cumple. "
    "Solo descarta si los boxes claramente no corresponden a lo que dice la etiqueta "
    "(ej: el box etiquetado 'gorila' está sobre un objeto inanimado, no una persona).\n"
    "• Si la condición SE CUMPLE → llama a enviar_email (razonamiento breve de lo que ves "
    "en los boxes, asunto conciso, cuerpo jocoso ≤ 2 frases).\n"
    "• Si NO se cumple → llama a descartar_hito con la razón concreta."
)


def _construir_html_email(tipo: str, descripcion: str, mensaje: str, marca_tiempo: float) -> str:
    import datetime
    color = _COLORES_HITO.get(tipo, _COLOR_HITO_DEFAULT)
    ts = datetime.datetime.fromtimestamp(marca_tiempo).strftime("%d/%m/%Y  %H:%M:%S")
    tipo_label = tipo.replace("_", " ").upper()
    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:24px;background:#f0f2f5;font-family:Arial,Helvetica,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;margin:0 auto">
  <tr><td style="background:#ffffff;border-radius:10px;overflow:hidden;border:1px solid #dde1e7;
                 box-shadow:0 2px 8px rgba(0,0,0,0.08)">

    <!-- Cabecera -->
    <table width="100%" cellpadding="0" cellspacing="0"
           style="background:{color}12;border-bottom:3px solid {color};padding:20px 24px">
      <tr>
        <td>
          <div style="font-size:1.4em;font-weight:700;color:{color};letter-spacing:0.06em">
            FAUNA URBANA NYC
          </div>
          <div style="font-size:0.85em;color:#6b7280;margin-top:3px">
            Times Square · Vigilancia en directo
          </div>
        </td>
        <td style="text-align:right;font-size:0.82em;color:#9ca3af;white-space:nowrap;padding-left:16px">
          {ts}
        </td>
      </tr>
    </table>

    <table width="100%" cellpadding="0" cellspacing="0" style="padding:22px 24px">

      <!-- Badge hito -->
      <tr><td style="padding-bottom:18px">
        <table cellpadding="0" cellspacing="0"
               style="background:{color}0d;border-left:4px solid {color};
                      border-radius:0 6px 6px 0;padding:12px 16px;width:100%">
          <tr>
            <td>
              <div style="font-size:0.72em;color:#6b7280;text-transform:uppercase;
                          letter-spacing:0.1em;margin-bottom:5px">HITO DETECTADO</div>
              <div style="font-size:1.35em;font-weight:700;color:{color}">{tipo_label}</div>
            </td>
          </tr>
        </table>
      </td></tr>

      <!-- Condición -->
      <tr><td style="padding-bottom:18px">
        <div style="font-size:0.72em;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px">CONDICIÓN DETECTADA</div>
        <div style="font-size:1em;color:#374151;line-height:1.6">{descripcion}</div>
      </td></tr>

      <!-- Mensaje Gemma -->
      <tr><td style="padding-bottom:22px">
        <table cellpadding="0" cellspacing="0"
               style="background:#f8f9fa;border-left:3px solid {color};
                      border-radius:0 6px 6px 0;padding:14px 16px;width:100%">
          <tr><td>
            <div style="font-size:0.72em;color:#6b7280;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:8px">REPORTE DE GEMMA</div>
            <div style="font-size:1em;color:#1f2937;font-style:italic;line-height:1.7">
              {mensaje}
            </div>
          </td></tr>
        </table>
      </td></tr>

      <!-- Footer -->
      <tr><td style="border-top:1px solid #e5e7eb;padding-top:14px">
        <div style="font-size:0.78em;color:#9ca3af">
          Notificación automática · Fauna Urbana NYC
        </div>
      </td></tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""


@dataclass
class HitoVerificado:
    tipo: str
    frame: np.ndarray
    descripcion: str
    confirmado: bool
    razonamiento: str
    mensaje: str | None  # None si es FALSO_POSITIVO
    texto_tts: str | None = None  # set cuando Gemma llama sintetizar_voz
    marca_tiempo_deteccion: float = field(default_factory=time.time)
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
        self._tts_activo = config_notificaciones.get("tts", {}).get("activo", False)
        self._activo = False
        self._hilo: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="verificador")
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
        self._executor.shutdown(wait=False)
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
            self._executor.submit(self._verificar_y_encolar, hito)

    def _verificar_y_encolar(self, hito: HitoPotencial) -> None:
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
            contenido, llamadas = self._llamar_ollama(prompt, frame_b64)
        elif self._proveedor == "google":
            contenido, llamadas = self._llamar_google(prompt, frame_b64)
        else:
            contenido, llamadas = self._llamar_huggingface(prompt, frame_b64)

        args_email    = next((l["argumentos"] for l in llamadas if l["nombre"] == "enviar_email"), None)
        args_voz      = next((l["argumentos"] for l in llamadas if l["nombre"] == "sintetizar_voz"), None)
        args_descartar = next((l["argumentos"] for l in llamadas if l["nombre"] == "descartar_hito"), None)

        if args_email:
            if self._email_activo:
                self._enviar_email(args_email["asunto"], args_email["cuerpo"], hito)
            logger.info(f"Hito '{hito.tipo}' → CONFIRMADO")
            logger.debug(f"Mensaje Gemma: {args_email['cuerpo']}")
            return HitoVerificado(
                tipo=hito.tipo,
                frame=hito.frame,
                descripcion=hito.descripcion,
                confirmado=True,
                razonamiento=args_email.get("razonamiento") or contenido or "",
                mensaje=args_email["cuerpo"],
                texto_tts=args_voz["texto"] if args_voz else None,
                marca_tiempo_deteccion=hito.marca_tiempo,
            )

        logger.info(f"Hito '{hito.tipo}' → FALSO POSITIVO")
        razonamiento_fp = (args_descartar or {}).get("razon") or contenido or "FALSO_POSITIVO"
        return HitoVerificado(
            tipo=hito.tipo,
            frame=hito.frame,
            descripcion=hito.descripcion,
            confirmado=False,
            razonamiento=razonamiento_fp,
            mensaje=None,
            marca_tiempo_deteccion=hito.marca_tiempo,
        )

    def _herramientas(self) -> list:
        return [_HERRAMIENTA_EMAIL]

    def _llamar_huggingface(self, prompt: str, frame_b64: str) -> tuple[str, list[dict]]:
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
            "tools": self._herramientas(),
            "tool_choice": "auto",
            "max_tokens": 300,
        }
        cabeceras = {"Authorization": f"Bearer {token}"}

        respuesta = httpx.post(url, json=payload, headers=cabeceras, timeout=self._timeout)
        respuesta.raise_for_status()
        mensaje = respuesta.json()["choices"][0]["message"]

        contenido = mensaje.get("content") or ""
        llamadas = []
        for tc in (mensaje.get("tool_calls") or []):
            llamadas.append({
                "nombre": tc["function"]["name"],
                "argumentos": json.loads(tc["function"]["arguments"]),
            })

        return contenido, llamadas

    def _llamar_ollama(self, prompt: str, frame_b64: str) -> tuple[str, list[dict]]:
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
            "tools": self._herramientas(),
            "stream": False,
        }

        respuesta = httpx.post(f"{url_base}/api/chat", json=payload, timeout=self._timeout)
        respuesta.raise_for_status()
        mensaje = respuesta.json()["message"]

        contenido = mensaje.get("content") or ""
        llamadas = []
        for tc in (mensaje.get("tool_calls") or []):
            # Ollama devuelve argumentos ya como dict, no como string JSON
            args = tc["function"]["arguments"]
            llamadas.append({
                "nombre": tc["function"]["name"],
                "argumentos": args if isinstance(args, dict) else json.loads(args),
            })

        return contenido, llamadas

    def _llamar_google(self, prompt: str, frame_b64: str) -> tuple[str, list[dict]]:
        clave = os.getenv("GEMINI_API_KEY", "")
        modelo = self._config.get("gemini_modelo", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={clave}"

        # Convertir herramientas del formato OpenAI al formato nativo de Gemini
        declaraciones = []
        for h in self._herramientas() + [_HERRAMIENTA_DESCARTAR]:
            fn = h["function"]
            props_native = {
                nombre: {"type": spec["type"].upper(), "description": spec.get("description", "")}
                for nombre, spec in fn["parameters"]["properties"].items()
            }
            declaraciones.append({
                "name": fn["name"],
                "description": fn["description"],
                "parameters": {
                    "type": "OBJECT",
                    "properties": props_native,
                    "required": fn["parameters"].get("required", []),
                },
            })

        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}},
                {"text": prompt},
            ]}],
            "tools": [{"function_declarations": declaraciones}],
            "tool_config": {"function_calling_config": {"mode": "ANY"}},
            "generation_config": {
                "thinking_config": {"thinking_budget": 0},
                "max_output_tokens": 500,
            },
        }

        respuesta = httpx.post(url, json=payload, timeout=self._timeout)
        respuesta.raise_for_status()
        parts = respuesta.json()["candidates"][0]["content"]["parts"]

        contenido = ""
        llamadas = []
        for part in parts:
            if "text" in part:
                contenido += part["text"]
            elif "functionCall" in part:
                llamadas.append({
                    "nombre": part["functionCall"]["name"],
                    "argumentos": part["functionCall"]["args"],
                })

        return contenido, llamadas

    def _enviar_email(self, asunto: str, cuerpo: str, hito: "HitoPotencial") -> None:
        url = os.getenv("GAS_EMAIL_URL", "")
        if not url:
            logger.warning("GAS_EMAIL_URL no configurada — email no enviado")
            return
        asunto_completo = f"Fauna Urbana NYC — {asunto}"
        html_cuerpo = _construir_html_email(hito.tipo, hito.descripcion, cuerpo, hito.marca_tiempo)
        respuesta = httpx.post(
            url,
            json={"asunto": asunto_completo, "cuerpo": cuerpo, "html_cuerpo": html_cuerpo},
            timeout=10,
            follow_redirects=True,
        )
        respuesta.raise_for_status()
        logger.debug(f"Email enviado: {asunto_completo}")


# ------------------------------------------------------------------
# Utilidades de módulo

def _codificar_frame(frame: np.ndarray) -> str:
    alto, ancho = frame.shape[:2]
    if ancho > 960:
        factor = 960 / ancho
        frame = cv2.resize(frame, (960, int(alto * factor)))
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode("utf-8")
