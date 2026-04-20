"""Extrae frames del stream de YouTube para construir el dataset de fine-tuning.

Uso:
    python entrenamiento/recopilar_frames.py
    python entrenamiento/recopilar_frames.py --intervalo 10 --maximo 300 --salida datos/frames

El script captura un frame cada N segundos hasta alcanzar el máximo o recibir Ctrl+C.
Cubre variedad de iluminación ejecutándolo en distintos momentos del día.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yt_dlp
from loguru import logger

URL_STREAM = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"


def obtener_url_directa(url: str) -> str:
    opciones = {
        "format": "best[ext=mp4]/bestvideo[ext=mp4]/best",
        "quiet": True,
        "js_runtimes": {"node": {}},
    }
    with yt_dlp.YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


def recopilar(intervalo_segundos: int, maximo_frames: int, carpeta_salida: Path) -> None:
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando frames en: {carpeta_salida.resolve()}")
    logger.info(f"Intervalo: {intervalo_segundos}s — máximo: {maximo_frames} frames")

    logger.info("Obteniendo URL del stream...")
    url_directa = obtener_url_directa(URL_STREAM)
    cap = cv2.VideoCapture(url_directa)

    if not cap.isOpened():
        logger.error("No se pudo abrir el stream")
        sys.exit(1)

    logger.info("Stream abierto. Capturando... (Ctrl+C para detener)")

    guardados = 0
    ultimo_guardado = 0.0

    try:
        while guardados < maximo_frames:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Error leyendo frame, reconectando...")
                cap.release()
                time.sleep(5)
                url_directa = obtener_url_directa(URL_STREAM)
                cap = cv2.VideoCapture(url_directa)
                continue

            ahora = time.monotonic()
            if ahora - ultimo_guardado >= intervalo_segundos:
                marca = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre = carpeta_salida / f"frame_{marca}_{guardados:04d}.jpg"
                cv2.imwrite(str(nombre), frame)
                guardados += 1
                ultimo_guardado = ahora
                logger.info(f"[{guardados}/{maximo_frames}] {nombre.name}")

    except KeyboardInterrupt:
        logger.info("Interrupción del usuario")
    finally:
        cap.release()

    logger.info(f"Recopilación finalizada: {guardados} frames guardados en {carpeta_salida.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recopila frames del stream para el dataset")
    parser.add_argument("--intervalo", type=int, default=30,
                        help="Segundos entre frames guardados (default: 30)")
    parser.add_argument("--maximo", type=int, default=500,
                        help="Número máximo de frames a guardar (default: 500)")
    parser.add_argument("--salida", type=Path, default=Path("datos/frames"),
                        help="Carpeta de destino (default: datos/frames)")
    args = parser.parse_args()

    recopilar(
        intervalo_segundos=args.intervalo,
        maximo_frames=args.maximo,
        carpeta_salida=args.salida,
    )


if __name__ == "__main__":
    main()
