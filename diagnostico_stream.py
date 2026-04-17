"""Diagnóstico: comprueba si los frames del stream tienen contenido real."""
import sys
import cv2
import numpy as np
import yt_dlp

URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"

def obtener_url_directa(url):
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

print("Obteniendo URL...")
url_directa = obtener_url_directa(URL)
print(f"URL obtenida: {url_directa[:80]}...")

cap = cv2.VideoCapture(url_directa)
print(f"Stream abierto: {cap.isOpened()}")

for i in range(5):
    ok, frame = cap.read()
    if not ok:
        print(f"Frame {i+1}: ERROR al leer")
        continue
    media = frame.mean()
    print(f"Frame {i+1}: shape={frame.shape}, media_pixeles={media:.1f} {'(NEGRO)' if media < 1 else '(OK)'}")

# Guardar el último frame para inspeccionarlo visualmente
if ok:
    cv2.imwrite("frame_prueba.jpg", frame)
    print("\nFrame guardado en frame_prueba.jpg — ábrelo para ver si tiene contenido.")

cap.release()
