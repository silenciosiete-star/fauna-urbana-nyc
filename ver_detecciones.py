"""Script temporal para visualizar el stream con detecciones en vivo."""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "xcb"  # Forzar X11 en sistemas con Wayland

import cv2
import supervision as sv
import yt_dlp
from ultralytics import YOLO

URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
RUTA_MODELO = "modelos/yolo11n.pt"
FRAMES_INFERENCIA = 10  # Inferencia cada N frames
ANCHO_DISPLAY = 960     # Solo para la ventana, no afecta a la inferencia


def obtener_url_directa(url: str) -> str:
    opciones = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "extractor_args": {"youtube": {"js_runtimes": ["node"]}},
    }
    with yt_dlp.YoutubeDL(opciones) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


print("Obteniendo URL del stream...")
url_directa = obtener_url_directa(URL)

modelo = YOLO(RUTA_MODELO)
anotador_cajas = sv.BoxAnnotator()
anotador_etiquetas = sv.LabelAnnotator()

cap = cv2.VideoCapture(url_directa)
if not cap.isOpened():
    print("Error: no se pudo abrir el stream")
    sys.exit(1)

print("Stream abierto. Pulsa Q para salir.")

contador = 0
ultimas_detecciones = sv.Detections.empty()
ultimas_etiquetas: list[str] = []

while True:
    ok, frame = cap.read()
    if not ok:
        print("Stream interrumpido")
        break

    contador += 1
    if contador % FRAMES_INFERENCIA == 0:
        resultado = modelo(frame, verbose=False)[0]
        ultimas_detecciones = sv.Detections.from_ultralytics(resultado)
        ultimas_etiquetas = [
            f"{modelo.names[int(cid)]} {conf:.0%}"
            for cid, conf in zip(ultimas_detecciones.class_id, ultimas_detecciones.confidence)
        ]

    frame_display = frame.copy()
    if len(ultimas_detecciones) > 0:
        frame_display = anotador_cajas.annotate(scene=frame_display, detections=ultimas_detecciones)
        frame_display = anotador_etiquetas.annotate(scene=frame_display, detections=ultimas_detecciones, labels=ultimas_etiquetas)

    alto = int(frame.shape[0] * ANCHO_DISPLAY / frame.shape[1])
    frame_display = cv2.resize(frame_display, (ANCHO_DISPLAY, alto))

    cv2.imshow("Fauna Urbana NYC — detecciones", frame_display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
