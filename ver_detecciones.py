"""Script temporal para visualizar el stream con detecciones en vivo."""
import sys
import time

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, "src")
from captura import CapturadorStream
from detector import Detector, ResultadoDeteccion

URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
ANCHO_DISPLAY = 960    # Solo afecta a la ventana, YOLO recibe el frame completo
FRAMES_INFERENCIA = 10  # 1 de cada 10 frames en máquina modesta

anotador_cajas = sv.BoxAnnotator()
anotador_etiquetas = sv.LabelAnnotator()

capturador = CapturadorStream(URL)
detector = Detector(
    capturador.cola,
    ruta_modelo="modelos/fauna_urbana.pt",
    frames_por_inferencia=FRAMES_INFERENCIA,
)

capturador.iniciar()
detector.iniciar()

print("Iniciando... (pulsa Q para salir)")

ultimo: ResultadoDeteccion | None = None

while True:
    if not detector.cola_salida.empty():
        ultimo = detector.cola_salida.get()

    if ultimo is None:
        time.sleep(0.05)
        continue

    frame = ultimo.frame.copy()
    dets = ultimo.detecciones

    # Escalar bounding boxes a la resolución de display
    alto_orig, ancho_orig = frame.shape[:2]
    alto_display = int(alto_orig * ANCHO_DISPLAY / ancho_orig)
    escala_x = ANCHO_DISPLAY / ancho_orig
    escala_y = alto_display / alto_orig

    frame = cv2.resize(frame, (ANCHO_DISPLAY, alto_display))

    if len(dets) > 0:
        dets_escaladas = sv.Detections(
            xyxy=dets.xyxy * [escala_x, escala_y, escala_x, escala_y],
            confidence=dets.confidence,
            class_id=dets.class_id,
        )
        etiquetas = [
            f"{detector._modelo.names[int(cid)]} {conf:.0%}"
            for cid, conf in zip(dets.class_id, dets.confidence)
        ]
        frame = anotador_cajas.annotate(scene=frame, detections=dets_escaladas)
        frame = anotador_etiquetas.annotate(scene=frame, detections=dets_escaladas, labels=etiquetas)

    cv2.imshow("Fauna Urbana NYC — detecciones", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

capturador.detener()
detector.detener()
cv2.destroyAllWindows()
