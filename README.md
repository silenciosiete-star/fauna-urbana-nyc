# Fauna Urbana NYC

Agente de visión por ordenador que monitoriza en tiempo real la fauna de personajes disfrazados de Times Square: gorilas, Spider-Man, Deadpool, Mickey y Minnie Mouse, y cualquier criatura que se cruce por ahí.

---

## Idea general

Times Square tiene su propia vida salvaje. Este sistema analiza el stream en vivo de YouTube, detecta y clasifica a los personajes disfrazados, los rastrea por la imagen, y reacciona cuando ocurren situaciones dignas de atención (o de risa).

---

## Tecnologías

| Componente | Tecnología | Notas |
|------------|------------|-------|
| Captura del stream | OpenCV + yt-dlp | Hilo independiente para no bloquear la inferencia |
| Detección y clasificación | YOLO v11 (fine-tuned) | Se analiza 1 de cada N frames para aligerar carga |
| Tracking entre frames | Supervision (ByteTrack) | Interpola posiciones en los frames no analizados |
| Verificador y narrador | Gemma 4 (Ollama / OpenRouter) | Confirma el hito y redacta la notificación en tono jocoso. Asíncrono. |
| Panel web | Dash + Plotly | |
| Base de datos | SQLite | |
| Notificaciones | python-telegram-bot | Alertas y control remoto |
| Síntesis de voz | pyttsx3 / Coqui TTS | |
| Configuración | YAML | Zonas, hitos y umbrales sin tocar código |

> **SAM3 descartado:** la segmentación pixel a pixel no aporta lo suficiente para el uso real del proyecto (conteo, tracking, zonas, heatmap) como para justificar su coste computacional.

---

## Arquitectura de procesamiento

La captura y la inferencia van en **hilos separados** para que un frame lento del modelo no bloquee la lectura del stream. Gemma se llama de forma asíncrona para no bloquear el stream mientras razona:

```
[Hilo captura]  →  cola de frames  →  [Hilo inferencia]  →  [Hilo UI / panel]
   yt-dlp                                YOLO + Supervision
   OpenCV                                cada N frames
                                              |
                                     ¿posible hito?
                                              ↓
                                    [Gemma 4 — asíncrono]
                                    1. ¿Es real el hito?
                                    2. Redacta mensaje jocoso
                                              ↓
                                    [Notificador — Telegram / TTS]
```

**Ejemplo de notificación generada por Gemma:**
> *"Spider-Man y Deadpool llevan varios minutos negociándose la esquina norte. El gorila los observa desde el fondo con escepticismo evidente. He decidido alertarte."*

Si Gemma determina que es un falso positivo, el hito no se dispara y queda registrado como descartado en SQLite junto al razonamiento.

---

## Fine-tuning de YOLO v11

Sin el fine-tuning, YOLO solo detecta "persona". El reentrenamiento es lo que permite distinguir gorila de Spider-Man de Deadpool, y también mantener la detección de vehículos para la zona de tráfico.

### Estrategia: entrenamiento mixto (personajes + vehículos)

El modelo se entrena con **un único dataset** que combina:
- Imágenes de personajes disfrazados de Times Square (anotadas a mano)
- Imágenes de vehículos del dataset COCO (importadas directamente desde Roboflow sin anotar a mano)

Así el modelo resultante detecta personajes Y vehículos, y basta con un solo modelo en el sistema.

Clases objetivo: `gorila` · `spider-man` · `deadpool` · `mickey` · `minnie` · `coche` · `taxi` · `autobus` · `moto`

> **Desequilibrio de clases:** COCO tiene miles de imágenes de vehículos y el dataset de personajes será pequeño. Hay que compensarlo con pesos por clase o sobremuestreo de personajes durante el entrenamiento.

### Paso 1 — Recopilar frames (`entrenamiento/recopilar_frames.py`)

Extraer capturas del stream en distintos horarios (mañana, tarde, noche) para cubrir variedad de iluminación y distancias. Complementar con imágenes de personajes de otras fuentes (eventos, convenciones) si el stream no da suficientes muestras.

### Paso 2 — Etiquetar con Roboflow

- Personajes: anotar bounding boxes a mano. Roboflow tiene auto-etiquetado que acelera el proceso.
- Vehículos: importar directamente desde COCO en Roboflow, ya vienen anotados.
- Mezclar ambos en un único proyecto de Roboflow.

### Paso 3 — Preparar el dataset (`entrenamiento/preparar_dataset.py`)

- Exportar desde Roboflow en formato YOLO
- Split: 70% train / 10% validación / 20% test
- Aumentado de datos: rotaciones, cambios de brillo y contraste, recortes

### Paso 4 — Entrenar (`entrenamiento/entrenar.py`)

Fine-tuning de YOLO v11 partiendo de los pesos preentrenados de Ultralytics (transfer learning). No se entrena desde cero.

### Paso 5 — Evaluar y sustituir

Medir **mAP por clase** sobre el set de test para personajes y vehículos por separado. Si los resultados son aceptables, sustituir el modelo genérico en `config.yaml`.

---

## MVP — Lo que tiene que funcionar

El mínimo presentable y funcional:

- [ ] Captura del stream de YouTube en tiempo real
- [ ] Detección y clasificación de personajes con YOLO fine-tuned
- [ ] Conteo por personaje con visualización sobre el frame
- [ ] Al menos 2 zonas configurables por YAML
- [ ] 3 hitos funcionando con guardado de frame + notificación Telegram
- [ ] Registro de cada detección en SQLite (timestamp, personaje, zona)
- [ ] Tracking de trayectorias con Supervision

---

## Extras — Mejoras una vez el MVP funciona

- [ ] Panel web con histórico y gráficas temporales (Dash + Plotly)
- [ ] Bot de Telegram interactivo con comandos (`/donde gorila`, `/cuantos ahora`, `/captura`)
- [ ] Verificador Gemma 4: confirmación de hito + mensaje jocoso generado
- [ ] Síntesis de voz (TTS) anunciando los eventos por altavoz
- [ ] Mapa de calor de zonas con más actividad
- [ ] Zonas dibujables en tiempo real (en lugar de solo por config)
- [ ] Despliegue en Docker

---

## Hitos definidos

Para evitar falsos positivos, un hito solo se dispara si la condición se mantiene durante **al menos 5 frames consecutivos**.

| Hito | Condición | Zona | Acción |
|------|-----------|------|--------|
| **Avengers Assemble** | 3 o más superhéroes detectados | Cualquier zona | Email + captura + Telegram |
| **Conflicto de identidad** | 2 personajes de la misma clase | Misma zona | Telegram + registro en BD |
| **Hora punta de la fauna** | Total de personajes > umbral configurable | Frame completo | Captura automática |
| **Avistamiento raro** | Personaje ausente más de X minutos reaparece | Cualquier zona | Telegram |
| **Marvel vs DC** | Spider-Man y Batman simultáneos | Cualquier zona | TTS por altavoz |

---

## Estructura del proyecto

```
fauna-urbana-nyc/
├── config/
│   └── config.yaml          # Zonas, hitos, umbrales, URL del stream
├── datos/                   # Frames recopilados y dataset etiquetado
├── modelos/                 # Pesos del modelo fine-tuned
├── src/
│   ├── captura.py           # Lectura del stream (hilo independiente)
│   ├── detector.py          # Inferencia YOLO cada N frames
│   ├── rastreador.py        # Tracking con Supervision/ByteTrack
│   ├── zonas.py             # Gestión de zonas configurables
│   ├── eventos.py           # Detección de hitos, llama a verificador.py
│   ├── verificador.py       # Gemma 4 asíncrono: confirma hito y genera mensaje jocoso
│   ├── base_datos.py        # Registro en SQLite (incluye razonamiento de Gemma)
│   ├── notificador.py       # Telegram, TTS — despacha el mensaje generado por Gemma
│   └── panel.py             # Dashboard web (Dash)
├── entrenamiento/
│   ├── recopilar_frames.py  # Extrae frames del stream para el dataset
│   ├── preparar_dataset.py  # Conversión y splits train/val/test
│   └── entrenar.py          # Fine-tuning de YOLO v11
├── capturas/                # Frames guardados al dispararse un hito
└── principal.py             # Punto de entrada
```
