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
| Detección y clasificación | YOLO26 (fine-tuned) | Se analiza 1 de cada N frames para aligerar carga |
| Tracking entre frames | Supervision (ByteTrack) | Interpola posiciones en los frames no analizados |
| Verificador y narrador | Gemma 4 — Google AI Studio (desarrollo) / Ollama local (producción) | Confirma el hito y redacta la notificación en tono jocoso. Asíncrono. Tool calling para envío de email. |
| Panel web | Dash + Plotly | Stream en directo, histórico de hitos, simulador integrado |
| Base de datos | SQLite | |
| Notificaciones | python-telegram-bot | Push de hitos con foto + comandos `/donde`, `/cuantos`, `/captura`, `/estado` |
| Síntesis de voz | Kokoro TTS 82M (hexgrad/Kokoro-82M) | Voz `ef_dora` (español). Cola automática en el panel con toggle 🔊/🔇 |
| Visualización en directo | OpenCV | Ventana con bboxes, IDs de tracking y límites de zona |
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

## Fine-tuning de YOLO26

Sin el fine-tuning, YOLO solo detecta "persona". El reentrenamiento es lo que permite distinguir gorila de Spider-Man de Deadpool.

Clases: `gorila` · `transformer` · `deadpool` · `estatua_libertad` · `sonic` · `spiderman` · `super_mario` · `batman` · `minnie_mouse` · `elmo` · `mickey_mouse`

### Paso 1 — Recopilar frames (`entrenamiento/recopilar_frames.py`)

**Cuándo capturar:** los personajes están en la plaza de **11h a 20h hora de Nueva York** (UTC-4 en verano, UTC-5 en invierno). Fuera de esa franja la plaza está vacía — no tiene sentido capturar.

**Sesiones recomendadas:** 3-4 sesiones en distintos días y horas para cubrir variedad de iluminación:

```bash
# Sesión tipo: ~240 frames en ~2 horas reales de stream
python entrenamiento/recopilar_frames.py --intervalo 30 --maximo 240 --salida datos/frames

# Si quieres una carpeta por sesión para organizarte mejor:
python entrenamiento/recopilar_frames.py --intervalo 30 --maximo 240 --salida datos/frames_manana
python entrenamiento/recopilar_frames.py --intervalo 30 --maximo 240 --salida datos/frames_tarde
```

**Después de cada sesión:** revisar la carpeta y borrar los frames sin personajes visibles antes de subir a Roboflow. El objetivo es llegar a **400-600 frames útiles** en total (con al menos un personaje visible).

> **Desequilibrio de clases:** spider-man aparece con mucha más frecuencia que el gorila o deadpool. Si al revisar los frames ves que alguna clase tiene menos de ~80 imágenes, haz una sesión extra buscando activamente esos momentos o complementa con imágenes externas (convenciones, eventos).

### Paso 2 — Etiquetar con Roboflow

- Crear un proyecto en [Roboflow](https://roboflow.com) con las 11 clases de personajes.
- Subir los frames filtrados y etiquetar bounding boxes. Usar el **auto-label** de Roboflow para acelerar — revisa y corrige, no te fíes al 100%.
- Ritmo orientativo: ~150-200 imágenes/hora. Con 500 frames útiles, ~3-4 horas de etiquetado.

### Paso 3 — Preparar el dataset (`entrenamiento/preparar_dataset.py`)

- Exportar desde Roboflow en formato YOLO
- Split: 70% train / 10% validación / 20% test
- Aumentado de datos: rotaciones, cambios de brillo y contraste, recortes

### Paso 4 — Entrenar (`entrenamiento/entrenar.py`)

Fine-tuning de YOLO26 partiendo de los pesos preentrenados de Ultralytics (transfer learning). No se entrena desde cero.

### Resultados del entrenamiento

Entrenado en RTX 4080 Super · 100 épocas · imgsz=1280 · batch=8.

| Clase | mAP50 | Nota |
|-------|-------|------|
| gorila | 0.995 | |
| transformer | 0.990 | |
| deadpool | 0.995 | |
| estatua_libertad | 0.957 | |
| sonic | 0.911 | |
| spiderman | 0.910 | |
| super_mario | 0.900 | |
| batman | 0.849 | |
| minnie_mouse | 0.823 | |
| elmo | 0.765 | |
| mickey_mouse | 0.580 | Recall bajo (0.37) — confusión con minnie. Mejorable añadiendo más imágenes. |
| **global** | **0.879** | |

El modelo resultante está en `modelos/fauna_urbana.pt` y es el que usa el sistema por defecto.

---

## Instalación

### Dependencias del sistema

```bash
# Node.js — necesario para que yt-dlp resuelva el n-challenge de YouTube
sudo apt-get install -y nodejs

# espeak-ng — motor de pronunciación requerido por Kokoro TTS
sudo apt-get install -y espeak-ng
```

### Entorno Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
HUGGINGFACE_TOKEN=hf_...       # Token de HuggingFace (verificador en desarrollo)
GEMMA_PROVEEDOR=huggingface    # o "ollama" para producción local
```

### Cookies de YouTube (si YouTube bloquea la IP)

Si yt-dlp devuelve `Sign in to confirm you're not a bot`:

1. Instala la extensión **"Get cookies.txt LOCALLY"** en Chrome/Edge
2. Ve a `youtube.com` con sesión iniciada
3. Exporta las cookies y guárdalas como `cookies.txt` en la raíz del proyecto

El capturador las detecta automáticamente.

---

## MVP — Lo que tiene que funcionar

El mínimo presentable y funcional:

- [x] Captura del stream de YouTube en tiempo real
- [x] Detección y clasificación de personajes con YOLO fine-tuned (mAP50 global 0.879)
- [x] Conteo por personaje con visualización sobre el frame
- [x] Al menos 2 zonas configurables por YAML
- [x] 5 hitos implementados con guardado de frame + notificación por email (Google Apps Script)
- [x] Notificación Telegram al disparar hito (push con foto + comandos interactivos)
- [x] Registro de cada detección en SQLite (timestamp, personaje, zona)
- [x] Tracking de trayectorias con Supervision
- [x] **Pruebas de integración con stream real** — superadas
- [x] Visualización en directo del stream con bboxes, IDs de tracking y límites de zona

---

## Extras — Mejoras una vez el MVP funciona

- [x] Panel web con stream en directo, histórico y gráficas temporales (Dash + Plotly)
- [x] Simulador de hitos — inyecta frames del dataset para probar el pipeline sin necesidad del stream real
- [x] Panel de información de reentrenamiento del modelo — métricas, curvas y resultados por clase
- [x] Bot de Telegram — push automático de hitos con foto + comandos interactivos (`/donde`, `/cuantos`, `/captura`, `/estado`)
- [x] Síntesis de voz con Kokoro TTS — cola automática en el panel, botón manual en el drawer, toggle 🔊/🔇
- [x] Mapa de calor — acumulación con decay temporal, colormap JET, toggle en el panel
- [x] Trayectorias por personaje — trail con fade y color único por ID, toggle en el panel
- [x] Verificación paralela de hitos — múltiples hitos verificados simultáneamente con ThreadPoolExecutor
- [x] Panel de modelo mejorado — tooltips ℹ en cada gráfica con explicación de elementos, bloque de conclusiones con valoración por clase en la matriz de confusión
- [x] Selector de zonas custom — editor canvas HTML5 interactivo sobre el frame en directo: dibujado, resize por arista/esquina, anti-solapamiento entre zonas, colores coordinados con el stream
- [x] Galería de capturas — drawer con filtros por categoría (Manuales / Automáticas) y por tipo de hito; lightbox de pantalla completa al hacer clic en cada imagen
- [ ] Despliegue web (Docker)

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
├── modelos/
│   ├── fauna_urbana.pt      # Pesos del modelo fine-tuned
│   └── fauna_urbana/        # Artefactos del entrenamiento (curvas, matriz, CSV)
├── src/
│   ├── captura.py           # Lectura del stream (hilo independiente)
│   ├── detector.py          # Inferencia YOLO cada N frames
│   ├── rastreador.py        # Tracking con Supervision/ByteTrack
│   ├── zonas.py             # Gestión de zonas configurables
│   ├── eventos.py           # Detección de hitos, llama a verificador.py
│   ├── verificador.py       # Gemma 4 asíncrono: confirma hito y genera mensaje jocoso
│   ├── base_datos.py        # Registro en SQLite (incluye razonamiento de Gemma)
│   ├── notificador.py       # Telegram, TTS — despacha el mensaje generado por Gemma
│   ├── simulador.py         # Simula hitos inyectando frames del dataset
│   └── panel.py             # Dashboard web (Dash): stream, hitos, simulador, métricas
├── entrenamiento/
│   ├── recopilar_frames.py  # Extrae frames del stream para el dataset
│   ├── preparar_dataset.py  # Conversión y splits train/val/test
│   └── entrenar.py          # Fine-tuning de YOLO26
├── tools/
│   └── analizar_simulaciones.py  # Análisis de resultados de simulaciones
├── assets/
│   ├── estilo.css           # Estilos del panel web
│   ├── audio.js             # Desbloqueo de autoplay en primer click del usuario
│   └── simulaciones/        # Frames y textos de ejemplo para el simulador
├── capturas/                # Frames guardados al dispararse un hito
└── principal.py             # Punto de entrada
```
