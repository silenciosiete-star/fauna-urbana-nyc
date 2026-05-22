# Fauna Urbana NYC

Agente de visión por ordenador que monitoriza en tiempo real la fauna de personajes disfrazados de Times Square: gorilas, Spider-Man, Deadpool, Mickey y Minnie Mouse, y cualquier criatura que se cruce por ahí.

---

## Idea general

Times Square tiene su propia vida salvaje. Este sistema analiza el stream en vivo de YouTube, detecta y clasifica a los personajes disfrazados, los rastrea por la imagen, y reacciona cuando ocurren situaciones dignas de atención (o de risa).

---

## Tecnologías

| Componente | Tecnología | Notas |
|------------|------------|-------|
| Captura del stream | OpenCV + yt-dlp | Hilo independiente con timeouts FFmpeg para no bloquear la inferencia |
| Detección y clasificación | YOLO26 (fine-tuned) | Se analiza 1 de cada N frames para aligerar carga |
| Tracking entre frames | Supervision (ByteTrack) | Interpola posiciones en los frames no analizados |
| Verificador y narrador | Gemma 4 vía Ollama local | Confirma el hito y redacta la notificación en tono jocoso. Asíncrono. Tool calling para envío de email. Fallback opcional a Gemini 2.5 Flash en desarrollo. |
| Panel web | Dash + Plotly servido con Waitress | Stream en directo (MJPEG compartido), histórico de hitos, simulador integrado |
| Base de datos | SQLite con caché en memoria | |
| Notificaciones | python-telegram-bot | Push de hitos con foto + comandos `/donde`, `/cuantos`, `/captura`, `/estado` |
| Síntesis de voz | Kokoro TTS 82M (hexgrad/Kokoro-82M) | Voz `ef_dora` (español). Cola automática en el panel con toggle 🔊/🔇 |
| Despliegue | Docker + docker-compose | Imagen multi-stage CPU (~1.5 GB). Volúmenes para BD, capturas, modelo y config. |
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

Sin el fine-tuning, YOLO solo detecta "persona". El reentrenamiento permite distinguir las 11 clases del proyecto: `gorila` · `transformer` · `deadpool` · `estatua_libertad` · `sonic` · `spiderman` · `super_mario` · `batman` · `minnie_mouse` · `elmo` · `mickey_mouse`.

**Dataset:** 499 imágenes capturadas del stream con `entrenamiento/recopilar_frames.py`, etiquetadas en Roboflow y preparadas con `entrenamiento/preparar_dataset.py` (split 70/10/20, aumentado con variaciones de brillo y contraste). **Entrenamiento:** `entrenamiento/entrenar.py` — transfer learning sobre pesos preentrenados de Ultralytics, RTX 4080 Super, 100 épocas, imgsz=1280, batch=8.

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
| mickey_mouse | 0.580 | Recall bajo (0.37) — confusión con minnie. |
| **global** | **0.879** | |

El modelo fine-tuned está en `modelos/fauna_urbana.pt` (Git LFS) y es el que usa el sistema por defecto.

---

## Despliegue

### Con Docker (recomendado)

Modo único de producción. La imagen incluye Python, OpenCV, Ultralytics, Kokoro TTS y todas las dependencias de sistema (`ffmpeg`, `espeak-ng`, `libsndfile1`).

**Requisitos previos:**
- Docker Engine + plugin `compose` (versión moderna `docker compose`, no `docker-compose`)
- [Git LFS](https://git-lfs.com) instalado en el host (`sudo apt-get install git-lfs && git lfs install`)
- Servidor Ollama accesible en la red local con Gemma 4 cargado

**1. Clonar el repo y descargar el modelo:**

El modelo fine-tuned (`modelos/fauna_urbana.pt`, 20 MB) está gestionado con Git LFS. Tras clonar:

```bash
git clone https://github.com/silenciosiete-star/fauna-urbana-nyc.git
cd fauna-urbana-nyc
git lfs pull                  # descarga los .pt reales (sin esto solo tendrías los punteros LFS)
```

Verificar que el fichero pesa ~20 MB (no ~130 bytes):

```bash
ls -lh modelos/fauna_urbana.pt
```

**2. Configurar variables de entorno:**

```bash
cp .env.example .env
```

Edita `.env` y rellena estas claves (todas obligatorias salvo `HUGGINGFACE_TOKEN` y `GEMINI_API_KEY`, que son alternativas opcionales):

| Variable | Cómo obtenerla |
|----------|----------------|
| `GEMMA_PROVEEDOR=ollama` | Fija. Indica el proveedor del verificador. |
| `OLLAMA_URL` | URL del servidor Ollama en la red local con Gemma 4 cargado (ej. `http://192.168.0.135:11434`). Verificable con `curl <url>/api/tags`. |
| `TELEGRAM_TOKEN` | Habla con [@BotFather](https://t.me/BotFather) en Telegram → `/newbot` → sigue los pasos → te devuelve el token. |
| `TELEGRAM_CHAT_ID` | Habla con [@userinfobot](https://t.me/userinfobot), te responde con tu ID al instante. Alternativa: escribe `/start` al bot recién creado y luego abre `https://api.telegram.org/bot<TOKEN>/getUpdates` — el `result[0].message.chat.id` es tu ID. |
| `GAS_EMAIL_URL` | URL del despliegue web del Google Apps Script que envía el email. Ver sección «Configurar Google Apps Script». |
| `HOST_UID` / `HOST_GID` | `id -u` / `id -g` en tu host. Necesarios para que el contenedor pueda escribir en los bind-mounts (`datos_bd/`, `capturas/`). |

**3. Construir y arrancar:**

```bash
mkdir -p capturas datos_bd      # directorios para los volúmenes
docker compose build            # ~5-10 min la primera vez
docker compose up -d
docker compose logs -f fauna
```

Cuando los logs muestren `Serving on http://0.0.0.0:8050`, abre el panel en [http://localhost:8050](http://localhost:8050).

**4. Operación habitual:**

```bash
docker compose ps               # estado del contenedor
docker compose logs -f fauna    # seguir logs en vivo
docker compose restart fauna    # reiniciar tras cambios en config/
docker compose down             # parar y eliminar
```

`config/` se monta read-only desde el host: editar `config/config.yaml` y reiniciar para aplicar cambios sin rebuild.

### Configurar Google Apps Script (email)

Cuando Gemma confirma un hito, el verificador hace POST a `GAS_EMAIL_URL` con `{asunto, cuerpo, html_cuerpo}`. El script lo recibe y manda el email vía la cuenta Google del propietario del script.

**Pasos:**

1. Entra en [script.google.com](https://script.google.com) → `Nuevo proyecto`.
2. Pega este código:

   ```javascript
   function doPost(e) {
     const datos = JSON.parse(e.postData.contents);
     MailApp.sendEmail({
       to: Session.getActiveUser().getEmail(),
       subject: datos.asunto,
       htmlBody: datos.html_cuerpo || datos.cuerpo,
     });
     return ContentService.createTextOutput("ok");
   }
   ```

3. `Implementar` → `Nuevo despliegue` → tipo **Aplicación web**.
4. Ejecutar como: **yo**. Quién tiene acceso: **cualquiera**.
5. Acepta los permisos cuando Google los pida (acceso a tu Gmail para enviar).
6. Copia la URL del despliegue → es tu `GAS_EMAIL_URL`.

> Si modificas el script más tarde, hay que crear un **nuevo** despliegue (no editar el existente) para que la URL cambie y los permisos se refresquen.

### En local (modo desarrollo)

Útil para iterar sobre el código sin reconstruir la imagen.

```bash
# Dependencias del sistema (Ubuntu/Debian)
sudo apt-get install -y ffmpeg espeak-ng libsndfile1

# Entorno Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configurar .env igual que en modo Docker
cp .env.example .env
# (editar .env)

# Arrancar
python principal.py
```

### Cookies de YouTube (si YouTube bloquea la IP)

Si yt-dlp devuelve `Sign in to confirm you're not a bot`:

1. Instala la extensión **"Get cookies.txt LOCALLY"** en Chrome/Edge
2. Ve a `youtube.com` con sesión iniciada
3. Exporta las cookies y guárdalas como `cookies.txt` en la raíz del proyecto

El capturador las detecta automáticamente.

### Modo demo (stream offline)

Si el stream de YouTube no está disponible, el sistema puede leer un vídeo local en su lugar. El repositorio incluye `demo_times_square.mp4` (1 minuto, 1080p, grabado del stream `a9J1OP_x5Rg`) gestionado con **Git LFS**.

Para alternar en caliente sin reiniciar, pulsa el botón **📹 Demo** en la barra de controles del panel: el capturador reconecta a `demo_times_square.mp4`, omite yt-dlp y rebobina al llegar al final (bucle indefinido). Pulsa de nuevo para volver al stream original definido en `config/config.yaml`.

Alternativa permanente (sin botón): edita `stream.url` en `config/config.yaml` a `"demo_times_square.mp4"` y reinicia el contenedor.

> En Docker, `demo_times_square.mp4` y la carpeta de artefactos de entrenamiento `modelos/fauna_urbana/` se bind-mountan desde el host (ver [docker-compose.yml](docker-compose.yml)) — no van en la imagen, así que `git lfs pull` en el host es suficiente.

---

## MVP — Lo que tiene que funcionar

El mínimo presentable y funcional:

- [x] Captura del stream de YouTube en tiempo real
- [x] Detección y clasificación de personajes con YOLO fine-tuned (mAP50 global 0.879)
- [x] Conteo por personaje con visualización sobre el frame
- [x] Al menos 2 zonas configurables por YAML
- [x] 5 hitos implementados con guardado de frame
- [x] Notificación por email al disparar hito (Google Apps Script)
- [x] Notificación Telegram al disparar hito (push con foto + comandos interactivos)
- [x] Registro de cada detección en SQLite (timestamp, personaje, zona)
- [x] **Pruebas de integración con stream real** — superadas

---

## Extras — Mejoras una vez el MVP funciona

- [x] Panel web con stream en directo, histórico y gráficas temporales (Dash + Plotly)
- [x] Simulador de hitos — inyecta frames del dataset para probar el pipeline sin necesidad del stream real
- [x] Panel de información del modelo — métricas, curvas y resultados por clase; tooltips ℹ en cada gráfica, bloque de conclusiones con valoración por clase
- [x] Bot de Telegram — push automático de hitos con foto + comandos interactivos (`/donde`, `/cuantos`, `/captura`, `/estado`)
- [x] Síntesis de voz con Kokoro TTS — cola automática en el panel, botón manual en el drawer, toggle 🔊/🔇
- [x] Mapa de calor — acumulación con decay temporal, colormap JET, toggle en el panel
- [x] Trayectorias por personaje — trail con fade y color único por ID, toggle en el panel
- [x] Verificación paralela de hitos — múltiples hitos verificados simultáneamente con ThreadPoolExecutor
- [x] Selector de zonas custom — editor canvas HTML5 interactivo sobre el frame en directo: dibujado, resize por arista/esquina, anti-solapamiento entre zonas, colores coordinados con el stream
- [x] Galería de capturas — drawer con filtros por categoría (Manuales / Automáticas) y por tipo de hito; lightbox de pantalla completa al hacer clic en cada imagen
- [x] Botón 📹 Demo — alterna en caliente entre stream real y `demo_times_square.mp4` sin reiniciar el contenedor
- [x] **Despliegue Docker** — imagen multi-stage CPU, panel servido con Waitress, volúmenes para BD/capturas/modelo, UID/GID configurables por build-arg para que los bind-mounts sean escribibles

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
├── Dockerfile               # Imagen multi-stage CPU con Python 3.11 + apt deps
├── docker-compose.yml       # Orquestación: volúmenes, UID/GID, restart policy
├── .dockerignore            # Excluye .venv, .git, datos, capturas, etc. del build
├── .env / .env.example      # Credenciales y UID/GID del host
├── requirements.txt         # opencv-headless + ultralytics + dash + waitress + kokoro
├── principal.py             # Punto de entrada: arranca hilos y sirve panel con Waitress
├── config/
│   └── config.yaml          # Zonas, hitos, umbrales, URL del stream
├── datos/                   # Frames recopilados y dataset etiquetado (modo dev)
├── datos_bd/                # SQLite del sistema en producción (volumen)
├── capturas/                # Frames y audios TTS generados al dispararse un hito (volumen)
├── modelos/
│   ├── fauna_urbana.pt      # Pesos del modelo fine-tuned
│   └── fauna_urbana/        # Artefactos del entrenamiento (curvas, matriz, CSV)
├── src/
│   ├── captura.py           # Lectura del stream con timeouts FFmpeg (hilo independiente)
│   ├── detector.py          # Inferencia YOLO cada N frames
│   ├── rastreador.py        # Tracking con Supervision/ByteTrack
│   ├── zonas.py             # Gestión de zonas configurables
│   ├── eventos.py           # Detección de hitos, llama a verificador.py
│   ├── verificador.py       # Gemma 4 asíncrono: confirma hito y genera mensaje jocoso
│   ├── base_datos.py        # SQLite con caché en memoria + invalidación al insertar
│   ├── notificador.py       # Telegram + TTS en paralelo, limpieza periódica de audios
│   ├── bot_telegram.py      # Push de hitos + comandos interactivos
│   ├── simulador.py         # Simula hitos inyectando frames del dataset
│   └── panel.py             # Dashboard Dash: stream MJPEG compartido, hitos, simulador, métricas
├── entrenamiento/
│   ├── recopilar_frames.py  # Extrae frames del stream para el dataset
│   ├── preparar_dataset.py  # Conversión y splits train/val/test
│   └── entrenar.py          # Fine-tuning de YOLO26
├── tools/
│   └── analizar_simulaciones.py  # Análisis de resultados de simulaciones
└── assets/
    ├── estilo.css           # Estilos del panel web
    ├── audio.js             # Desbloqueo de autoplay en primer click del usuario
    ├── zona_editor.js       # Editor canvas HTML5 para zonas custom
    └── simulaciones/        # Frames y textos de ejemplo para el simulador
```
