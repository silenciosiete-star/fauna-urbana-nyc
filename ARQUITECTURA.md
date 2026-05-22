# Arquitectura del código — Fauna Urbana NYC

Este documento explica **cómo está construido cada módulo**, **qué librerías usa** y **para qué sirven**. Es una guía orientada a entender el flujo interno del programa, complementaria al [README.md](README.md) (que se centra en uso y despliegue).

---

## 1. Visión general

El sistema vigila un *stream* en directo de Times Square (YouTube), detecta personajes disfrazados con un modelo YOLO *fine-tuned*, evalúa condiciones de "hito" (ej. *crossover*, *hora punta*), verifica visualmente cada hito con un LLM multimodal (Gemma/Gemini) y notifica por email, Telegram y voz (TTS). Todo se observa desde un panel web.

El programa se organiza como un **pipeline de hilos productor/consumidor** comunicados por `queue.Queue`. Cada módulo es un `threading.Thread` daemon con métodos `iniciar()` / `detener()`.

```
CapturadorStream → Detector → Rastreador → GestorEventos → Verificador → Notificador
                                                                          ├─→ BaseDatos
                                                                          ├─→ BotTelegram
                                                                          └─→ Kokoro (TTS)
                                              ↘ (colas display) ↘
                                                                  Panel (Dash + MJPEG)
```

[principal.py](principal.py) instancia cada módulo, conecta sus colas y los lanza. El servidor WSGI ([waitress](https://docs.pylonsproject.org/projects/waitress/)) sirve el panel Dash en el hilo principal.

### Librerías base

| Librería | Para qué se usa |
|---|---|
| **opencv-python-headless** (`cv2`) | Lectura del stream, redimensionado, codificación JPEG, anotaciones gráficas |
| **yt-dlp** | Resolver la URL HLS real de YouTube (caduca cada ~5 min) |
| **ultralytics** (`YOLO`) | Inferencia y entrenamiento del modelo de detección |
| **supervision** (`sv`) | `Detections`, `ByteTrack` (tracking), `PolygonZone`, anotadores de cajas/etiquetas |
| **httpx** | Cliente HTTP para Ollama, Gemini, HuggingFace y Google Apps Script |
| **dash** + **plotly** | Panel web reactivo (componentes + callbacks) y gráficos del modelo |
| **flask** (incluido en Dash) | Endpoints custom para MJPEG, imágenes y audio |
| **waitress** | Servidor WSGI de producción para Dash |
| **python-telegram-bot** | Bot interactivo (`/donde`, `/captura`, `/estado`) + envío de fotos |
| **kokoro** + **soundfile** + **sounddevice** | TTS local (síntesis a WAV) |
| **loguru** | Logging unificado |
| **pyyaml** | Carga de `config/config.yaml` y `config/entrenamiento.yaml` |
| **python-dotenv** | Carga de credenciales desde `.env` |
| **numpy** | Matrices de imagen y heatmap acumulado |
| **sqlite3** (stdlib) | Persistencia del historial de hitos |

---

## 2. Módulos del pipeline

### 2.1 [src/captura.py](src/captura.py) — `CapturadorStream`

**Qué hace:** Abre el stream HLS de YouTube (o un archivo `.mp4` local en modo demo), captura frames y los publica en dos colas (`cola` para inferencia, `cola_display` para el panel). Reconecta cada 5 minutos (la URL firmada caduca) y ante errores de lectura.

**Librerías clave:**
- `yt_dlp.YoutubeDL` — extrae la URL HLS firmada para la calidad ≥720p (`extractor_args` apunta al cliente Android para evitar restricciones de YouTube).
- `cv2.VideoCapture(url, cv2.CAP_FFMPEG)` — decodificador FFmpeg. Configura `CAP_PROP_OPEN_TIMEOUT_MSEC` / `READ_TIMEOUT_MSEC` para no quedarse colgado.
- `queue.Queue(maxsize=10)` con descarte del más viejo si la cola está llena → el productor nunca bloquea.
- `threading.Thread(daemon=True)` con bucle `_bucle_captura`.

**Detalle interesante:** [captura.py:42](src/captura.py#L42) detecta si la URL es archivo local (modo demo); en ese caso reinicia el vídeo al llegar al final y desactiva el timer de renovación HLS.

---

### 2.2 [src/detector.py](src/detector.py) — `Detector`

**Qué hace:** Lee frames de la cola del capturador y ejecuta inferencia YOLO **1 de cada N frames** (configurable). Empaqueta el resultado en un `dataclass ResultadoDeteccion` y lo encola para el rastreador.

**Librerías clave:**
- `ultralytics.YOLO(ruta)` — carga el modelo `.pt`. Si no existe `modelos/fauna_urbana.pt`, cae a `yolo26n.pt` genérico.
- `supervision.Detections.from_ultralytics(resultado)` — convierte la salida de Ultralytics a `sv.Detections` (formato común con `xyxy`, `confidence`, `class_id`, `data["class_name"]`).
- `dataclasses.dataclass` — define `ResultadoDeteccion(frame, detecciones, marca_tiempo)`.

**Detalle interesante:** Si la cola de salida está llena, descarta el frame más viejo antes de meter el nuevo. Esto asegura que el detector nunca se atasca esperando al consumidor.

---

### 2.3 [src/rastreador.py](src/rastreador.py) — `Rastreador`

**Qué hace:** Asigna **IDs persistentes** a las detecciones (ByteTrack) para que el mismo personaje conserve un `tracker_id` entre frames. Es lo que permite dibujar las trayectorias y diferenciar "dos spidermans distintos" de "el mismo en dos frames".

**Librerías clave:**
- `supervision.ByteTrack()` — implementación del algoritmo ByteTrack (asociación de detecciones por IoU + filtro de Kalman). `tracker.update_with_detections(det)` devuelve `sv.Detections` con `tracker_id` poblado.

**Salida:** dos colas (`cola_salida` → eventos, `cola_display` → panel) con `ResultadoTracking`. Mantiene además `ultimo_resultado` para el bot de Telegram (comando `/donde`).

---

### 2.4 [src/zonas.py](src/zonas.py) — Zonas geométricas

**Qué hace:** Carga polígonos desde `config.yaml` y filtra detecciones que caen dentro.

**Librerías clave:**
- `supervision.PolygonZone(polygon=puntos)` — crea una zona con `trigger(detections)` que devuelve una máscara booleana de qué detecciones caen dentro (usa el "anchor" del centro inferior por defecto).
- `numpy.array(..., dtype=np.int32)` para los puntos del polígono.

**Diseño:** `Zona` es un `dataclass` con `nombre`, `clases_detectables` y `poligono`. `detecciones_en_zona()` devuelve un `sv.Detections` filtrado.

---

### 2.5 [src/eventos.py](src/eventos.py) — `GestorEventos`

**Qué hace:** Evalúa frame a frame **5 tipos de hito** y dispara un `HitoPotencial` cuando la condición se cumple durante N frames consecutivos y el cooldown ha expirado.

Tipos de hito:
- **crossover** — ≥ N universos distintos (Marvel + DC + Disney + …) simultáneos.
- **conflicto_identidad** — dos personajes del mismo tipo en la misma zona.
- **hora_punta** — más de N personajes visibles a la vez.
- **avistamiento_raro** — un personaje aparece tras > 30 min sin verse.
- **marvel_vs_dc** — `spiderman` y `batman` simultáneos.

**Librerías clave:**
- `collections.Counter` — cuenta clases para el `detecciones_str` legible.
- `cv2.polylines` + `cv2.putText` — dibuja las zonas y nombres encima del frame que se envía al verificador.
- `sv.BoxAnnotator`, `sv.LabelAnnotator` — anota cajas y etiquetas (con porcentaje de confianza) sobre la imagen que verá Gemma.

**Diseño clave:**
- `_consecutivos[tipo]` y `_ultimo_disparo[tipo]` mantienen el estado de cada hito.
- `preparar_simulacion(tipo)` y `restaurar_cooldowns()` permiten al [simulador.py](src/simulador.py) forzar el disparo de un solo hito sin que disparen los demás presentes en la imagen.
- `_zonas_conflicto` permite sustituir en caliente las zonas (desde el editor de zonas del panel) sin tocar el YAML.

---

### 2.6 [src/verificador.py](src/verificador.py) — `Verificador`

**Qué hace:** Recibe cada `HitoPotencial`, lo manda a un LLM multimodal **con la imagen ya anotada con las cajas YOLO**, y este decide vía *tool calling* si:
- llamar a `enviar_email` (hito confirmado) → genera asunto + cuerpo jocoso,
- llamar a `descartar_hito` (falso positivo) → razón,
- opcionalmente `sintetizar_voz` con el texto a vocalizar.

**Librerías clave:**
- `httpx.post(...)` — cliente HTTP síncrono hacia tres proveedores:
  - **Ollama** local (`/api/chat`, Gemma 4 con tool calling nativo) — producción.
  - **Google AI Studio** (`generativelanguage.googleapis.com`, Gemini 2.5 Flash con `function_declarations`) — desarrollo.
  - **HuggingFace Inference API** (formato OpenAI) — alternativa.
- `base64` + `cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])` — codifica la imagen para enviarla inline (redimensiona a 960 px para reducir latencia).
- `concurrent.futures.ThreadPoolExecutor(max_workers=3)` — varias verificaciones en paralelo (las llamadas LLM son lentas, ~5–15 s).

**Diseño clave:**
- Las herramientas (`_HERRAMIENTA_EMAIL`, `_HERRAMIENTA_DESCARTAR`, `_HERRAMIENTA_VOZ`) están definidas en formato OpenAI tool-calling. `_llamar_google()` las convierte al formato nativo de Gemini (`function_declarations` con tipos en MAYÚSCULAS).
- El email HTML se genera *server-side* con `_construir_html_email()` y se envía vía un **Google Apps Script** (URL en `.env`), que actúa de relay SMTP gratuito.

---

### 2.7 [src/base_datos.py](src/base_datos.py) — `BaseDatos`

**Qué hace:** Persiste los hitos verificados en SQLite y los expone con caché en memoria.

**Librerías clave:**
- `sqlite3` (stdlib) — con `con.row_factory = sqlite3.Row` para acceder por nombre de columna.
- `threading.Lock()` — todas las escrituras y lecturas del caché serializadas.
- `pathlib.Path(...).mkdir(parents=True, exist_ok=True)` — asegura carpeta.

**Diseño clave:**
- Tabla `hitos` creada con `CREATE TABLE IF NOT EXISTS`; columnas añadidas posteriormente (`marca_tiempo_deteccion`, `acciones`, `errores`) se aplican con `ALTER TABLE` envuelto en try/except → migraciones idempotentes y sin downtime.
- Caché de 500 filas invalidado al insertar — evita que el panel consulte SQLite cada 2 s.

---

### 2.8 [src/bot_telegram.py](src/bot_telegram.py) — `BotTelegram`

**Qué hace:** Bot de Telegram con dos vertientes en un único proceso:
- **Push:** envía foto + caption Markdown cuando se confirma un hito.
- **Interactivo:** responde a `/donde`, `/cuantos`, `/captura`, `/estado`.

**Librerías clave:**
- `telegram.ext.Application`, `CommandHandler` — del paquete `python-telegram-bot` v20+ (API totalmente asíncrona).
- `asyncio.new_event_loop()` corriendo en un `threading.Thread` propio — convive con el resto del pipeline (que es sincrono multihilo) sin contaminarlo.
- `asyncio.run_coroutine_threadsafe(coro, loop)` — el [notificador.py](src/notificador.py) (hilo sincrono) lanza envíos al loop asyncio del bot.
- `io.BytesIO` + `cv2.imencode` — convierte el frame `np.ndarray` a un fichero JPG en memoria para `send_photo`.

---

### 2.9 [src/notificador.py](src/notificador.py) — `Notificador`

**Qué hace:** Orquesta las acciones de salida cuando llega un `HitoVerificado`: guardar frame en disco, escribir en BD, enviar Telegram y generar audio TTS. Para hitos descartados solo guarda el registro.

**Librerías clave:**
- `kokoro.KPipeline(lang_code="e")` — pipeline TTS en español. Se **precarga en un hilo aparte** al arrancar para que el primer hito no sufra el coste de la primera carga (~10 s).
- `soundfile.write(ruta, audio, 24000)` — guarda el WAV a 24 kHz.
- `numpy.concatenate(chunks)` — Kokoro emite chunks de audio que hay que unir.
- `re.sub(re.escape(orig), fonet, ...)` — sustituciones fonéticas (`Spiderman` → `Espáiderman`) para que **espeak-ng** (backend de Kokoro) pronuncie correctamente palabras inglesas en una voz española.

**Diseño clave:**
- Telegram (asyncio) y TTS (CPU intensiva) se lanzan en **paralelo**: primero se dispara el future de Telegram sin esperar, luego se genera el TTS en este mismo hilo, y al final se hace `futuro_tg.result(timeout=10)`. El TTS aprovecha el tiempo de red de Telegram.
- Bucle de limpieza secundario elimina audios `.wav` de más de 1 h cada 10 min.

---

### 2.10 [src/simulador.py](src/simulador.py) — `Simulador`

**Qué hace:** Permite forzar un hito específico desde el panel inyectando una imagen de `assets/simulaciones/<tipo>.jpg` directamente en la cola del capturador. YOLO la procesa como si fuera un frame real, el `GestorEventos` la detecta y el resto del pipeline funciona idéntico.

**Librerías clave:**
- `cv2.imread(ruta)` — carga la imagen de simulación.
- `threading.Thread(daemon=True)` — el simulador corre en su propio hilo para no bloquear el callback del panel.

**Diseño clave:**
- Antes de inyectar llama a `gestor.preparar_simulacion(tipo)` que pone en cooldown todos los demás hitos (cooldown de 9999 s) — solo dispara el que se quiere simular aunque la imagen contenga varios personajes.
- Pausa el capturador real (`pausado=True`) durante la simulación.
- Inyecta frames a 8 fps hasta que `_ultimo_disparo[tipo]` cambie (detecta el disparo) o hasta el timeout de 12 s.

---

### 2.11 [src/panel.py](src/panel.py) — `Panel` (≈ 2000 líneas)

**Qué hace:** Interfaz web con tres áreas:
1. **Stream en vivo** vía MJPEG con overlays (heatmap, trayectorias, zonas, panel de stats).
2. **Historial de hitos** con galería filtrable, reproducción de audio TTS, modal de detalle.
3. **Sección "Modelo"** con gráficos Plotly del entrenamiento (mAP por clase, matriz de confusión, etc.).

**Librerías clave:**
- `dash.Dash` con callbacks (`Input` / `Output` / `State` / `ALL` pattern-matching) — toda la interactividad reactiva.
- `flask.Response` — endpoints custom registrados en `app.server.route(...)`:
  - `/stream` → genera MJPEG `multipart/x-mixed-replace` desde `_generar_mjpeg()` (yield de bytes con boundaries `--frame`).
  - `/modelo-img/<n>`, `/captura-img/<n>`, `/audio/<n>` → `send_from_directory`.
  - `/frame-zona` → frame crudo (sin anotar) para el editor de zonas en el navegador.
- `plotly.graph_objects` — gráficos de la sección del modelo.
- `cv2` — anotar cada frame (cajas, etiquetas, heatmap, trails).
- `numpy` — matriz acumulada del heatmap con decay temporal.
- `collections.deque(maxlen=...)` — trayectorias por `tracker_id` (300 posiciones, ~12 s a 25 fps) y ventana de FPS.

**Diseño clave:**
- Un solo hilo (`_bucle_frames`) consume las dos colas display y produce un único JPEG cacheado (`_ultimo_jpeg`). **Todos los clientes MJPEG comparten ese buffer** → coste constante.
- Heatmap: matriz `float32` de tamaño del frame; cada detección suma un disco en el suelo del personaje (`cv2.circle(acum, (cx, by1), 20, 2.0, -1)`); cada frame se multiplica por 0.99 (decay) y al renderizar se aplica `GaussianBlur` + `applyColorMap(JET)` + blending con `addWeighted`.
- Trails: por cada `tracker_id` se guarda un `deque` de centros; se dibujan con `cv2.line` y alpha-fading creciente (más reciente = más opaco).
- Editor de zonas en el navegador: el `dcc.Store` `zona-editor-formas` guarda los polígonos dibujados; al confirmar, `actualizar_zonas_conflicto()` los aplica al `GestorEventos` sin reiniciar.
- El servidor lo arranca **waitress** desde [principal.py](principal.py#L103) (8 threads) — Dash NO se autoarranca con `app.run_server()` para tener control del shutdown.

---

## 3. Configuración y arranque

### 3.1 [config/config.yaml](config/config.yaml)
Única fuente de verdad: URL del stream, ruta del modelo, umbrales de confianza, parámetros de cada hito (`activo`, `cooldown_segundos`, mínimos), zonas (polígonos en coordenadas del frame 1920×1080), puerto del panel y flags de notificación.

### 3.2 [.env](/) (no versionado)
Credenciales sensibles (`GEMINI_API_KEY`, `OLLAMA_URL`, `GAS_EMAIL_URL`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `HOST_UID`, `HOST_GID`). Cargado con `python-dotenv` en el primer import de [principal.py](principal.py).

### 3.3 [principal.py](principal.py)
- Carga config + `.env`.
- Instancia los módulos en orden, encadenando las colas (`detector.cola_salida` → `rastreador.cola_entrada`, etc.).
- Conecta el panel a `simulador`, `verificador`, `notificador` y `gestor_eventos` (referencias inversas, no por cola).
- Arranca todos los módulos con `modulo.iniciar()`.
- Sirve el WSGI de Dash con `waitress.serve(panel.app_wsgi(), host="0.0.0.0", port=8050, threads=8)`. Waitress instala sus propios handlers de SIGINT/SIGTERM, así que retorna limpiamente con `Ctrl+C` o `docker stop`.
- En el `finally`, llama a `modulo.detener()` en orden inverso (Panel → Detector → Capturador).

---

## 4. Entrenamiento del modelo

Carpeta [entrenamiento/](entrenamiento/), independiente del runtime.

### 4.1 [recopilar_frames.py](entrenamiento/recopilar_frames.py)
Extrae 1 frame cada N segundos del stream con `yt-dlp` + `cv2.VideoCapture`. Output → `datos/frames/`.

### 4.2 [preparar_dataset.py](entrenamiento/preparar_dataset.py)
Coge el export de Roboflow (formato YOLO) y hace un **split estratificado por clase dominante** 70/10/20 (train/valid/test). Las imágenes sin labels se asignan todas a `train` como negativos. Con `--meteo` añade augmentaciones meteorológicas (lluvia/niebla) generadas con `cv2`.

### 4.3 [entrenar.py](entrenamiento/entrenar.py)
Carga `config/entrenamiento.yaml`, llama a `YOLO(modelo_base).train(**parametros)` (Ultralytics se encarga de mosaic, mixup, hsv, lr scheduling, early stopping…), copia `best.pt` a `modelos/fauna_urbana.pt` y ejecuta `model.val(split="test")` para reportar mAP50 por clase.

---

## 5. Empaquetado

### [Dockerfile](Dockerfile)
- **Multi-stage:** `builder` compila wheels (`pip wheel`), `runtime` instala con `--no-index --find-links=/wheels` (rebuilds rápidos).
- Dependencias del sistema en runtime: `ffmpeg` (HLS), `espeak-ng` (Kokoro), `libsndfile1` (soundfile), `libglib2.0-0` (opencv-headless), `tini` (PID 1 que propaga SIGTERM).
- Usuario no-root `fauna` con UID/GID parametrizables vía `--build-arg` para que los bind-mounts (`datos_bd`, `capturas`) sean escribibles desde el host.
- Healthcheck contra `http://127.0.0.1:8050/`.

### [docker-compose.yml](docker-compose.yml)
Bind-mounts del modelo (read-only), `capturas/`, `datos_bd/`, `.env` (read-only) y `config/` (read-only para editar sin rebuild).

---

## 6. Patrones recurrentes

- **Hilos productor/consumidor con `queue.Queue(maxsize=N)`** y descarte del más viejo cuando la cola está llena → el pipeline degrada con elegancia bajo carga.
- **`dataclass` para los mensajes que viajan entre módulos** (`ResultadoDeteccion`, `ResultadoTracking`, `HitoPotencial`, `HitoVerificado`) — tipado claro y `field(default_factory=time.time)` para marcas de tiempo.
- **Locks finos por recurso** (`_lock_heatmap`, `_lock_trails`, `_lock_editor`, `_lock_proceso`, `_lock_cola_audio`) en lugar de un único mutex global.
- **`loguru`** con un nivel por mensaje (`logger.info` para hitos / inicio, `logger.debug` para tracing por frame, `logger.warning` para fallos recuperables).
- **Configuración por YAML + sobreescritura por `.env`** — el código nunca lleva URLs ni credenciales hardcodeadas.
