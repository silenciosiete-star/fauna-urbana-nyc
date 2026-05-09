# Contexto del proyecto para Claude

## Qué es esto

Agente de visión por ordenador que monitoriza el stream de YouTube de Times Square en tiempo real para detectar, clasificar y rastrear personajes disfrazados (gorila, Spider-Man, Deadpool, Mickey, Minnie). Es un proyecto de clase con un lado jocoso: detecta situaciones absurdas y reacciona a ellas.

Repositorio privado: https://github.com/silenciosiete-star/fauna-urbana-nyc

---

## Convenciones — leerlas antes de escribir código

- **Todo en español**: nombres de variables, funciones, clases, comentarios, mensajes de log, docstrings. Sin excepciones.
- Sin co-autoría de Claude en los commits. Solo aparece el usuario.
- No añadir manejo de errores, abstracciones ni funcionalidades no pedidas explícitamente.

---

## Decisiones de arquitectura ya tomadas — no reabrir

| Decisión | Motivo |
|----------|--------|
| YOLO26 fine-tuned como único modelo de detección | Clasifica y detecta en un solo paso, rápido. Dataset mixto: personajes (anotados a mano) + vehículos COCO (ya anotados). Un solo modelo cubre ambas zonas. STAL (Small-Target-Aware Label Assignment) mejora la detección de personajes pequeños/lejanos, que es el caso de la cámara de Times Square. |
| SAM3 descartado | Bounding boxes son suficientes para todos los casos de uso del proyecto |
| Gemma 4 como verificador y narrador de hitos | YOLO detecta la condición; Gemma confirma con criterio semántico y redacta la notificación. Se llama de forma asíncrona para no congelar el stream. |
| Dos proveedores para Gemma según entorno | Desarrollo: `GEMMA_PROVEEDOR=google` → **Gemini 2.5 Flash** vía Google AI Studio (requiere `GEMINI_API_KEY`). Producción: `GEMMA_PROVEEDOR=ollama` → **Gemma 4** local en 192.168.0.135. **Importante**: Gemma 4 en Google AI Studio NO soporta tool calling (devuelve 500). Por eso en desarrollo se usa Gemini 2.5 Flash, que sí lo soporta. Ollama local con Gemma 4 sí soporta tool calling. Se cambia en `.env`, sin tocar código. |
| API nativa de Gemini para tool calling (no endpoint OpenAI-compatible) | `tool_choice: "required"` en el endpoint OpenAI-compatible (`/v1beta/openai/...`) falla silenciosamente con Gemini 2.5 Flash: devuelve `tool_calls: null` porque el thinking interno consume el token budget antes de generar la tool call. Solución definitiva: API nativa (`/v1beta/models/{modelo}:generateContent`) con `thinking_config.thinking_budget: 0` (desactiva thinking → menos tokens, más determinista) y `tool_config.function_calling_config.mode: "ANY"` (equivalente nativo a `required`). Ver `verificador.py::_llamar_google`. |
| Prompting del verificador: anclar en bounding boxes | En modo tool calling, Gemini 2.5 Flash reclasifica el contenido visual desde cero ignorando las detecciones de YOLO, y usa `descartar_hito` como salida fácil ante cualquier duda. El frame que llega al verificador ya tiene los bounding boxes dibujados con etiquetas (lo hace `eventos.py::_anotar_frame`). El prompt debe decirlo explícitamente y dar un criterio de descarte muy concreto ("solo si el box está sobre un objeto inanimado"), nunca dejarlo abierto. Frases como "confirma visualmente" sin más contexto dan al modelo licencia para re-evaluar y siempre descarta. |
| Dos modelos YOLO en inferencia | `modelos/fauna_urbana.pt` para las 11 clases de personajes. Modelo pretrained COCO para vehículos (zona izquierda). No se mezclan en el mismo fine-tuning: el dataset de personajes es demasiado pequeño para coexistir con COCO sin degradar la detección de personajes. |
| Captura e inferencia en hilos separados | Evita que un frame lento de YOLO bloquee la lectura del stream |
| Procesar 1 de cada N frames (configurable) | Supervision/ByteTrack interpola el tracking entre frames no analizados |
| imgsz=1280 en entrenamiento e inferencia | La cámara es lejana y los personajes son objetos pequeños. Se entrenó con imgsz=1280 en una RTX 4080 Super (batch=8, ~10 GB VRAM). Inferencia a 6.9 ms/imagen. |
| Hitos con umbral de 5 frames consecutivos | Evita falsos positivos por detecciones puntuales |

---

## Patrón del verificador Gemma — cómo funciona

Cuando `eventos.py` detecta un posible hito (condición cumplida durante N frames consecutivos), en lugar de disparar directamente la acción se llama a `verificador.py` de forma **asíncrona** para no bloquear el stream.

**Flujo:**
```
eventos.py         →       verificador.py          →     notificador.py
                                                    
"posible hito"     →  frame + prompt a Gemma 4     →  mensaje de Gemma
                      (verificación + narración)       vía Telegram / TTS
```

**El prompt a Gemma tiene dos objetivos en una sola llamada:**
```
Eres el vigilante sarcástico de Times Square.
El sistema ha detectado un posible hito: [descripción del hito].

1. ¿Se cumple realmente la condición en la imagen? Razona brevemente.
2. Si se cumple, escribe un mensaje de alerta divertido describiendo
   lo que está pasando (máximo 2 frases, tono jocoso).
   Si NO se cumple, responde únicamente: FALSO_POSITIVO
```

**Ejemplo de salida de Gemma:**
> *"Spider-Man y Deadpool llevan varios minutos negociándose la esquina norte. El gorila los observa desde el fondo con escepticismo evidente. He decidido alertarte."*

**Qué se registra en SQLite por cada hito:**
- Timestamp
- Tipo de hito
- Zona
- Razonamiento de Gemma (por qué confirmó o descartó)
- Mensaje de notificación generado
- Ruta al frame guardado

Si Gemma devuelve `FALSO_POSITIVO`, el hito no se dispara y se registra como descartado.

---

## Bot de Telegram — cómo funciona

`src/bot_telegram.py` agrupa en un único módulo las dos funciones Telegram del sistema: notificaciones push automáticas y comandos interactivos del usuario.

**Configuración necesaria (en `.env`):**
```
TELEGRAM_TOKEN=<token de @BotFather>
TELEGRAM_CHAT_ID=<ID del chat que recibirá las notificaciones>
```
Para obtener el token: habla con `@BotFather` en Telegram → `/newbot`.
Para obtener el chat ID: escríbele `/start` al bot recién creado y luego abre en el navegador `https://api.telegram.org/bot<TOKEN>/getUpdates` — el campo `result[0].message.chat.id` es tu ID. Alternativamente, `@userinfobot` te lo da al instante.

Si `TELEGRAM_TOKEN` no está configurado, `BotTelegram.iniciar()` emite un warning y el resto del sistema sigue funcionando sin Telegram.

**Hilo de ejecución:**

El bot corre en su propio hilo con un event loop asyncio dedicado. Esto evita conflictos con el resto de hilos del sistema que son síncronos.

**Notificaciones push (automáticas):**

Cuando `Notificador` recibe un hito confirmado, llama a `bot_telegram.enviar_hito(hito)`. El bot envía al chat una foto del frame anotado con el tipo de hito en negrita y el mensaje jocoso generado por Gemma como pie de foto.

**Comandos interactivos:**

| Comando | Respuesta |
|---------|-----------|
| `/donde` | Personajes visibles en el último frame analizado con su conteo y hora |
| `/cuantos` | Estadística de hitos confirmados vs descartados (últimos 100) |
| `/captura` | Foto del frame actual con timestamp |
| `/estado` | Lista de los últimos 5 hitos registrados en BD |

`/donde` y `/captura` leen `rastreador.ultimo_resultado`, un atributo que el `Rastreador` actualiza en cada frame procesado. Si el sistema lleva menos de un ciclo activo, responden que aún no hay frame disponible.

---

## Stack

| Componente | Tecnología |
|------------|------------|
| Captura del stream | OpenCV + yt-dlp |
| Detección y clasificación | YOLO26 (ultralytics) |
| Tracking | Supervision + ByteTrack |
| LLM verificador y narrador | Gemma 4 — HuggingFace (desarrollo) / Ollama local en 192.168.0.135 (producción) |
| Panel web | Dash + Plotly |
| Base de datos | SQLite |
| Notificaciones | python-telegram-bot |
| Síntesis de voz | Kokoro TTS 82M (hexgrad/Kokoro-82M) — voz `ef_dora`, español |
| Config | YAML (config/config.yaml) |

---

## Estructura de módulos

```
src/
├── captura.py       # Hilo de lectura del stream. Expone una cola de frames.
├── detector.py      # Inferencia YOLO cada N frames. Lee la cola, escribe resultados.
├── rastreador.py    # Supervision/ByteTrack. Mantiene IDs de tracking entre frames.
├── zonas.py         # Carga zonas desde config.yaml. Comprueba si un bbox está en zona.
├── eventos.py       # Evalúa condiciones de hitos frame a frame. Llama a verificador.py.
├── verificador.py   # Llama a Gemma 4 de forma asíncrona: verifica el hito y genera el mensaje jocoso.
├── base_datos.py    # Inserta y consulta registros en SQLite (incluye razonamiento de Gemma).
├── notificador.py   # Telegram, TTS. Recibe el mensaje generado por Gemma y lo despacha.
├── visualizador.py  # Ventana OpenCV en directo: bboxes, IDs de tracking, límites de zona.
└── panel.py         # Servidor Dash. Lee de SQLite para las gráficas.

entrenamiento/
├── recopilar_frames.py  # Extrae frames del stream para construir el dataset.
├── preparar_dataset.py  # Convierte etiquetas Roboflow a formato YOLO, hace splits.
└── entrenar.py          # Fine-tuning de YOLO26 con el dataset preparado.

principal.py         # Arranca los hilos y conecta los módulos.
config/config.yaml   # Única fuente de verdad para parámetros.
```

---

## Fases de desarrollo

### Fase 1 — MVP (mínimo presentable)
- [x] `captura.py`: leer stream de YouTube con yt-dlp + OpenCV en hilo separado
- [x] `detector.py`: inferencia con YOLO genérico (aún sin fine-tuning) para validar el pipeline
- [x] `rastreador.py`: tracking básico con Supervision
- [x] `zonas.py`: cargar zonas desde config y comprobar pertenencia de detecciones
- [x] `eventos.py`: 5 hitos implementados con umbral de frames consecutivos y cooldown
- [x] `base_datos.py`: registro en SQLite
- [x] `notificador.py`: notificación Telegram al disparar hito
- [x] `principal.py`: orquestar todo
- [x] `visualizador.py`: ventana en directo con bboxes, IDs de tracking y límites de zona
- [x] **Pruebas de integración con stream real** — superadas

### Fase 2 — Fine-tuning
- [x] `recopilar_frames.py`: script listo. Ver instrucciones detalladas en README.
- [x] Recolección de datos: 499 imágenes capturadas y etiquetadas en Roboflow
- [x] Análisis exploratorio y corrección de etiquetas
- [x] `preparar_dataset.py`: splits estratificados + augmentación meteorológica opcional
- [x] `entrenar.py`: fine-tuning con `config/entrenamiento.yaml`
- [x] **Entrenamiento completado** — RTX 4080 Super, 100 épocas, imgsz=1280, batch=8
- [x] Modelo fine-tuned en `modelos/fauna_urbana.pt` (excluido de git por `.gitignore`)
- [x] `config/config.yaml` actualizado con las 11 clases reales del dataset

**Resultados del modelo (set de test, 93 imágenes):**

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
| **mickey_mouse** | **0.580** | Recall bajo (0.37) — confusión con minnie. Mejorable añadiendo más imágenes. |
| **global** | **0.879** | |

- [x] Prueba en vivo con stream real — superada con stream `a9J1OP_x5Rg`

### Fase 3 — Extras
- [x] Panel web (`panel.py`) — Dash + MJPEG, stats en stream, controles pausa/captura, zonas ajustadas
- [x] `simulador.py` — inyecta frames del dataset para simular hitos desde el panel
- [x] `verificador.py` — Gemma 4 vía Google AI Studio (tool calling), proveedor configurable
- [x] Email vía Google Apps Script — Gemma llama a `enviar_email` como tool call
- [x] Panel: drawer lateral de detalle, desplegable Simular mejorado, feedback "verificando..."
- [x] **Email vía GAS** — ciclo completo verificado: Gemma confirma hito → email recibido en `78818937f@cifpzonzamas.es`
- [x] **Bot de Telegram** — `src/bot_telegram.py`: push de hitos con foto + comandos `/donde`, `/cuantos`, `/captura`, `/estado`
- [x] **Drawer de hito mejorado** — imagen del frame capturado, razonamiento de Gemma, badges de acciones disparadas (email/Telegram/captura)
- [x] **Mapa de calor** — toggle "🌡 Calor" en el panel; acumulación en el punto suelo del bbox (cx, by1), radio fijo 20 px para igualar todos los personajes, decay temporal (0.990/frame), normalización por cap fijo, colormap JET; compatible con modo trayectorias activo simultáneamente
- [x] **Trayectorias** — toggle "🛤 Trayectorias" en el panel; trail por `tracker_id` (deque maxlen=100, ~4 s a 25 fps), dibujado de antiguo→reciente con fade de color y grosor, color único por ID derivado de HSV
- [x] **TTS con Kokoro — completado y verificado en producción** — `notificador.py` ejecuta Kokoro TTS 82M con voz `ef_dora` (español latinoamericano; es la única voz española del modelo, no hay variante castellana). Reproducción en el panel vía `<audio>` HTML5 con clientside callbacks (autoplay desbloqueado con `assets/audio.js` en primer click). Botón "🔊 Leer" en el drawer: toggle play/stop sin roundtrip. Cola automática: `window._audioQueue` + `window._ttsPlayNext`, reproduce hitos en orden. Toggle "🔊/🔇" en encabezado de hitos para silenciar autoplay. Normalización fonética previa: "Times Square"→"Taims Skuér", etc. Requiere: `apt install espeak-ng` + `pip install kokoro>=0.9.2 sounddevice soundfile`
- [x] **Verificación paralela de hitos** — `Verificador` usa `ThreadPoolExecutor(max_workers=3)`; múltiples hitos se verifican simultáneamente y aparecen en el panel como "⏳ verificando..." en paralelo.
- [x] **Panel de modelo — tooltips y conclusiones** — icono ℹ con tooltip CSS en cada sección de la pestaña Modelo (métricas, pérdidas, mAP50 por clase, matriz de confusión). Bloque de conclusiones al final con valoración por clase codificada por color. Tooltip de pérdidas abre hacia la izquierda (`.info-icono--izquierda`) para no salirse del drawer.
- [x] **Artefactos de entrenamiento en git** — `modelos/fauna_urbana/` ya no está en `.gitignore`; se trackean imágenes, CSV y args. Solo `modelos/fauna_urbana/weights/` sigue excluido. El panel de modelo funciona en cualquier clon del repo sin copiar archivos a mano.
- [ ] Docker
- [ ] **Audio en el panel** ← solución correcta: reemplazar MJPEG por `<video>` HTML5 + `<canvas>` + WebSocket

---

## Estado actual

- [x] Planning y propuesta definidos
- [x] Repositorio creado en GitHub
- [x] Entorno virtual y dependencias instaladas
- [x] `captura.py`: hilo de lectura del stream, cliente android yt-dlp (sin cookies)
- [x] `detector.py`: inferencia YOLO cada N frames, fallback automático a modelo genérico
- [x] `rastreador.py`: tracking con Supervision/ByteTrack
- [x] `zonas.py`: carga zonas desde config, filtra detecciones con PolygonZone
- [x] `eventos.py`: 5 hitos con umbral de frames consecutivos, cooldown y cola de salida
- [x] `verificador.py`: Gemma 4 (Google AI Studio) con tool calling — confirma hito y envía email
- [x] `base_datos.py`: registro en SQLite con razonamiento de Gemma
- [x] `notificador.py`: guarda frame, registra en BD, delega Telegram a BotTelegram y TTS
- [x] `bot_telegram.py`: bot unificado — push de hitos + comandos interactivos
- [x] `simulador.py`: simula hitos inyectando frames del dataset, pausa el stream real
- [x] `panel.py`: Dash completo con stream, hitos, simulador, drawer de detalle, mapa de calor con toggle, trayectorias por tracker_id con toggle, TTS con cola automática y toggle 🔊/🔇, pestaña Modelo con tooltips ℹ y conclusiones por clase
- [x] `principal.py`: orquesta todos los hilos con arranque y parada ordenados

### Pendiente al retomar

- **Stream activo**: el principal (`rnXIjl_Rzy4`) está caído. Usar el secundario `https://www.youtube.com/watch?v=a9J1OP_x5Rg`.
- **Gemma en producción**: cambiar `GEMMA_PROVEEDOR=ollama` en `.env` para usar Gemma 4 local en `192.168.0.135` (en desarrollo se usa `google` con Google AI Studio).
- **mickey_mouse**: recall bajo (0.37) — recolectar más imágenes si falla en producción.
- **transformer**: detección débil observada en producción — apunte para futura iteración del dataset, no prioritario.
