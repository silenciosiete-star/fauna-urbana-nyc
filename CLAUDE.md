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
| Dos proveedores para Gemma según entorno | Desarrollo: HuggingFace Inference API. Producción: Ollama en servidor de red local (192.168.0.135). Se cambia con `GEMMA_PROVEEDOR` en `.env`, sin tocar código. |
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
| Síntesis de voz | pyttsx3 |
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

- [ ] Prueba en vivo con stream real ← **PENDIENTE** (cámara offline al momento de las pruebas)

### Fase 3 — Extras
- [x] Panel web (`panel.py`) — Dash + MJPEG, stats en stream, controles pausa/captura, zonas ajustadas
- [ ] **Audio en el panel** ← PENDIENTE: reemplazar MJPEG por `<video>` HTML5 (URL directa vía yt-dlp) + `<canvas>` superpuesto con anotaciones enviadas por WebSocket (Flask-Sock). Patrón estándar de dashboards CCTV profesionales.
- [ ] Bot de Telegram con comandos interactivos
- [ ] Descripción de escenas con Gemma 4 en `notificador.py`
- [ ] Síntesis de voz (TTS)
- [ ] Mapa de calor
- [ ] Docker

---

## Estado actual

- [x] Planning y propuesta definidos
- [x] Repositorio creado en GitHub
- [x] Entorno virtual y dependencias instaladas
- [x] `captura.py`: hilo de lectura del stream con reconexión automática
- [x] `detector.py`: inferencia YOLO cada N frames, fallback automático a modelo genérico
- [x] `rastreador.py`: tracking con Supervision/ByteTrack
- [x] `zonas.py`: carga zonas desde config, filtra detecciones con PolygonZone
- [x] `eventos.py`: 5 hitos con umbral de frames consecutivos, cooldown y cola de salida
- [x] `verificador.py`: llama a Gemma (HuggingFace o Ollama) con imagen en base64
- [x] `base_datos.py`: registro en SQLite con razonamiento de Gemma
- [x] `notificador.py`: guarda frame, registra en BD, envía Telegram y TTS
- [x] `principal.py`: orquesta todos los hilos con arranque y parada ordenados

### Pendiente al retomar
- **Prueba en vivo (Fase 2)**: conectar el stream cuando la cámara vuelva y verificar detecciones reales.
- **Dos modelos en `detector.py`**: añadir modelo pretrained COCO para vehículos en paralelo al modelo de personajes. La decisión arquitectónica está tomada (ver tabla de arriba).
- **mickey_mouse**: si el recall sigue bajo en producción, recolectar más imágenes y reentrenar.
- **Audio en el panel (Fase 3)**: ver tarea pendiente arriba.
