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
| YOLO v11 fine-tuned como único modelo de detección | Clasifica y detecta en un solo paso, rápido |
| SAM3 descartado | Bounding boxes son suficientes para todos los casos de uso del proyecto |
| Gemma 4 solo cuando salta un hito | No es viable llamarla por frame; así no hay coste de rendimiento |
| Captura e inferencia en hilos separados | Evita que un frame lento de YOLO bloquee la lectura del stream |
| Procesar 1 de cada N frames (configurable) | Supervision/ByteTrack interpola el tracking entre frames no analizados |
| Hitos con umbral de 5 frames consecutivos | Evita falsos positivos por detecciones puntuales |

---

## Stack

| Componente | Tecnología |
|------------|------------|
| Captura del stream | OpenCV + yt-dlp |
| Detección y clasificación | YOLO v11 (ultralytics) |
| Tracking | Supervision + ByteTrack |
| LLM para descripción | Gemma 4 vía Ollama o OpenRouter |
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
├── eventos.py       # Evalúa condiciones de hitos frame a frame. Dispara acciones.
├── base_datos.py    # Inserta y consulta registros en SQLite.
├── notificador.py   # Telegram, TTS. Recibe mensajes y los despacha.
└── panel.py         # Servidor Dash. Lee de SQLite para las gráficas.

entrenamiento/
├── recopilar_frames.py  # Extrae frames del stream para construir el dataset.
├── preparar_dataset.py  # Convierte etiquetas Roboflow a formato YOLO, hace splits.
└── entrenar.py          # Fine-tuning de YOLO v11 con el dataset preparado.

principal.py         # Arranca los hilos y conecta los módulos.
config/config.yaml   # Única fuente de verdad para parámetros.
```

---

## Fases de desarrollo

### Fase 1 — MVP (mínimo presentable)
- [ ] `captura.py`: leer stream de YouTube con yt-dlp + OpenCV en hilo separado
- [ ] `detector.py`: inferencia con YOLO genérico (aún sin fine-tuning) para validar el pipeline
- [ ] `rastreador.py`: tracking básico con Supervision
- [ ] `zonas.py`: cargar zonas desde config y comprobar pertenencia de detecciones
- [ ] `eventos.py`: 3 hitos funcionando con guardado de frame
- [ ] `base_datos.py`: registro en SQLite
- [ ] `notificador.py`: notificación Telegram al disparar hito
- [ ] `principal.py`: orquestar todo

### Fase 2 — Fine-tuning
- [ ] `recopilar_frames.py`: extraer frames del stream
- [ ] Etiquetado en Roboflow
- [ ] `preparar_dataset.py`: preparar dataset en formato YOLO
- [ ] `entrenar.py`: fine-tuning y evaluación (mAP por clase)
- [ ] Sustituir modelo genérico por el fine-tuned en `detector.py`

### Fase 3 — Extras
- [ ] Panel web (`panel.py`)
- [ ] Bot de Telegram con comandos interactivos
- [ ] Descripción de escenas con Gemma 4 en `notificador.py`
- [ ] Síntesis de voz (TTS)
- [ ] Mapa de calor
- [ ] Docker

---

## Estado actual

- [x] Planning y propuesta definidos
- [x] Repositorio creado en GitHub
- [ ] Fase 1 en curso
