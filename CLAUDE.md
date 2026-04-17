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
| YOLO v11 fine-tuned como único modelo de detección | Clasifica y detecta en un solo paso, rápido. Dataset mixto: personajes (anotados a mano) + vehículos COCO (ya anotados). Un solo modelo cubre ambas zonas. |
| SAM3 descartado | Bounding boxes son suficientes para todos los casos de uso del proyecto |
| Gemma 4 como verificador y narrador de hitos | YOLO detecta la condición; Gemma confirma con criterio semántico y redacta la notificación. Se llama de forma asíncrona para no congelar el stream. |
| Dos proveedores para Gemma según entorno | Desarrollo: HuggingFace Inference API. Producción: Ollama en servidor de red local (192.168.0.135). Se cambia con `GEMMA_PROVEEDOR` en `.env`, sin tocar código. |
| Captura e inferencia en hilos separados | Evita que un frame lento de YOLO bloquee la lectura del stream |
| Procesar 1 de cada N frames (configurable) | Supervision/ByteTrack interpola el tracking entre frames no analizados |
| imgsz pendiente de ajustar en fine-tuning | La cámara es lejana y los personajes son objetos pequeños. Valorar subir imgsz a 1280 o recortar la zona derecha del frame antes de la inferencia. Tradeoff velocidad/precisión. |
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
| Detección y clasificación | YOLO v11 (ultralytics) |
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
- [x] Entorno virtual y dependencias instaladas
- [x] `captura.py`: hilo de lectura del stream con reconexión automática
- [x] `detector.py`: inferencia YOLO cada N frames, fallback automático a modelo genérico
- [ ] `rastreador.py`: tracking con Supervision/ByteTrack
- [ ] `zonas.py`, `eventos.py`, `verificador.py`, `base_datos.py`, `notificador.py`, `principal.py`

### Pendiente al retomar
- Verificar que `ver_detecciones.py` muestra el stream con detecciones (posible bloqueo temporal de YouTube por IP — el código es correcto)
- Continuar con `rastreador.py`
