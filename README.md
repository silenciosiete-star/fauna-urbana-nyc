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
| Descripción de escenas | Gemma 4 (Ollama / OpenRouter) | Solo se llama cuando salta un hito, no por frame |
| Panel web | Dash + Plotly | |
| Base de datos | SQLite | |
| Notificaciones | python-telegram-bot | Alertas y control remoto |
| Síntesis de voz | pyttsx3 / Coqui TTS | |
| Configuración | YAML | Zonas, hitos y umbrales sin tocar código |

> **SAM3 descartado:** la segmentación pixel a pixel no aporta lo suficiente para el uso real del proyecto (conteo, tracking, zonas, heatmap) como para justificar su coste computacional.

---

## Arquitectura de procesamiento

La captura y la inferencia van en **hilos separados** para que un frame lento del modelo no bloquee la lectura del stream:

```
[Hilo captura]  →  cola de frames  →  [Hilo inferencia]  →  [Hilo UI / panel]
   yt-dlp                                YOLO + Supervision
   OpenCV                                cada N frames
```

---

## Dataset para el fine-tuning

El punto más crítico del proyecto. Plan de construcción:

1. **Recopilar frames** del propio stream de Times Square en distintos horarios
2. **Etiquetar** con Roboflow (clases: `gorila`, `spider-man`, `deadpool`, `mickey`, `minnie`, `otro-disfraz`)
3. **Aumentado de datos**: rotaciones, cambios de brillo, recortes — los personajes aparecen a distintas distancias y con distintas iluminaciones
4. **Validación**: reservar un 20% del dataset para test, medir mAP por clase

Clases objetivo iniciales: `gorila` · `spider-man` · `deadpool` · `mickey` · `minnie`

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
- [ ] Descripción de la escena con Gemma 4 cuando salta un hito
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
│   ├── eventos.py           # Detección de hitos y acciones
│   ├── base_datos.py        # Registro en SQLite
│   ├── notificador.py       # Telegram, email, TTS
│   └── panel.py             # Dashboard web (Dash)
├── entrenamiento/
│   ├── recopilar_frames.py  # Extrae frames del stream para el dataset
│   ├── preparar_dataset.py  # Conversión y splits train/val/test
│   └── entrenar.py          # Fine-tuning de YOLO v11
├── capturas/                # Frames guardados al dispararse un hito
└── principal.py             # Punto de entrada
```
