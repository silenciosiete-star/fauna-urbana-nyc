# Fauna Urbana NYC

Agente de visión por ordenador que monitoriza en tiempo real la fauna de personajes disfrazados de Times Square: gorilas, Spider-Man, Deadpool, Mickey y Minnie Mouse, y cualquier criatura que se cruce por ahí.

---

## Idea general

Times Square tiene su propia vida salvaje. Este sistema analiza el stream en vivo de YouTube, detecta y clasifica a los personajes disfrazados, los rastrea por la imagen, y reacciona cuando ocurren situaciones dignas de atención (o de risa).

---

## Funcionalidades previstas

### Detección y clasificación
- YOLO v11 reentrenado con un dataset propio de personajes de Times Square
- Clasificación de personajes: gorila, Spider-Man, Deadpool, Mickey Mouse, Minnie Mouse, y otros
- Conteo en tiempo real por tipo de personaje

### Zonas dinámicas
- Definición interactiva de zonas de interés sobre la imagen
- Conteo independiente por zona
- Detección de qué personaje ocupa qué zona en cada momento

### Hitos y alertas
Situaciones que disparan acciones automáticas:

| Hito | Descripción | Acción |
|------|-------------|--------|
| **Avengers Assemble** | 3 o más superhéroes en la misma zona | Email + captura del frame |
| **Conflicto de identidad** | Dos personajes iguales en la misma zona | Notificación Telegram + registro |
| **Hora punta de la fauna** | Más de N personajes simultáneos | Alerta + guardado automático |
| **Avistamiento raro** | Personaje no visto en las últimas horas | Notificación Telegram |
| **Marvel vs DC** | Spider-Man y Batman detectados a la vez | TTS lo anuncia por altavoz |

### Seguimiento y análisis
- Tracking de trayectorias por personaje (con color diferente para cada uno)
- Mapa de calor de zonas con más actividad
- Historial de apariciones en SQLite
- Guardado automático de frames en los hitos

### Panel web en tiempo real
- Conteo actual por personaje
- Histórico y gráficas temporales
- Ranking de "celebridades" (quién aparece más y en qué horarios)
- Localización actual de personajes bajo petición

### Control por Telegram
- Bot para consultar el estado en tiempo real
- Comandos: `/donde gorila`, `/cuantos ahora`, `/captura`
- Configuración de alertas desde el móvil

### Descripción de escenas con LLM
- Gemma 4 (local vía Ollama u OpenRouter) describe la situación cuando se dispara un hito
- Ejemplo: *"Hay dos Spider-Men disputándose la esquina norte mientras un gorila observa desde el fondo"*

### Síntesis de voz
- Anuncio por altavoz de los eventos más destacados

---

## Tecnologías

| Componente | Tecnología |
|------------|------------|
| Captura del stream | OpenCV + yt-dlp |
| Detección y clasificación | YOLO v11 (fine-tuned) |
| Segmentación | SAM3 |
| Tracking | Supervision |
| Descripción de escenas | Gemma 4 (Ollama / OpenRouter) |
| Panel web | Dash + Plotly |
| Base de datos | SQLite |
| Notificaciones | python-telegram-bot |
| Síntesis de voz | pyttsx3 / Coqui TTS |
| Configuración | YAML |

---

## Estructura del proyecto

```
fauna-urbana-nyc/
├── config/              # Configuración: zonas, hitos, umbrales
├── datos/               # Datasets para entrenamiento
├── modelos/             # Pesos del modelo entrenado
├── src/
│   ├── captura.py       # Lectura del stream de YouTube
│   ├── detector.py      # Inferencia con YOLO
│   ├── rastreador.py    # Tracking con Supervision
│   ├── zonas.py         # Gestión de zonas interactivas
│   ├── eventos.py       # Hitos y acciones asociadas
│   ├── base_datos.py    # Registro en SQLite
│   ├── notificador.py   # Telegram, email, TTS
│   └── panel.py         # Dashboard web
├── entrenamiento/
│   ├── preparar_dataset.py
│   └── entrenar.py
├── capturas/            # Frames guardados en hitos
└── principal.py         # Punto de entrada
```

---

## Estado del proyecto

- [ ] Estructura base y captura del stream
- [ ] Pipeline de detección con YOLO genérico
- [ ] Preparación del dataset de personajes
- [ ] Fine-tuning de YOLO v11
- [ ] Sistema de zonas interactivas
- [ ] Tracking y trayectorias
- [ ] Hitos y acciones
- [ ] Registro en base de datos
- [ ] Panel web
- [ ] Bot de Telegram
- [ ] Descripción de escenas con Gemma 4
- [ ] Síntesis de voz
- [ ] Mapa de calor
