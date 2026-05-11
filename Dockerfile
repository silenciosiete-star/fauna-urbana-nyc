# syntax=docker/dockerfile:1.6

# ──────────────────── BUILDER ────────────────────
# Compila wheels una sola vez para acelerar rebuilds.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ──────────────────── RUNTIME ────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Madrid

# ffmpeg     → yt-dlp / OpenCV para leer el stream HLS
# espeak-ng  → backend fonético de Kokoro TTS
# libsndfile → lectura/escritura de WAV (soundfile)
# libglib    → dependencia de tiempo de ejecución de opencv-headless
# tini       → PID 1 robusto: propaga SIGTERM y cosecha zombies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
        libglib2.0-0 \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Código y assets
COPY src/ src/
COPY config/ config/
COPY assets/ assets/
COPY principal.py .

# UID/GID del usuario del contenedor — deben coincidir con los del host
# para que los bind-mounts (datos_bd, capturas) sean escribibles.
# Por defecto 1000 (estándar primer usuario Linux). Override:
#   docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
ARG UID=1000
ARG GID=1000

# Directorios para volúmenes y usuario no-root
RUN mkdir -p capturas datos_bd modelos \
    && groupadd --gid ${GID} fauna \
    && useradd --create-home --uid ${UID} --gid ${GID} fauna \
    && chown -R fauna:fauna /app

USER fauna

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/', timeout=5)" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["python", "principal.py"]
