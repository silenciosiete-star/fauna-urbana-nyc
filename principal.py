"""Punto de entrada. Arranca todos los hilos y los conecta."""
import os
import signal
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.captura import CapturadorStream
from src.detector import Detector
from src.rastreador import Rastreador
from src.zonas import cargar_zonas
from src.eventos import GestorEventos
from src.verificador import Verificador
from src.base_datos import BaseDatos
from src.notificador import Notificador
from src.panel import Panel
from src.visualizador import Visualizador

load_dotenv()


def cargar_config(ruta: str = "config/config.yaml") -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = cargar_config()

    capturador = CapturadorStream(url=config["stream"]["url"])

    detector = Detector(
        cola_entrada=capturador.cola,
        ruta_modelo=config["modelo"]["ruta"],
        frames_por_inferencia=config["stream"]["frames_por_inferencia"],
        confianza_minima=config["modelo"]["confianza_minima"],
    )

    rastreador = Rastreador(cola_entrada=detector.cola_salida)

    zonas = cargar_zonas(config["zonas"])

    gestor_eventos = GestorEventos(
        cola_entrada=rastreador.cola_salida,
        config_hitos=config["hitos"],
        zonas=zonas,
        clases_modelo=config["gemma"]["clases"],
    )

    verificador = Verificador(
        cola_entrada=gestor_eventos.cola_salida,
        config_gemma=config["gemma"],
    )

    base_datos = BaseDatos(ruta=config["base_datos"]["ruta"])

    notificador = Notificador(
        cola_entrada=verificador.cola_salida,
        base_datos=base_datos,
        config_notificaciones=config["notificaciones"],
        config_capturas=config["capturas"],
    )

    if config.get("panel", {}).get("activo", False):
        interfaz = Panel(
            cola_frames=capturador.cola_display,
            cola_tracking=rastreador.cola_display,
            zonas=zonas,
            base_datos=base_datos,
            puerto=config["panel"]["puerto"],
            carpeta_capturas=config["capturas"]["carpeta"],
        )
    else:
        interfaz = Visualizador(
            cola_frames=capturador.cola_display,
            cola_tracking=rastreador.cola_display,
            zonas=zonas,
        )

    modulos = [capturador, detector, rastreador, gestor_eventos, verificador, notificador, interfaz]

    def apagar(sig, frame):
        logger.info("Señal de parada recibida. Deteniendo módulos...")
        for modulo in reversed(modulos):
            modulo.detener()
        sys.exit(0)

    signal.signal(signal.SIGINT, apagar)
    signal.signal(signal.SIGTERM, apagar)

    logger.info("Iniciando Fauna Urbana NYC...")
    for modulo in modulos:
        modulo.iniciar()

    logger.info("Sistema activo. Pulsa Ctrl+C para detener.")
    signal.pause()


if __name__ == "__main__":
    main()
