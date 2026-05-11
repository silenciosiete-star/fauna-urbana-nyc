"""Punto de entrada. Arranca todos los hilos y sirve el panel con waitress."""
import sys

import waitress
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
from src.bot_telegram import BotTelegram
from src.notificador import Notificador
from src.panel import Panel
from src.simulador import Simulador

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
        config_notificaciones=config["notificaciones"],
    )

    base_datos = BaseDatos(ruta=config["base_datos"]["ruta"])

    bot_telegram = None
    if config.get("notificaciones", {}).get("telegram", {}).get("activo", False):
        bot_telegram = BotTelegram(base_datos=base_datos, rastreador=rastreador)

    notificador = Notificador(
        cola_entrada=verificador.cola_salida,
        base_datos=base_datos,
        config_notificaciones=config["notificaciones"],
        config_capturas=config["capturas"],
        bot_telegram=bot_telegram,
    )

    simulador = Simulador(
        capturador=capturador,
        gestor_eventos=gestor_eventos,
    )
    panel = Panel(
        cola_frames=capturador.cola_display,
        cola_tracking=rastreador.cola_display,
        zonas=zonas,
        base_datos=base_datos,
        puerto=config["panel"]["puerto"],
        carpeta_capturas=config["capturas"]["carpeta"],
    )
    panel.conectar_simulador(simulador)
    panel.conectar_verificador(verificador)
    panel.conectar_notificador(notificador)
    panel.conectar_gestor_eventos(gestor_eventos)

    modulos = [capturador, detector, rastreador, gestor_eventos, verificador, notificador]
    if bot_telegram:
        modulos.append(bot_telegram)
    modulos.append(panel)

    logger.info("Iniciando Fauna Urbana NYC...")
    for modulo in modulos:
        modulo.iniciar()

    puerto = config["panel"]["puerto"]
    logger.info(f"Sistema activo. Panel: http://localhost:{puerto}")
    try:
        # waitress instala sus propios handlers de SIGINT/SIGTERM
        # y retorna limpiamente cuando llega la señal.
        waitress.serve(
            panel.app_wsgi(),
            host="0.0.0.0",
            port=puerto,
            threads=8,
        )
    finally:
        logger.info("Deteniendo módulos...")
        for modulo in reversed(modulos):
            try:
                modulo.detener()
            except Exception as error:
                logger.error(f"Error deteniendo módulo: {error}")
    sys.exit(0)


if __name__ == "__main__":
    main()
