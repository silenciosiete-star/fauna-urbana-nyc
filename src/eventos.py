"""Evalúa condiciones de hitos frame a frame. Publica HitoPotencial cuando se cumplen N frames consecutivos."""
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from loguru import logger

from .rastreador import ResultadoTracking
from .zonas import Zona, detecciones_en_zona

_SUPERHEROES = {"spiderman", "deadpool", "batman"}


@dataclass
class HitoPotencial:
    tipo: str
    frame: np.ndarray
    descripcion: str
    marca_tiempo: float = field(default_factory=time.time)


class GestorEventos:

    def __init__(
        self,
        cola_entrada: queue.Queue,
        config_hitos: dict,
        zonas: dict[str, Zona],
        clases_modelo: dict[int, str],
    ):
        self.cola_entrada = cola_entrada
        self.cola_salida: queue.Queue = queue.Queue(maxsize=20)
        self._config = config_hitos
        self._zonas = zonas
        self._clases = clases_modelo
        self._umbral = config_hitos["frames_consecutivos"]

        # Contadores de frames consecutivos por hito
        self._consecutivos: dict[str, int] = {tipo: 0 for tipo in _tipos_activos(config_hitos)}

        # Timestamps del último disparo por hito (para cooldown)
        self._ultimo_disparo: dict[str, float] = {}

        # Último avistamiento por clase (para avistamiento_raro)
        self._ultimo_avistamiento: dict[str, float] = {}

        self._activo = False
        self._hilo: threading.Thread | None = None

    def iniciar(self) -> None:
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle_eventos, daemon=True)
        self._hilo.start()
        logger.info("Gestor de eventos iniciado")

    def detener(self) -> None:
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=10)
        logger.info("Gestor de eventos detenido")

    # ------------------------------------------------------------------

    def _bucle_eventos(self) -> None:
        while self._activo:
            try:
                resultado: ResultadoTracking = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            clases_presentes = _clases_en_detecciones(resultado.detecciones)
            self._evaluar_hitos(resultado, clases_presentes)
            self._actualizar_avistamientos(clases_presentes, resultado.marca_tiempo)

    def _evaluar_hitos(self, resultado: ResultadoTracking, clases_presentes: set[str]) -> None:
        cfg = self._config

        self._evaluar(
            tipo="avengers_assemble",
            condicion=cfg.get("avengers_assemble", {}).get("activo", False)
            and len(clases_presentes & _SUPERHEROES) >= cfg["avengers_assemble"]["superheroes_minimos"],
            resultado=resultado,
            descripcion=f"{len(clases_presentes & _SUPERHEROES)} superhéroes simultáneos: {clases_presentes & _SUPERHEROES}",
            cooldown=cfg["avengers_assemble"]["cooldown_segundos"],
        )

        self._evaluar(
            tipo="hora_punta",
            condicion=cfg.get("hora_punta", {}).get("activo", False)
            and len(resultado.detecciones) >= cfg["hora_punta"]["personajes_minimos"],
            resultado=resultado,
            descripcion=f"{len(resultado.detecciones)} personajes visibles simultáneamente",
            cooldown=cfg["hora_punta"]["cooldown_segundos"],
        )

        self._evaluar(
            tipo="marvel_vs_dc",
            condicion=cfg.get("marvel_vs_dc", {}).get("activo", False)
            and cfg["marvel_vs_dc"]["personaje_marvel"] in clases_presentes
            and cfg["marvel_vs_dc"]["personaje_dc"] in clases_presentes,
            resultado=resultado,
            descripcion=f"{cfg['marvel_vs_dc']['personaje_marvel']} y {cfg['marvel_vs_dc']['personaje_dc']} simultáneos",
            cooldown=cfg["marvel_vs_dc"]["cooldown_segundos"],
        )

        self._evaluar(
            tipo="conflicto_identidad",
            condicion=cfg.get("conflicto_identidad", {}).get("activo", False)
            and self._hay_conflicto_identidad(resultado.detecciones),
            resultado=resultado,
            descripcion="Dos personajes del mismo tipo en la misma zona",
            cooldown=cfg["conflicto_identidad"]["cooldown_segundos"],
        )

        avistamiento = self._clase_para_avistamiento_raro(
            clases_presentes, cfg, resultado.marca_tiempo
        )
        self._evaluar(
            tipo="avistamiento_raro",
            condicion=cfg.get("avistamiento_raro", {}).get("activo", False)
            and avistamiento is not None,
            resultado=resultado,
            descripcion=f"Reaparición de {avistamiento} tras ausencia prolongada",
            cooldown=cfg["avistamiento_raro"]["cooldown_segundos"],
        )

    def _evaluar(
        self,
        tipo: str,
        condicion: bool,
        resultado: ResultadoTracking,
        descripcion: str,
        cooldown: int,
    ) -> None:
        if condicion:
            self._consecutivos[tipo] = self._consecutivos.get(tipo, 0) + 1
        else:
            self._consecutivos[tipo] = 0
            return

        if self._consecutivos[tipo] < self._umbral:
            return

        ahora = resultado.marca_tiempo
        ultimo = self._ultimo_disparo.get(tipo, 0)
        if ahora - ultimo < cooldown:
            return

        self._consecutivos[tipo] = 0
        self._ultimo_disparo[tipo] = ahora

        hito = HitoPotencial(tipo=tipo, frame=resultado.frame, descripcion=descripcion)
        if self.cola_salida.full():
            try:
                self.cola_salida.get_nowait()
            except queue.Empty:
                pass
        self.cola_salida.put(hito)
        logger.info(f"Hito potencial detectado: {tipo} — {descripcion}")

    def _hay_conflicto_identidad(self, detecciones: sv.Detections) -> bool:
        for zona in self._zonas.values():
            en_zona = detecciones_en_zona(detecciones, zona)
            if len(en_zona) < 2:
                continue
            clases = _clases_en_detecciones(en_zona)
            # Hay conflicto si alguna clase aparece más de una vez en la zona
            nombres = list(en_zona.data.get("class_name", []))
            if len(nombres) != len(set(nombres)):
                return True
        return False

    def _actualizar_avistamientos(self, clases_presentes: set[str], marca_tiempo: float) -> None:
        for clase in clases_presentes:
            self._ultimo_avistamiento[clase] = marca_tiempo

    def _clase_para_avistamiento_raro(
        self, clases_presentes: set[str], cfg: dict, ahora: float
    ) -> str | None:
        ausencia_seg = cfg["avistamiento_raro"]["ausencia_minutos"] * 60
        for clase in clases_presentes:
            ultimo = self._ultimo_avistamiento.get(clase)
            if ultimo is not None and (ahora - ultimo) >= ausencia_seg:
                return clase
        return None


# ------------------------------------------------------------------
# Utilidades de módulo

def _clases_en_detecciones(detecciones: sv.Detections) -> set[str]:
    nombres = detecciones.data.get("class_name", np.array([]))
    return set(nombres.tolist())


def _tipos_activos(config_hitos: dict) -> list[str]:
    tipos = ["avengers_assemble", "conflicto_identidad", "hora_punta", "avistamiento_raro", "marvel_vs_dc"]
    return [t for t in tipos if config_hitos.get(t, {}).get("activo", False)]
