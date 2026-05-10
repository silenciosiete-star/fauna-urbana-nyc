"""Evalúa condiciones de hitos frame a frame. Publica HitoPotencial cuando se cumplen N frames consecutivos."""
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from .rastreador import ResultadoTracking
from .zonas import Zona, detecciones_en_zona

_UNIVERSOS: dict[str, str] = {
    "spiderman":        "marvel",
    "deadpool":         "marvel",
    "batman":           "dc",
    "mickey_mouse":     "disney",
    "minnie_mouse":     "disney",
    "sonic":            "sega",
    "super_mario":      "nintendo",
    "elmo":             "sesame",
    "estatua_libertad": "nyc",
    "gorila":           "nyc",
    "transformer":      "nyc",
}


@dataclass
class HitoPotencial:
    tipo: str
    frame: np.ndarray
    descripcion: str
    detecciones_str: str = ""  # lista legible de lo que YOLO tiene en ese frame
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

        # Zonas personalizadas para conflicto_identidad (None = usar self._zonas)
        self._zonas_conflicto: dict[str, Zona] | None = None

        self._activo = False
        self._hilo: threading.Thread | None = None
        self.en_pausa: bool = False

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

    def preparar_simulacion(self, tipo: str) -> None:
        # Pone en cooldown todos los hitos excepto el objetivo para que solo
        # dispare el que se quiere simular, aunque la imagen contenga más personajes.
        # Guarda los cooldowns actuales para restaurarlos al terminar.
        self._cooldowns_guardados = {
            t: self._ultimo_disparo.get(t) for t in self._consecutivos if t != tipo
        }
        cooldown_largo = 9999.0
        for t in self._consecutivos:
            if t != tipo:
                self._ultimo_disparo[t] = time.time() + cooldown_largo
        self._ultimo_disparo.pop(tipo, None)
        self._consecutivos[tipo] = self._umbral - 1
        if tipo == "avistamiento_raro":
            self._ultimo_avistamiento["gorila"] = time.time() - 31 * 60

    def restaurar_cooldowns(self) -> None:
        for t, v in getattr(self, "_cooldowns_guardados", {}).items():
            if v is None:
                self._ultimo_disparo.pop(t, None)
            else:
                self._ultimo_disparo[t] = v
        self._cooldowns_guardados = {}

    def actualizar_zonas_conflicto(self, zonas_custom: dict[str, Zona] | None) -> None:
        """Reemplaza las zonas usadas para conflicto_identidad. None restaura las del YAML."""
        self._zonas_conflicto = zonas_custom
        if zonas_custom is None:
            logger.info("Zonas conflicto_identidad: restauradas desde YAML")
        else:
            logger.info(f"Zonas conflicto_identidad: {list(zonas_custom.keys())} (personalizadas)")

    # ------------------------------------------------------------------

    def _bucle_eventos(self) -> None:
        while self._activo:
            try:
                resultado: ResultadoTracking = self.cola_entrada.get(timeout=1)
            except queue.Empty:
                continue

            if self.en_pausa:
                continue

            clases_presentes = _clases_en_detecciones(resultado.detecciones)
            self._evaluar_hitos(resultado, clases_presentes)
            self._actualizar_avistamientos(clases_presentes, resultado.marca_tiempo)

    def _evaluar_hitos(self, resultado: ResultadoTracking, clases_presentes: set[str]) -> None:
        cfg = self._config

        universos_presentes = {_UNIVERSOS[c] for c in clases_presentes if c in _UNIVERSOS}
        self._evaluar(
            tipo="crossover",
            condicion=cfg.get("crossover", {}).get("activo", False)
            and len(universos_presentes) >= cfg["crossover"]["universos_minimos"],
            resultado=resultado,
            descripcion=f"{len(universos_presentes)} universos simultáneos: {universos_presentes}",
            cooldown=cfg["crossover"]["cooldown_segundos"],
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

        resultado_conflicto = self._clase_conflicto_identidad(resultado.detecciones)
        zonas_conflicto = self._zonas_conflicto if self._zonas_conflicto is not None else self._zonas
        self._evaluar(
            tipo="conflicto_identidad",
            condicion=cfg.get("conflicto_identidad", {}).get("activo", False)
            and resultado_conflicto is not None,
            resultado=resultado,
            descripcion=(
                f"{resultado_conflicto[0]}×2 en zona «{resultado_conflicto[1]}»"
                if resultado_conflicto else ""
            ),
            cooldown=cfg["conflicto_identidad"]["cooldown_segundos"],
            zonas=zonas_conflicto,
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
        zonas: dict[str, Zona] | None = None,
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

        nombres_todos = list(resultado.detecciones.data.get("class_name", np.array([])))
        conteo = Counter(n for n in nombres_todos if n)
        detecciones_str = ", ".join(
            f"{c}×{n}" if n > 1 else c for c, n in sorted(conteo.items())
        ) or "ninguno"
        hito = HitoPotencial(
            tipo=tipo,
            frame=_anotar_frame(resultado.frame, resultado.detecciones, zonas),
            descripcion=descripcion,
            detecciones_str=detecciones_str,
        )
        if self.cola_salida.full():
            try:
                self.cola_salida.get_nowait()
            except queue.Empty:
                pass
        self.cola_salida.put(hito)
        logger.info(f"Hito potencial detectado: {tipo} — {descripcion}")

    def _clase_conflicto_identidad(self, detecciones: sv.Detections) -> tuple[str, str] | None:
        zonas = self._zonas_conflicto if self._zonas_conflicto is not None else self._zonas
        for nombre_zona, zona in zonas.items():
            en_zona = detecciones_en_zona(detecciones, zona)
            if len(en_zona) < 2:
                continue
            nombres = list(en_zona.data.get("class_name", []))
            visto: set[str] = set()
            for nombre in nombres:
                if nombre in visto:
                    return nombre, nombre_zona
                visto.add(nombre)
        return None

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

def _anotar_frame(
    frame: np.ndarray,
    detecciones: sv.Detections,
    zonas: dict | None = None,
) -> np.ndarray:
    if (detecciones is None or len(detecciones) == 0) and not zonas:
        return frame
    frame = frame.copy()
    if zonas:
        for nombre, zona in zonas.items():
            pts = zona.poligono.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 220), thickness=2)
            x, y = zona.poligono.polygon[0]
            cv2.putText(frame, f"[{nombre}]", (int(x) + 4, int(y) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 220), 1, cv2.LINE_AA)
    if detecciones is not None and len(detecciones) > 0:
        nombres = detecciones.data.get("class_name", np.array([]))
        etiquetas = [f"{nombres[i]} {detecciones.confidence[i]:.0%}" for i in range(len(detecciones))]
        frame = sv.BoxAnnotator().annotate(scene=frame, detections=detecciones)
        frame = sv.LabelAnnotator().annotate(scene=frame, detections=detecciones, labels=etiquetas)
    return frame


def _clases_en_detecciones(detecciones: sv.Detections) -> set[str]:
    nombres = detecciones.data.get("class_name", np.array([]))
    return set(nombres.tolist())


def _tipos_activos(config_hitos: dict) -> list[str]:
    tipos = ["crossover", "conflicto_identidad", "hora_punta", "avistamiento_raro", "marvel_vs_dc"]
    return [t for t in tipos if config_hitos.get(t, {}).get("activo", False)]
