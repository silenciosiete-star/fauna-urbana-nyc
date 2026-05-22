"""Simula hitos inyectando frames del dataset en el pipeline real hasta que el modelo dispara."""
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .captura import CapturadorStream
from .eventos import GestorEventos

_DIR_SIMULACIONES = Path(__file__).parent.parent / "assets" / "simulaciones"
_INTERVALO_INYECCION_S = 0.12   # ~8 frames/s → suficiente para que YOLO procese entre inyecciones
_TIMEOUT_S = 12.0               # tiempo máximo esperando que el modelo detecte


class Simulador:

    def __init__(
        self,
        capturador: CapturadorStream,
        gestor_eventos: GestorEventos,
    ):
        self._capturador = capturador
        self._gestor = gestor_eventos
        self.simulando: str | None = None
        self.frame_override: np.ndarray | None = None
        self._lock = threading.Lock()

    def simular(self, tipo: str) -> None:
        threading.Thread(target=self._ejecutar, args=(tipo,), daemon=True).start()

    def _ejecutar(self, tipo: str) -> None:
        ruta_img = _DIR_SIMULACIONES / f"{tipo}.jpg"
        if not ruta_img.exists():
            return

        frame = cv2.imread(str(ruta_img))
        if frame is None:
            return

        pausado_antes = self._capturador.pausado
        self._gestor.preparar_simulacion(tipo)
        self._capturador.pausado = True

        with self._lock:
            self.simulando = tipo
            self.frame_override = frame

        inicio = time.time()
        disparo_antes = self._gestor._ultimo_disparo.get(tipo, 0)
        hito_detectado = False

        while time.time() - inicio < _TIMEOUT_S:
            # Re-sembrar el contador en cada iteración: cualquier frame "en
            # vuelo" del stream real que llegue al gestor antes que los
            # inyectados pone _consecutivos[tipo] a 0 y la primera inferencia
            # válida del simulador ya no dispara. Repuesto a umbral-1 cada
            # 120 ms tapamos la ventana de carrera.
            self._gestor._consecutivos[tipo] = self._gestor._umbral - 1
            # avistamiento_raro: re-sembrar timestamp en cada iteración porque
            # _actualizar_avistamientos lo sobreescribe tras cada frame procesado
            if tipo == "avistamiento_raro":
                self._gestor._ultimo_avistamiento["gorila"] = time.time() - 31 * 60

            _put_dropping(self._capturador.cola, frame)
            _put_dropping(self._capturador.cola_display, frame)

            # Salir en cuanto el gestor haya registrado el disparo del hito
            if self._gestor._ultimo_disparo.get(tipo, 0) > disparo_antes:
                hito_detectado = True
                break

            time.sleep(_INTERVALO_INYECCION_S)

        self._gestor.restaurar_cooldowns()
        self._capturador.pausado = pausado_antes

        # Si el timeout se agotó antes de que YOLO detectara, esperar un poco más:
        # puede haber frames en el pipeline que aún no han llegado al gestor.
        if not hito_detectado:
            t_drenado = time.time()
            while time.time() - t_drenado < 5.0:
                if self._gestor._ultimo_disparo.get(tipo, 0) > disparo_antes:
                    break
                time.sleep(0.1)

        with self._lock:
            self.simulando = None
            self.frame_override = None


def _put_dropping(cola: queue.Queue, item) -> None:
    if cola.full():
        try:
            cola.get_nowait()
        except queue.Empty:
            pass
    cola.put(item)
