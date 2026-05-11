"""Registro de hitos verificados en SQLite."""
import sqlite3
import threading
from pathlib import Path

from loguru import logger

from .verificador import HitoVerificado

_CREAR_TABLA = """
CREATE TABLE IF NOT EXISTS hitos (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    marca_tiempo             REAL    NOT NULL,
    marca_tiempo_deteccion   REAL,
    tipo                     TEXT    NOT NULL,
    descripcion              TEXT    NOT NULL,
    confirmado               INTEGER NOT NULL,
    razonamiento             TEXT    NOT NULL,
    mensaje                  TEXT,
    ruta_frame               TEXT
)
"""


class BaseDatos:

    def __init__(self, ruta: str):
        self._ruta = ruta
        self._lock = threading.Lock()
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        with self._conectar() as con:
            con.execute(_CREAR_TABLA)
            for columna_sql in (
                "ALTER TABLE hitos ADD COLUMN marca_tiempo_deteccion REAL",
                "ALTER TABLE hitos ADD COLUMN acciones TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE hitos ADD COLUMN errores TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    con.execute(columna_sql)
                except Exception:
                    pass  # columna ya existe
        logger.info(f"Base de datos lista: {ruta}")

    def registrar_hito(
        self,
        hito: HitoVerificado,
        ruta_frame: str | None = None,
        acciones: list[str] | None = None,
        errores: list[str] | None = None,
    ) -> None:
        with self._lock:
            with self._conectar() as con:
                con.execute(
                    """INSERT INTO hitos
                       (marca_tiempo, marca_tiempo_deteccion, tipo, descripcion,
                        confirmado, razonamiento, mensaje, ruta_frame, acciones, errores)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        hito.marca_tiempo,
                        hito.marca_tiempo_deteccion,
                        hito.tipo,
                        hito.descripcion,
                        int(hito.confirmado),
                        hito.razonamiento,
                        hito.mensaje,
                        ruta_frame,
                        ",".join(acciones) if acciones else "",
                        ",".join(errores) if errores else "",
                    ),
                )
        logger.debug(f"Hito registrado en BD: {hito.tipo} ({'confirmado' if hito.confirmado else 'falso positivo'})")

    def hitos_recientes(self, limite: int = 50) -> list[dict]:
        with self._lock:
            with self._conectar() as con:
                filas = con.execute(
                    "SELECT * FROM hitos ORDER BY marca_tiempo DESC LIMIT ?", (limite,)
                ).fetchall()
        return [dict(fila) for fila in filas]

    # ------------------------------------------------------------------

    def _conectar(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._ruta)
        con.row_factory = sqlite3.Row
        return con
