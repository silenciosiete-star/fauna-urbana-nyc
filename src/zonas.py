"""Carga zonas desde config.yaml y comprueba si una detección pertenece a una zona."""
from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass
class Zona:
    nombre: str
    clases_detectables: list[str]
    poligono: sv.PolygonZone


def cargar_zonas(config_zonas: list[dict]) -> dict[str, Zona]:
    """Devuelve un dict {nombre_zona: Zona} a partir de la sección 'zonas' del config."""
    zonas: dict[str, Zona] = {}
    for entrada in config_zonas:
        puntos = np.array(entrada["puntos"], dtype=np.int32)
        zonas[entrada["nombre"]] = Zona(
            nombre=entrada["nombre"],
            clases_detectables=entrada["deteccion"],
            poligono=sv.PolygonZone(polygon=puntos),
        )
    return zonas


def detecciones_en_zona(detecciones: sv.Detections, zona: Zona) -> sv.Detections:
    """Filtra y devuelve solo las detecciones cuyo centro cae dentro de la zona."""
    if len(detecciones) == 0:
        return detecciones
    mascara = zona.poligono.trigger(detecciones)
    return detecciones[mascara]
