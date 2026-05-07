"""Analiza capturas existentes y elige las mejores imágenes para cada simulación.

Uso:
    python tools/analizar_simulaciones.py [--copiar]

    --copiar  Copia las mejores candidatas a assets/simulaciones/ (sobreescribe).

Para cada hito imprime las capturas ordenadas por "pureza" (dispara solo ese hito),
y muestra qué otros hitos también dispararía cada candidata.
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.zonas import cargar_zonas, detecciones_en_zona
from src.eventos import _UNIVERSOS


def cargar_config() -> dict:
    ruta = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def inferir(modelo: YOLO, imagen: np.ndarray, umbral: float) -> sv.Detections:
    resultado = modelo.predict(imagen, conf=umbral, verbose=False, imgsz=1280)[0]
    if len(resultado.boxes) == 0:
        return sv.Detections.empty()
    xyxy = resultado.boxes.xyxy.cpu().numpy()
    confianza = resultado.boxes.conf.cpu().numpy()
    cls_ids = resultado.boxes.cls.cpu().numpy().astype(int)
    nombres = np.array([modelo.names[c] for c in cls_ids])
    det = sv.Detections(xyxy=xyxy, confidence=confianza, class_id=cls_ids)
    det.data["class_name"] = nombres
    return det


def evaluar_hitos(det: sv.Detections, zonas: dict, cfg: dict) -> set[str]:
    """Devuelve el conjunto de hitos que dispararía este frame (sin contar cooldowns)."""
    nombres = list(det.data.get("class_name", np.array([])))
    clases = set(nombres)
    total = len(det)
    hitos = set()

    universos = {_UNIVERSOS[c] for c in clases if c in _UNIVERSOS}

    # crossover
    c = cfg.get("crossover", {})
    if c.get("activo") and len(universos) >= c["universos_minimos"]:
        hitos.add("crossover")

    # hora_punta
    c = cfg.get("hora_punta", {})
    if c.get("activo") and total >= c["personajes_minimos"]:
        hitos.add("hora_punta")

    # marvel_vs_dc
    c = cfg.get("marvel_vs_dc", {})
    if c.get("activo") and c["personaje_marvel"] in clases and c["personaje_dc"] in clases:
        hitos.add("marvel_vs_dc")

    # conflicto_identidad: mismo tipo en misma zona
    c = cfg.get("conflicto_identidad", {})
    if c.get("activo") and total >= 2:
        for zona in zonas.values():
            en_zona = detecciones_en_zona(det, zona)
            if len(en_zona) >= 2:
                nombres_zona = list(en_zona.data.get("class_name", []))
                if len(nombres_zona) != len(set(nombres_zona)):
                    hitos.add("conflicto_identidad")
                    break

    return hitos


def resumen_clases(det: sv.Detections) -> str:
    nombres = list(det.data.get("class_name", np.array([])))
    conteo: dict[str, int] = defaultdict(int)
    for n in nombres:
        conteo[n] += 1
    partes = [f"{n}×{c}" if c > 1 else n for n, c in sorted(conteo.items())]
    return ", ".join(partes) if partes else "(vacío)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza capturas para simulaciones")
    parser.add_argument("--copiar", action="store_true", help="Copia las mejores a assets/simulaciones/")
    args = parser.parse_args()

    config = cargar_config()
    ruta_modelo = Path(__file__).parent.parent / config["modelo"]["ruta"]
    umbral = config["modelo"]["confianza_minima"]
    zonas = cargar_zonas(config["zonas"])
    cfg_hitos = config["hitos"]

    if not ruta_modelo.exists():
        print(f"[ERROR] Modelo no encontrado en {ruta_modelo}")
        sys.exit(1)

    print(f"Cargando modelo {ruta_modelo.name}...")
    modelo = YOLO(str(ruta_modelo))

    capturas_dir = Path(__file__).parent.parent / "capturas"
    sims_dir = Path(__file__).parent.parent / "assets" / "simulaciones"

    hitos_objetivo = [
        "crossover",
        "conflicto_identidad",
        "hora_punta",
        "marvel_vs_dc",
    ]

    mejores: dict[str, tuple[Path, set]] = {}

    for hito in hitos_objetivo:
        patron = "avengers_assemble_*.jpg" if hito == "crossover" else f"{hito}_*.jpg"
        archivos = sorted(capturas_dir.glob(patron))
        if not archivos:
            print(f"\n[{hito}] Sin capturas disponibles")
            continue

        print(f"\n{'='*60}")
        print(f"HITO: {hito}  ({len(archivos)} capturas)")
        print(f"{'='*60}")

        candidatos: list[tuple[int, Path, set, str]] = []

        for archivo in archivos:
            imagen = cv2.imread(str(archivo))
            if imagen is None:
                continue
            det = inferir(modelo, imagen, umbral)
            disparados = evaluar_hitos(det, zonas, cfg_hitos)
            extra = disparados - {hito}
            pureza = 0 if extra else 1
            clases_str = resumen_clases(det)
            candidatos.append((pureza, archivo, extra, clases_str))

        # Ordenar: primero los limpios, luego por menos hitos extra
        candidatos.sort(key=lambda x: (-x[0], len(x[2])))

        for pureza, archivo, extra, clases_str in candidatos:
            marca = "✓ LIMPIO" if pureza else f"✗ +{sorted(extra)}"
            print(f"  {archivo.name:<55} {marca}")
            print(f"    → {clases_str}")

        # Mejor candidato: el primero de la lista (más limpio)
        mejor_pureza, mejor_archivo, mejor_extra, _ = candidatos[0]
        mejores[hito] = (mejor_archivo, mejor_extra)

    print(f"\n{'='*60}")
    print("RESUMEN — Mejores candidatos:")
    print(f"{'='*60}")
    for hito, (archivo, extra) in mejores.items():
        estado = "LIMPIO" if not extra else f"también dispara {sorted(extra)}"
        print(f"  {hito:<30} {archivo.name}  [{estado}]")

    if args.copiar:
        print(f"\nCopiando a {sims_dir}/...")
        for hito, (archivo, _) in mejores.items():
            destino = sims_dir / f"{hito}.jpg"
            shutil.copy2(archivo, destino)
            print(f"  {hito}.jpg  ←  {archivo.name}")
        print("Hecho.")
    else:
        print(f"\nEjecuta con --copiar para sustituir las imágenes de simulación.")


if __name__ == "__main__":
    main()
