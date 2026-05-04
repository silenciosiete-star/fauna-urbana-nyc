"""Prepara el dataset exportado de Roboflow para el entrenamiento de YOLO26.

Divide las imágenes en train/valid/test con split estratificado por clase dominante,
manteniendo la distribución de clases en cada partición.

Uso:
    python entrenamiento/preparar_dataset.py
    python entrenamiento/preparar_dataset.py --origen datos/dataset --destino datos/preparado
    python entrenamiento/preparar_dataset.py --meteo               # añade augmentación meteorológica
    python entrenamiento/preparar_dataset.py --meteo --factor-meteo 1.0  # una copia por imagen

Salida:
    <destino>/train/images|labels/
    <destino>/valid/images|labels/
    <destino>/test/images|labels/
    <destino>/data.yaml
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

RATIO_TRAIN = 0.70
RATIO_VALID = 0.10
# ratio_test = 1 - RATIO_TRAIN - RATIO_VALID
SEMILLA = 42
SPLITS = ["train", "valid", "test"]

# Imágenes sin anotaciones: van todas a train (negativos de fondo útiles en entrenamiento)
ESTRATO_FONDO = "__fondo__"


def cargar_configuracion(ruta_origen: Path) -> dict:
    ruta = ruta_origen / "data.yaml"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró data.yaml en {ruta_origen}")
    with open(ruta) as f:
        return yaml.safe_load(f)


def clase_dominante(ruta_label: Path) -> str | None:
    """Devuelve el id de clase más frecuente en la imagen, o None si está vacía."""
    texto = ruta_label.read_text().strip()
    if not texto:
        return None
    conteo: dict[str, int] = defaultdict(int)
    for linea in texto.splitlines():
        partes = linea.strip().split()
        if partes:
            conteo[partes[0]] += 1
    return max(conteo, key=conteo.__getitem__)


def agrupar_por_estrato(ruta_images: Path, ruta_labels: Path) -> dict[str, list[str]]:
    """Agrupa nombres de imagen por su clase dominante."""
    grupos: dict[str, list[str]] = defaultdict(list)
    for ruta_img in sorted(ruta_images.glob("*.jpg")):
        ruta_lbl = ruta_labels / ruta_img.with_suffix(".txt").name
        if not ruta_lbl.exists():
            grupos[ESTRATO_FONDO].append(ruta_img.stem)
            continue
        estrato = clase_dominante(ruta_lbl)
        grupos[ESTRATO_FONDO if estrato is None else estrato].append(ruta_img.stem)
    return grupos


def dividir_estrato(nombres: list[str], rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    """Divide una lista en train/valid/test respetando los ratios globales."""
    mezclados = nombres[:]
    rng.shuffle(mezclados)
    n = len(mezclados)
    n_valid = max(1, round(n * RATIO_VALID)) if n >= 5 else 0
    n_test  = max(1, round(n * (1 - RATIO_TRAIN - RATIO_VALID))) if n >= 5 else 0
    n_train = n - n_valid - n_test
    return mezclados[:n_train], mezclados[n_train:n_train + n_valid], mezclados[n_train + n_valid:]


def copiar_archivos(nombres: list[str], ruta_src_images: Path, ruta_src_labels: Path,
                    ruta_dst_images: Path, ruta_dst_labels: Path) -> None:
    for nombre in nombres:
        src_img = ruta_src_images / f"{nombre}.jpg"
        src_lbl = ruta_src_labels / f"{nombre}.txt"
        if src_img.exists():
            shutil.copy2(src_img, ruta_dst_images / src_img.name)
        if src_lbl.exists():
            shutil.copy2(src_lbl, ruta_dst_labels / src_lbl.name)


def escribir_data_yaml(ruta_destino: Path, nombres_clase: list[str]) -> None:
    contenido = {
        "train": str(ruta_destino / "train" / "images"),
        "val":   str(ruta_destino / "valid" / "images"),
        "test":  str(ruta_destino / "test"  / "images"),
        "nc":    len(nombres_clase),
        "names": nombres_clase,
    }
    with open(ruta_destino / "data.yaml", "w") as f:
        yaml.dump(contenido, f, allow_unicode=True, sort_keys=False)


def _pipeline_meteo():
    """Pipeline de albumentations con efectos meteorológicos de Times Square."""
    try:
        import albumentations as A
    except ImportError:
        return None

    return A.Compose([
        A.OneOf([
            A.RandomRain(
                slant_range=(-15, 15), drop_length=18, drop_width=1,
                drop_color=(180, 180, 180), blur_value=2, p=1.0,
            ),
            A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.08, p=1.0),
            A.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 0.4), src_radius=120,
                src_color=(255, 240, 180), num_flare_circles_range=(2, 5), p=1.0,
            ),
            # Calima: niebla ligera + tinte cálido anaranjado
            A.Compose([
                A.RandomFog(fog_coef_range=(0.04, 0.14), alpha_coef=0.05, p=1.0),
                A.RGBShift(r_shift_limit=(15, 30), g_shift_limit=(-5, 8), b_shift_limit=(-25, -10), p=1.0),
            ]),
        ], p=1.0),
    ])


def augmentar_meteorologicamente(
    ruta_images: Path,
    ruta_labels: Path,
    factor: float,
    rng: random.Random,
) -> int:
    """Genera copias con efectos meteorológicos de las imágenes de train.

    Las etiquetas no cambian (los efectos no mueven los bboxes).
    Devuelve el número de imágenes generadas.
    """
    pipeline = _pipeline_meteo()
    if pipeline is None:
        print("AVISO: albumentations no instalado — augmentación meteorológica omitida.")
        print("       Instálalo con: pip install albumentations")
        return 0

    imagenes = sorted(ruta_images.glob("*.jpg"))
    # Cuántas copias por imagen: factor=0.5 → aprox. una de cada dos imágenes recibe una copia
    n_copias_total = max(1, round(len(imagenes) * factor))
    seleccionadas = rng.choices(imagenes, k=n_copias_total)

    conteo_efectos: dict[str, int] = {"lluvia": 0, "niebla": 0, "destello": 0, "calima": 0}
    generadas = 0

    for idx, ruta_img in enumerate(seleccionadas):
        imagen_bgr = cv2.imread(str(ruta_img))
        if imagen_bgr is None:
            continue
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

        resultado = pipeline(image=imagen_rgb)
        imagen_aug = cv2.cvtColor(resultado["image"], cv2.COLOR_RGB2BGR)

        nombre_aug = f"{ruta_img.stem}_meteo_{idx}.jpg"
        cv2.imwrite(str(ruta_images / nombre_aug), imagen_aug)

        ruta_lbl = ruta_labels / ruta_img.with_suffix(".txt").name
        if ruta_lbl.exists():
            shutil.copy2(ruta_lbl, ruta_labels / f"{ruta_img.stem}_meteo_{idx}.txt")

        generadas += 1

    print(f"  Augmentación meteorológica: {generadas} imágenes generadas sobre {len(imagenes)} originales")
    return generadas


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el dataset Roboflow para YOLO26")
    parser.add_argument("--origen",       default="datos/dataset",   help="Dataset exportado de Roboflow")
    parser.add_argument("--destino",      default="datos/preparado", help="Carpeta de salida con los splits")
    parser.add_argument("--meteo",        action="store_true",        help="Añade augmentación meteorológica al train")
    parser.add_argument("--factor-meteo", type=float, default=0.5,   help="Copias meteorológicas por imagen de train (default: 0.5)")
    args = parser.parse_args()

    ruta_origen  = Path(args.origen)
    ruta_destino = Path(args.destino)

    config = cargar_configuracion(ruta_origen)
    nombres_clase: list[str] = config["names"]

    # Roboflow pone todas las imágenes en train/ si no se configura split en la plataforma
    ruta_src_images = ruta_origen / "train" / "images"
    ruta_src_labels = ruta_origen / "train" / "labels"

    if not ruta_src_images.exists():
        raise FileNotFoundError(f"No se encontró {ruta_src_images}")

    total_imagenes = len(list(ruta_src_images.glob("*.jpg")))
    print(f"Origen  : {ruta_origen}  ({total_imagenes} imágenes, {len(nombres_clase)} clases)")
    print(f"Destino : {ruta_destino}")
    print(f"Splits  : {int(RATIO_TRAIN*100)}% train / {int(RATIO_VALID*100)}% valid / {int((1-RATIO_TRAIN-RATIO_VALID)*100)}% test")
    print()

    if ruta_destino.exists():
        print(f"AVISO: {ruta_destino} ya existe — se sobreescribirá.")
        shutil.rmtree(ruta_destino)

    for split in SPLITS:
        (ruta_destino / split / "images").mkdir(parents=True)
        (ruta_destino / split / "labels").mkdir(parents=True)

    grupos = agrupar_por_estrato(ruta_src_images, ruta_src_labels)
    rng = random.Random(SEMILLA)

    conteo: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SPLITS}
    asignaciones: dict[str, list[str]] = {s: [] for s in SPLITS}

    for estrato, nombres in sorted(grupos.items()):
        if estrato == ESTRATO_FONDO:
            # Los negativos de fondo van todos a train
            asignaciones["train"].extend(nombres)
            conteo["train"][ESTRATO_FONDO] += len(nombres)
            continue

        train, valid, test = dividir_estrato(nombres, rng)
        asignaciones["train"].extend(train)
        asignaciones["valid"].extend(valid)
        asignaciones["test"].extend(test)
        conteo["train"][estrato] += len(train)
        conteo["valid"][estrato] += len(valid)
        conteo["test"][estrato]  += len(test)

    for split in SPLITS:
        copiar_archivos(
            asignaciones[split],
            ruta_src_images, ruta_src_labels,
            ruta_destino / split / "images",
            ruta_destino / split / "labels",
        )

    escribir_data_yaml(ruta_destino, nombres_clase)

    if args.meteo:
        print("Generando augmentación meteorológica sobre train...")
        augmentar_meteorologicamente(
            ruta_destino / "train" / "images",
            ruta_destino / "train" / "labels",
            args.factor_meteo,
            rng,
        )

    # Informe
    print("Distribución por split y clase:")
    ancho = max(len(n) for n in nombres_clase + [ESTRATO_FONDO])
    cabecera = f"  {'Clase':<{ancho}}  {'train':>6}  {'valid':>6}  {'test':>6}  {'total':>6}"
    print(cabecera)
    print("  " + "-" * (len(cabecera) - 2))

    todos_estratos = sorted(
        set(conteo["train"]) | set(conteo["valid"]) | set(conteo["test"]),
        key=lambda x: (x == ESTRATO_FONDO, x),
    )
    for estrato in todos_estratos:
        t = conteo["train"].get(estrato, 0)
        v = conteo["valid"].get(estrato, 0)
        e = conteo["test"].get(estrato, 0)
        nombre_mostrar = "(fondo)" if estrato == ESTRATO_FONDO else nombres_clase[int(estrato)]
        print(f"  {nombre_mostrar:<{ancho}}  {t:>6}  {v:>6}  {e:>6}  {t+v+e:>6}")

    totales = {s: len(asignaciones[s]) for s in SPLITS}
    print("  " + "-" * (len(cabecera) - 2))
    print(f"  {'TOTAL':<{ancho}}  {totales['train']:>6}  {totales['valid']:>6}  {totales['test']:>6}  {sum(totales.values()):>6}")
    print()
    print(f"data.yaml escrito en: {ruta_destino / 'data.yaml'}")
    print()
    print("AVISO: config/config.yaml tiene 5 clases en gemma.clases. Actualízalo")
    print(f"       tras el entrenamiento para incluir las {len(nombres_clase)} clases del modelo.")


if __name__ == "__main__":
    main()
