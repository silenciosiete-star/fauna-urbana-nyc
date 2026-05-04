"""Fine-tuning de YOLO26 con el dataset preparado.

Lee la configuración desde config/entrenamiento.yaml y lanza model.train().
Al terminar, copia el mejor modelo a modelos/fauna_urbana.pt y evalúa por clase.

Uso:
    python entrenamiento/entrenar.py
    python entrenamiento/entrenar.py --config config/entrenamiento.yaml
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


PARAMETROS_TRAIN = {
    "epochs", "patience", "batch", "imgsz", "workers",
    "fl_gamma",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
}


def cargar_config(ruta: Path) -> dict:
    if not ruta.exists():
        sys.exit(f"No se encontró {ruta}")
    with open(ruta) as f:
        return yaml.safe_load(f)


def verificar_dataset(ruta_data_yaml: Path) -> None:
    if not ruta_data_yaml.exists():
        sys.exit(
            f"No se encontró {ruta_data_yaml}.\n"
            "Ejecuta primero: python entrenamiento/preparar_dataset.py"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning de YOLO26 para Fauna Urbana NYC")
    parser.add_argument("--config", default="config/entrenamiento.yaml", help="Fichero de configuración")
    args = parser.parse_args()

    config = cargar_config(Path(args.config))

    ruta_datos    = Path(config["datos"])
    ruta_salida   = Path(config["salida"])
    nombre_exp    = config["nombre_experimento"]
    modelo_base   = config["modelo_base"]
    ruta_destino  = ruta_salida / "fauna_urbana.pt"

    verificar_dataset(ruta_datos)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    print(f"Modelo base       : {modelo_base}")
    print(f"Dataset           : {ruta_datos}")
    print(f"Épocas            : {config.get('epocas', 100)}  |  batch: {config.get('batch', 16)}  |  imgsz: {config.get('imgsz', 640)}")
    print(f"Salida final      : {ruta_destino}")
    print()

    modelo = YOLO(modelo_base)

    parametros = {k: config[k] for k in PARAMETROS_TRAIN if k in config}
    parametros["epochs"] = config.get("epocas", parametros.pop("epochs", 100))

    resultados = modelo.train(
        data=str(ruta_datos),
        project=str(ruta_salida),
        name=nombre_exp,
        **parametros,
    )

    # El mejor modelo queda en <salida>/<nombre_exp>/weights/best.pt
    mejor_pt = ruta_salida / nombre_exp / "weights" / "best.pt"
    if mejor_pt.exists():
        shutil.copy2(mejor_pt, ruta_destino)
        print(f"\nModelo copiado a: {ruta_destino}")
    else:
        print(f"\nAVISO: no se encontró best.pt en {mejor_pt.parent}")

    # Evaluación por clase en el set de test
    print("\nEvaluando en test...")
    modelo_final = YOLO(str(ruta_destino) if ruta_destino.exists() else str(mejor_pt))
    metricas = modelo_final.val(data=str(ruta_datos), split="test")

    print("\nmAP por clase (test):")
    with open(ruta_datos) as f:
        nombres_clase = yaml.safe_load(f)["names"]
    for i, nombre in enumerate(nombres_clase):
        try:
            mAP = metricas.box.maps[i]
            aviso = " ← revisar, muy bajo" if mAP < 0.3 else ""
            print(f"  {nombre:<20} mAP50: {mAP:.3f}{aviso}")
        except (IndexError, AttributeError):
            pass

    print(f"\nmAP50 global: {metricas.box.map50:.3f}")
    print("\nRecuerda actualizar gemma.clases en config/config.yaml con las nuevas clases.")


if __name__ == "__main__":
    main()
