"""Extrae beans128.npz del taller en una carpeta temporal de imágenes.

Uso en Colab: python 07_cierre/obtener_datos.py --destino /content/beans128
El destino debe estar vacío. No modifica el repositorio de origen.
"""

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np


URL = ("https://raw.githubusercontent.com/lobaton/AI-Workshop-UNTRM/main/"
       "session2_inside_the_black_box/src/data/beans128.npz")
CLASES = ("angular_leaf_spot", "bean_rust", "healthy")
PARTICIONES = ("train", "validation", "test")


def extraer(destino: Path, origen: Path | None = None):
    if destino.exists() and any(destino.iterdir()):
        raise ValueError(f"El destino debe estar vacío: {destino}")
    if origen is None:
        with urllib.request.urlopen(URL, timeout=60) as respuesta:
            fuente = io.BytesIO(respuesta.read())
    else:
        fuente = origen
    with np.load(fuente) as datos:
        jpeg, offsets = datos["jpeg"], datos["offsets"]
        etiquetas, particiones = datos["labels"], datos["split"]
        if len(etiquetas) != 1295 or len(offsets) != 1296 or len(particiones) != 1295:
            raise ValueError("El archivo no tiene la estructura esperada de beans128")
        destino.mkdir(parents=True, exist_ok=True)
        for i, (etiqueta, particion) in enumerate(zip(etiquetas, particiones)):
            nombre = f"{i + 1:04d}_{PARTICIONES[int(particion)]}_{CLASES[int(etiqueta)]}.jpg"
            (destino / nombre).write_bytes(jpeg[offsets[i]:offsets[i + 1]].tobytes())
    print(f"Se extrajeron 1.295 imágenes en {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, required=True)
    parser.add_argument("--origen", type=Path, help="NPZ local opcional; si se omite, se descarga del taller")
    args = parser.parse_args()
    extraer(args.destino, args.origen)

