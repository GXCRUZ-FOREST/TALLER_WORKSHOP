"""Comprueba las imágenes locales beans128 antes de entrenar la CNN.

Uso: python explorar.py --datos "C:\\ruta\\beans128"
Requiere Pillow: python -m pip install pillow
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PATRON = re.compile(r"^\d+_(train|validation|test)_(angular_leaf_spot|bean_rust|healthy)\.jpe?g$", re.I)


def revisar(carpeta: Path) -> dict:
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta: {carpeta}")

    archivos = sorted(ruta for ruta in carpeta.iterdir() if ruta.is_file())
    conteos = defaultdict(Counter)
    tamanos = Counter()
    modos = Counter()
    huellas = defaultdict(list)
    errores = []

    for ruta in archivos:
        coincidencia = PATRON.fullmatch(ruta.name)
        if not coincidencia:
            errores.append(f"Nombre inesperado: {ruta.name}")
            continue
        particion, clase = (valor.lower() for valor in coincidencia.groups())
        conteos[particion][clase] += 1
        try:
            with Image.open(ruta) as imagen:
                imagen.verify()
            with Image.open(ruta) as imagen:
                tamanos[f"{imagen.width}x{imagen.height}"] += 1
                modos[imagen.mode] += 1
        except (OSError, UnidentifiedImageError) as exc:
            errores.append(f"No se puede abrir {ruta.name}: {exc}")
        huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
        huellas[huella].append((ruta.name, particion))

    duplicados = [nombres for nombres in huellas.values() if len(nombres) > 1]
    cruces = [nombres for nombres in duplicados if len({parte for _, parte in nombres}) > 1]
    return {
        "total_archivos": len(archivos),
        "por_particion_y_clase": {parte: dict(conteos[parte]) for parte in ("train", "validation", "test")},
        "tamanos": dict(tamanos),
        "modos_de_color": dict(modos),
        "errores": errores,
        "grupos_de_duplicados_exactos": len(duplicados),
        "duplicados_entre_particiones": len(cruces),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, required=True)
    argumentos = parser.parse_args()
    print(json.dumps(revisar(argumentos.datos), indent=2, ensure_ascii=False))

