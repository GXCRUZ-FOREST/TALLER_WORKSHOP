"""Prepara beans128 para la CNN y, opcionalmente, crea una vista de ejemplo.

Uso: python 02_preparacion/preparar.py --datos "C:\\ruta\\beans128" --vista vista.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


CLASES = ("angular_leaf_spot", "bean_rust", "healthy")
PARTICIONES = ("train", "validation", "test")
PATRON = re.compile(r"^\d+_(train|validation|test)_(angular_leaf_spot|bean_rust|healthy)\.jpe?g$", re.I)


def buscar_imagenes(carpeta: Path) -> dict:
    """Devuelve rutas y etiquetas según la partición original del archivo."""
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta: {carpeta}")
    grupos = {parte: [] for parte in PARTICIONES}
    for ruta in sorted(carpeta.iterdir()):
        if not ruta.is_file():
            continue
        resultado = PATRON.fullmatch(ruta.name)
        if resultado is None:
            raise ValueError(f"Nombre inesperado: {ruta.name}")
        parte, clase = (valor.lower() for valor in resultado.groups())
        grupos[parte].append((ruta, CLASES.index(clase)))
    return grupos


def convertir(imagen: Image.Image) -> torch.Tensor:
    """RGB 128×128, enteros 0..255 -> tensor (3,128,128), decimales -1..1."""
    imagen = imagen.convert("RGB")
    if imagen.size != (128, 128):
        raise ValueError(f"Se esperaba 128×128, se recibió {imagen.size}")
    pixeles = np.asarray(imagen, dtype=np.float32).copy()
    return torch.from_numpy(pixeles).permute(2, 0, 1) / 127.5 - 1.0


def aumentar(tensor: torch.Tensor) -> torch.Tensor:
    """Volteos y rotación en múltiplos de 90 grados; conserva la etiqueta."""
    if torch.rand(()) < 0.5:
        tensor = tensor.flip(2)  # horizontal
    if torch.rand(()) < 0.5:
        tensor = tensor.flip(1)  # vertical
    vueltas = int(torch.randint(0, 4, ()).item())
    return torch.rot90(tensor, vueltas, dims=(1, 2))


class Hojas(Dataset):
    def __init__(self, muestras, entrenamiento=False):
        self.muestras = muestras
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, indice):
        ruta, etiqueta = self.muestras[indice]
        with Image.open(ruta) as imagen:
            tensor = convertir(imagen)
        if self.entrenamiento:
            tensor = aumentar(tensor)
        return tensor, etiqueta


def crear_datasets(carpeta: Path) -> dict:
    grupos = buscar_imagenes(carpeta)
    return {parte: Hojas(muestras, entrenamiento=(parte == "train"))
            for parte, muestras in grupos.items()}


def volver_a_imagen(tensor: torch.Tensor) -> Image.Image:
    pixeles = ((tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255)
    return Image.fromarray(pixeles.astype(np.uint8), "RGB")


def guardar_vista(grupos: dict, destino: Path) -> None:
    """Una foto original y tres transformaciones por clase, solo de train."""
    torch.manual_seed(7)
    lienzo = Image.new("RGB", (4 * 128, 3 * 152), "white")
    dibujo = ImageDraw.Draw(lienzo)
    for fila, clase in enumerate(CLASES):
        ruta, _ = next((ruta, etiqueta) for ruta, etiqueta in grupos["train"]
                       if etiqueta == fila)
        with Image.open(ruta) as imagen:
            original = convertir(imagen)
        for columna in range(4):
            tensor = original if columna == 0 else aumentar(original)
            lienzo.paste(volver_a_imagen(tensor), (columna * 128, fila * 152 + 24))
            titulo = clase if columna == 0 else f"giro/volteo {columna}"
            dibujo.text((columna * 128 + 3, fila * 152 + 4), titulo, fill="black")
    destino.parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, required=True)
    parser.add_argument("--vista", type=Path, help="PNG opcional con ejemplos")
    args = parser.parse_args()
    grupos = buscar_imagenes(args.datos)
    datasets = crear_datasets(args.datos)
    for parte in PARTICIONES:
        imagen, etiqueta = datasets[parte][0]
        print(f"{parte}: {len(datasets[parte])} fotos; primera: "
              f"forma={tuple(imagen.shape)}, clase={CLASES[etiqueta]}, "
              f"mínimo={imagen.min().item():.3f}, máximo={imagen.max().item():.3f}")
    if args.vista:
        guardar_vista(grupos, args.vista)
        print(f"Vista guardada en: {args.vista}")

