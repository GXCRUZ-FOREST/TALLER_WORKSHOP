"""Clasifica una foto y muestra filtros y mapas internos de la CNN.

Uso: python 06_interpretacion/interpretar.py --imagen "C:\\ruta\\hoja.jpg"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "02_preparacion"))
sys.path.insert(0, str(RAIZ / "03_modelo"))
from preparar import CLASES, convertir  # noqa: E402
from modelo import BeansCNN  # noqa: E402


def cargar_modelo(ruta):
    paquete = torch.load(ruta, map_location="cpu", weights_only=True)
    if tuple(paquete["clases"]) != CLASES:
        raise ValueError("El orden de clases del modelo no coincide")
    modelo = BeansCNN(filtros=paquete["filtros"])
    modelo.load_state_dict(paquete["pesos"])
    modelo.eval()
    return modelo, paquete


def preparar_foto(ruta):
    with Image.open(ruta) as original:
        foto = ImageOps.fit(original.convert("RGB"), (128, 128),
                            method=Image.Resampling.LANCZOS)
    return foto, convertir(foto).unsqueeze(0)


def guardar_filtros(modelo, destino):
    """Representación RGB relativa de los 16 filtros de la primera convolución."""
    pesos = modelo.bloques[0][0].weight.detach().cpu().numpy()
    lienzo = Image.new("RGB", (4 * 90, 4 * 110), "white")
    dibujo = ImageDraw.Draw(lienzo)
    for i, filtro in enumerate(pesos):
        matriz = np.transpose(filtro, (1, 2, 0))
        minimo, maximo = matriz.min(), matriz.max()
        colores = ((matriz - minimo) / max(maximo - minimo, 1e-8) * 255).astype(np.uint8)
        imagen = Image.fromarray(colores, "RGB").resize((80, 80), Image.Resampling.NEAREST)
        x, y = (i % 4) * 90 + 5, (i // 4) * 110 + 24
        lienzo.paste(imagen, (x, y))
        dibujo.text((x, y - 18), f"Filtro {i + 1}", fill="black")
    lienzo.save(destino)


def guardar_mapas(modelo, tensor, foto, destino):
    """Mapas más activos por capa para esta foto, tras ReLU y antes de pooling."""
    with torch.inference_mode():
        mapas = modelo.mapas_caracteristicas(tensor)
    lienzo = Image.new("RGB", (6 * 132, 4 * 150), "white")
    dibujo = ImageDraw.Draw(lienzo)
    lienzo.paste(foto, (2, 22))
    dibujo.text((2, 3), "Foto de entrada", fill="black")
    for capa, conjunto in enumerate(mapas, start=1):
        activaciones = conjunto[0].cpu()
        indices = torch.topk(activaciones.mean(dim=(1, 2)), k=6).indices.tolist()
        for columna, indice in enumerate(indices):
            matriz = activaciones[indice].numpy()
            minimo, maximo = float(matriz.min()), float(matriz.max())
            valores = ((matriz - minimo) / max(maximo - minimo, 1e-8) * 255).astype(np.uint8)
            imagen = Image.fromarray(valores, "L").convert("RGB").resize((128, 128), Image.Resampling.NEAREST)
            x, y = columna * 132 + 2, capa * 150 + 22
            lienzo.paste(imagen, (x, y))
            dibujo.text((x, y - 18), f"Capa {capa}, mapa {indice + 1}", fill="black")
    lienzo.save(destino)


def interpretar(imagen, modelo_archivo, salida):
    modelo, paquete = cargar_modelo(modelo_archivo)
    foto, tensor = preparar_foto(imagen)
    with torch.inference_mode():
        probabilidades = torch.softmax(modelo(tensor), dim=1)[0].tolist()
    salida.mkdir(parents=True, exist_ok=True)
    guardar_filtros(modelo, salida / "filtros.png")
    guardar_mapas(modelo, tensor, foto, salida / "mapas.png")
    resultado = {
        "imagen": imagen.name,
        "modelo": modelo_archivo.name,
        "epoca_modelo": paquete["epoca"],
        "preparacion": "RGB, recorte central cuadrado, 128x128, escala [-1,1]",
        "probabilidades": {clase: round(valor, 4) for clase, valor in zip(CLASES, probabilidades)},
        "clase_predicha": CLASES[int(np.argmax(probabilidades))],
        "nota": "La probabilidad estimada no demuestra certeza ni revela por sí sola qué zona causó la decisión.",
    }
    (salida / "prediccion.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", type=Path, required=True)
    parser.add_argument("--modelo", type=Path, default=RAIZ / "04_entrenamiento" / "resultados" / "sin_aumento.pt")
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados")
    args = parser.parse_args()
    interpretar(args.imagen, args.modelo, args.salida)

