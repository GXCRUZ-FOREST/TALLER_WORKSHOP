"""Entrena la CNN del taller y compara el uso de aumento de datos.

Uso: python 04_entrenamiento/entrenar.py --datos "C:\\ruta\\beans128"
La selección se hace solo con validación; este programa no carga imágenes de test.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "02_preparacion"))
sys.path.insert(0, str(RAIZ / "03_modelo"))
from preparar import Hojas, buscar_imagenes  # noqa: E402
from modelo import BeansCNN, numero_parametros  # noqa: E402


def fijar_semilla(valor):
    random.seed(valor)
    np.random.seed(valor)
    torch.manual_seed(valor)


def evaluar(modelo, cargador, dispositivo):
    modelo.eval()
    total, aciertos, perdida_total = 0, 0, 0.0
    with torch.inference_mode():
        for imagenes, etiquetas in cargador:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            salidas = modelo(imagenes)
            perdida_total += nn.functional.cross_entropy(salidas, etiquetas).item() * len(etiquetas)
            aciertos += (salidas.argmax(1) == etiquetas).sum().item()
            total += len(etiquetas)
    return {"perdida": perdida_total / total, "exactitud": aciertos / total}


def una_ejecucion(grupos, usar_aumento, args, dispositivo):
    fijar_semilla(args.semilla)
    generador = torch.Generator().manual_seed(args.semilla)
    entrenamiento = DataLoader(Hojas(grupos["train"], entrenamiento=usar_aumento),
                               batch_size=args.lote, shuffle=True, generator=generador, num_workers=0)
    validacion = DataLoader(Hojas(grupos["validation"], entrenamiento=False),
                            batch_size=args.lote, shuffle=False, num_workers=0)
    modelo = BeansCNN(filtros=args.filtros).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.tasa)
    historial = []
    mejor_perdida = float("inf")
    mejor_epoca = 0
    nombre = "con_aumento" if usar_aumento else "sin_aumento"
    ruta_pesos = args.salida / f"{nombre}.pt"

    for epoca in range(1, args.epocas + 1):
        modelo.train()
        total, aciertos, perdida_total = 0, 0, 0.0
        for imagenes, etiquetas in entrenamiento:
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            salidas = modelo(imagenes)
            perdida = nn.functional.cross_entropy(salidas, etiquetas)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(etiquetas)
            aciertos += (salidas.argmax(1) == etiquetas).sum().item()
            total += len(etiquetas)
        metrica_val = evaluar(modelo, validacion, dispositivo)
        fila = {
            "epoca": epoca,
            "entrenamiento_perdida": round(perdida_total / total, 5),
            "entrenamiento_exactitud": round(aciertos / total, 5),
            "validacion_perdida": round(metrica_val["perdida"], 5),
            "validacion_exactitud": round(metrica_val["exactitud"], 5),
        }
        historial.append(fila)
        print(f"{nombre} {epoca:02d}/{args.epocas}: entrenamiento "
              f"{fila['entrenamiento_exactitud']:.1%}, validación "
              f"{fila['validacion_exactitud']:.1%}, pérdida val. "
              f"{fila['validacion_perdida']:.4f}", flush=True)
        if metrica_val["perdida"] < mejor_perdida:
            mejor_perdida = metrica_val["perdida"]
            mejor_epoca = epoca
            torch.save({"pesos": modelo.state_dict(), "filtros": args.filtros,
                        "clases": ("angular_leaf_spot", "bean_rust", "healthy"),
                        "epoca": epoca, "aumento": usar_aumento}, ruta_pesos)

    return {"nombre": nombre, "mejor_epoca": mejor_epoca,
            "mejor_perdida_validacion": round(mejor_perdida, 5),
            "mejor_exactitud_validacion": historial[mejor_epoca - 1]["validacion_exactitud"],
            "historial": historial, "archivo_pesos": ruta_pesos.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados")
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--tasa", type=float, default=0.003)
    parser.add_argument("--filtros", type=int, default=16)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    if args.epocas < 1 or args.lote < 1 or args.tasa <= 0:
        parser.error("Épocas, lote y tasa deben ser positivos")
    grupos = buscar_imagenes(args.datos)
    if len(grupos["train"]) != 1034 or len(grupos["validation"]) != 133:
        parser.error("Las particiones no coinciden con beans128 del taller")
    args.salida.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}; parámetros: {numero_parametros(BeansCNN(args.filtros)):,}", flush=True)
    ejecuciones = [una_ejecucion(grupos, aumento, args, dispositivo) for aumento in (False, True)]
    elegida = min(ejecuciones, key=lambda e: e["mejor_perdida_validacion"])
    resumen = {
        "configuracion": {"epocas": args.epocas, "lote": args.lote, "tasa": args.tasa,
                          "filtros": args.filtros, "semilla": args.semilla,
                          "criterio_seleccion": "menor pérdida de validación"},
        "ejecuciones": ejecuciones,
        "modelo_elegido": elegida["nombre"],
        "archivo_modelo_elegido": elegida["archivo_pesos"],
        "nota": "Las imágenes de prueba no se cargaron durante el entrenamiento ni la selección.",
    }
    (args.salida / "historial.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"Elegido por validación: {elegida['nombre']} (época {elegida['mejor_epoca']}); "
          f"pérdida {elegida['mejor_perdida_validacion']:.4f}; "
          f"exactitud {elegida['mejor_exactitud_validacion']:.1%}", flush=True)


if __name__ == "__main__":
    main()

