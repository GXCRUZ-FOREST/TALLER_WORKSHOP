"""Evalúa el modelo elegido sobre el grupo de prueba de beans128.

Uso: python 05_evaluacion/evaluar.py --datos "C:\\ruta\\beans128"
No entrena ni modifica los pesos del modelo.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "02_preparacion"))
sys.path.insert(0, str(RAIZ / "03_modelo"))
from preparar import CLASES, Hojas, buscar_imagenes  # noqa: E402
from modelo import BeansCNN  # noqa: E402


def metricas(matriz):
    por_clase = {}
    for i, clase in enumerate(CLASES):
        verdaderos_positivos = matriz[i][i]
        predichos = sum(fila[i] for fila in matriz)
        reales = sum(matriz[i])
        precision = verdaderos_positivos / predichos if predichos else 0.0
        sensibilidad = verdaderos_positivos / reales if reales else 0.0
        f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if precision + sensibilidad else 0.0
        por_clase[clase] = {
            "aciertos": verdaderos_positivos,
            "total_real": reales,
            "precision": round(precision, 4),
            "sensibilidad": round(sensibilidad, 4),
            "f1": round(f1, 4),
        }
    total = sum(sum(fila) for fila in matriz)
    return {
        "total": total,
        "aciertos": sum(matriz[i][i] for i in range(len(CLASES))),
        "exactitud": round(sum(matriz[i][i] for i in range(len(CLASES))) / total, 4),
        "f1_macro": round(sum(x["f1"] for x in por_clase.values()) / len(CLASES), 4),
        "por_clase": por_clase,
    }


def evaluar(datos: Path, modelo_archivo: Path, salida: Path):
    grupos = buscar_imagenes(datos)
    muestras = grupos["test"]
    if len(muestras) != 128:
        raise ValueError(f"Se esperaban 128 imágenes de prueba, hay {len(muestras)}")
    paquete = torch.load(modelo_archivo, map_location="cpu", weights_only=True)
    if tuple(paquete["clases"]) != CLASES:
        raise ValueError("El orden de clases del modelo no coincide con el de los datos")
    modelo = BeansCNN(filtros=paquete["filtros"])
    modelo.load_state_dict(paquete["pesos"])
    modelo.eval()
    cargador = DataLoader(Hojas(muestras, entrenamiento=False), batch_size=32, shuffle=False)
    matriz = [[0] * len(CLASES) for _ in CLASES]
    predicciones = []
    posicion = 0
    with torch.inference_mode():
        for imagenes, etiquetas in cargador:
            probabilidades = torch.softmax(modelo(imagenes), dim=1)
            for etiqueta, probabilidades_foto in zip(etiquetas.tolist(), probabilidades.tolist()):
                predicha = max(range(len(CLASES)), key=lambda i: probabilidades_foto[i])
                matriz[etiqueta][predicha] += 1
                predicciones.append({
                    "archivo": muestras[posicion][0].name,
                    "clase_real": CLASES[etiqueta],
                    "clase_predicha": CLASES[predicha],
                    "confianza_prediccion": round(probabilidades_foto[predicha], 4),
                    "correcta": etiqueta == predicha,
                })
                posicion += 1
    resultado = {
        "modelo": modelo_archivo.name,
        "epoca_modelo": paquete["epoca"],
        "orden_matriz_confusion": list(CLASES),
        "matriz_confusion": matriz,
        "metricas": metricas(matriz),
        "predicciones": predicciones,
    }
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "prueba.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    with (salida / "errores.csv").open("w", newline="", encoding="utf-8") as archivo:
        columnas = ("archivo", "clase_real", "clase_predicha", "confianza_prediccion")
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows({k: fila[k] for k in columnas} for fila in predicciones if not fila["correcta"])
    print(json.dumps({"matriz_confusion": matriz, "metricas": resultado["metricas"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, required=True)
    parser.add_argument("--modelo", type=Path, default=RAIZ / "04_entrenamiento" / "resultados" / "sin_aumento.pt")
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados")
    args = parser.parse_args()
    evaluar(args.datos, args.modelo, args.salida)

