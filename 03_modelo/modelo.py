"""CNN pequeña del taller UNTRM, inicializada desde cero.

Uso: python 03_modelo/modelo.py
Este archivo define la arquitectura; todavía no entrena los pesos.
"""

import torch
from torch import nn


class BeansCNN(nn.Module):
    """Tres bloques convolucionales y un clasificador de tres clases."""

    def __init__(self, filtros=16):
        super().__init__()
        if filtros < 1:
            raise ValueError("El número de filtros debe ser positivo")
        self.bloques = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, filtros, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(filtros, 2 * filtros, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(2 * filtros, 4 * filtros, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
        ])
        self.promedio_global = nn.AdaptiveAvgPool2d(1)
        self.clasificador = nn.Linear(4 * filtros, 3)

    def forward(self, x):
        for bloque in self.bloques:
            x = bloque(x)
        x = self.promedio_global(x).flatten(1)
        return self.clasificador(x)

    def mapas_caracteristicas(self, x):
        """Devuelve los mapas tras ReLU de cada bloque, antes de max pooling."""
        mapas = []
        for convolucion, relu, reduccion in self.bloques:
            x = relu(convolucion(x))
            mapas.append(x)
            x = reduccion(x)
        return mapas


def numero_parametros(capa):
    return sum(parametro.numel() for parametro in capa.parameters())


def mostrar_recorrido(modelo):
    """Comprueba las formas con una entrada del mismo tamaño que beans128."""
    x = torch.zeros(1, 3, 128, 128)
    print("Entrada:", tuple(x.shape), "= lote, canales, alto, ancho")
    for indice, bloque in enumerate(modelo.bloques, start=1):
        x = bloque(x)
        print(f"Bloque {indice}: {tuple(x.shape)}; "
              f"parámetros: {numero_parametros(bloque):,}")
    x = modelo.promedio_global(x).flatten(1)
    print("Promedio global:", tuple(x.shape))
    x = modelo.clasificador(x)
    print("Salida:", tuple(x.shape), "= una puntuación por clase")
    print("Parámetros totales:", f"{numero_parametros(modelo):,}")


if __name__ == "__main__":
    torch.manual_seed(0)
    mostrar_recorrido(BeansCNN(filtros=16))

