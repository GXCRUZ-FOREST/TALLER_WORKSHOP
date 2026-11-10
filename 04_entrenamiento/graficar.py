"""Dibuja las curvas de validación del historial sin bibliotecas adicionales."""

import argparse
import json
from pathlib import Path


COLORES = {"sin_aumento": "#2563eb", "con_aumento": "#dc2626"}


def panel(historiales, campo, x0, y0, ancho, alto, titulo, minimo, maximo):
    partes = [f'<text x="{x0}" y="{y0 - 14}" font-size="16" font-weight="bold">{titulo}</text>',
              f'<rect x="{x0}" y="{y0}" width="{ancho}" height="{alto}" fill="white" stroke="#888"/>']
    for ejecucion in historiales:
        filas = ejecucion["historial"]
        puntos = []
        for i, fila in enumerate(filas):
            x = x0 + i * ancho / max(1, len(filas) - 1)
            y = y0 + alto * (1 - (fila[campo] - minimo) / (maximo - minimo))
            puntos.append(f"{x:.1f},{y:.1f}")
        color = COLORES[ejecucion["nombre"]]
        partes.append(f'<polyline points="{" ".join(puntos)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    partes.append(f'<text x="{x0}" y="{y0 + alto + 20}" font-size="12">Época 1</text>')
    partes.append(f'<text x="{x0 + ancho - 65}" y="{y0 + alto + 20}" font-size="12">Época 20</text>')
    partes.append(f'<text x="{x0 + 4}" y="{y0 + 14}" font-size="12">{maximo:.2f}</text>')
    partes.append(f'<text x="{x0 + 4}" y="{y0 + alto - 5}" font-size="12">{minimo:.2f}</text>')
    return partes


def crear_svg(datos):
    ejecuciones = datos["ejecuciones"]
    perdidas = [fila["validacion_perdida"] for e in ejecuciones for fila in e["historial"]]
    max_perdida = max(1.0, max(perdidas) * 1.05)
    partes = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="365" viewBox="0 0 900 365">',
              '<rect width="900" height="365" fill="#f8fafc"/>',
              '<text x="30" y="31" font-size="20" font-weight="bold">Validación durante el entrenamiento</text>']
    partes += panel(ejecuciones, "validacion_exactitud", 35, 82, 390, 220,
                    "Exactitud (0 a 1)", 0, 1)
    partes += panel(ejecuciones, "validacion_perdida", 475, 82, 390, 220,
                    "Pérdida (menor es mejor)", 0, max_perdida)
    for i, (nombre, color) in enumerate(COLORES.items()):
        x = 35 + i * 190
        partes.append(f'<line x1="{x}" y1="340" x2="{x + 25}" y2="340" stroke="{color}" stroke-width="3"/>')
        partes.append(f'<text x="{x + 32}" y="344" font-size="13">{nombre.replace("_", " ")}</text>')
    partes.append("</svg>")
    return "\n".join(partes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historial", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    datos = json.loads(args.historial.read_text(encoding="utf-8"))
    args.salida.write_text(crear_svg(datos), encoding="utf-8")

