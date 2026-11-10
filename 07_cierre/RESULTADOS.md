# Etapa 7: cierre y guía para reutilizar el proyecto

Las seis etapas previstas están completas. Esta etapa reúne una ruta para ejecutar el proyecto desde GitHub sin guardar una segunda copia del conjunto de imágenes en tu PC.

## Abrir en la nube

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GXCRUZ-FOREST/TALLER_WORKSHOP/blob/main/07_cierre/ejecutar_en_colab.ipynb)

El [cuaderno de Colab](ejecutar_en_colab.ipynb) copia este repositorio a una sesión temporal, descarga `beans128.npz` desde el [repositorio del taller UNTRM](https://github.com/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/data/README.md) y extrae las 1.295 fotos **dentro de Colab**. El [programa de extracción](obtener_datos.py) no altera el repositorio de la UNTRM. En el cuaderno puedes revisar los datos, repetir el entrenamiento o saltarlo para usar los pesos ya guardados, evaluar y probar otra foto.

Los archivos creados durante una sesión de Colab son temporales. Los resultados de este trabajo que ya están publicados en GitHub permanecen en este repositorio; si repites o cambias un experimento en Colab, tendrás que guardar sus nuevos resultados allí explícitamente.

## Resultado final de este experimento

- **Datos:** 1.295 imágenes de hojas de frijol, tres clases.
- **Modelo:** CNN de tres bloques, 32.739 parámetros, iniciada con pesos aleatorios.
- **Selección:** menor pérdida de validación entre las ejecuciones con y sin aumento; se eligió la de [sin aumento, época 19](../04_entrenamiento/RESULTADOS.md).
- **Prueba:** [96 de 128 aciertos, 75,0 % de exactitud](../05_evaluacion/RESULTADOS.md); la roya del frijol tuvo 51,2 % de sensibilidad y fue la clase más difícil.
- **Uso:** [clasificación y vistas internas](../06_interpretacion/RESULTADOS.md) de una foto.

Este modelo es un ejercicio de aprendizaje con imágenes del mismo conjunto. Para usarlo como herramienta de campo harían falta pruebas con fotografías nuevas, de otros lugares y condiciones de captura, y una evaluación de cuándo debe abstenerse de responder.

## Dónde está cada etapa

1. [Exploración y verificación de datos](../01_datos/RESULTADOS.md)
2. [Preparación de imágenes](../02_preparacion/RESULTADOS.md)
3. [Arquitectura de la CNN](../03_modelo/RESULTADOS.md)
4. [Entrenamiento y selección](../04_entrenamiento/RESULTADOS.md)
5. [Evaluación en prueba](../05_evaluacion/RESULTADOS.md)
6. [Interpretación e inferencia](../06_interpretacion/RESULTADOS.md)

El repositorio original del taller se utilizó solo como referencia y fuente de datos. Los archivos creados para este proyecto están en `GXCRUZ-FOREST/TALLER_WORKSHOP`.

