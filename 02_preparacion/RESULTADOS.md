# Etapa 2: preparar las imágenes para la CNN

El programa [`preparar.py`](preparar.py) convierte cada JPEG en un tensor que PyTorch puede recibir. Usa exactamente las particiones y clases identificadas en la [etapa 1](../01_datos/RESULTADOS.md).

## Del píxel al tensor

Cada foto tiene forma `(alto, ancho, canales) = (128, 128, 3)`. PyTorch espera los canales primero, así que se convierte a `(3, 128, 128)`. Los valores enteros de color de `0` a `255` pasan a decimales entre `−1` y `1` mediante `valor / 127,5 − 1`. Esta es la normalización utilizada por el [código del taller UNTRM](https://github.com/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/cnn_common.py).

| Etiqueta numérica | Clase |
|---:|---|
| 0 | Mancha angular (`angular_leaf_spot`) |
| 1 | Roya del frijol (`bean_rust`) |
| 2 | Hoja sana (`healthy`) |

El nombre del archivo determina la etiqueta para **entrenar y evaluar**. Al clasificar una foto nueva, el modelo recibe los píxeles y no necesita que el nombre contenga la clase.

## Aumento de datos

Durante el entrenamiento se pueden aplicar volteos horizontal y vertical y giros de 0°, 90°, 180° o 270°. La clase sigue siendo la misma. Es una forma de mostrarle distintas orientaciones de las hojas sin crear copias permanentes en el disco. La transformación se hace al cargar la foto.

**Validación y prueba no reciben aumentos aleatorios**: así se pueden comparar resultados de forma estable. Se comprobó que solicitar dos veces la misma imagen de esos grupos devuelve el mismo tensor.

## Comprobación realizada

| Grupo | Fotos | Forma de la primera entrada | Etiqueta de la primera entrada |
|---|---:|---|---|
| Entrenamiento | 1.034 | `(3, 128, 128)` | Mancha angular |
| Validación | 133 | `(3, 128, 128)` | Mancha angular |
| Prueba | 128 | `(3, 128, 128)` | Mancha angular |

Se generó y revisó una vista con una foto original y tres transformaciones por clase. Las lesiones visibles permanecen en la imagen después de los giros y volteos. La vista se puede regenerar con:

```powershell
python 02_preparacion\preparar.py --datos "C:\Users\LENOVO\Downloads\beans128" --vista vista_preparacion.png
```

Se necesitan `numpy`, `Pillow` y `torch`. El archivo PNG se genera solo si se solicita `--vista`; no se guardan copias aumentadas de las 1.034 fotos.

## Decisión para la etapa siguiente

La red recibirá lotes de tensores `(3, 128, 128)` y producirá tres puntuaciones, una por clase. La próxima etapa construirá y explicará la arquitectura pequeña del taller.

