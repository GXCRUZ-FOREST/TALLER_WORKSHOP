# Etapa 1: exploración y verificación de las imágenes

Se revisó la carpeta local `beans128` con [`explorar.py`](explorar.py). Los 1.295 JPEG coinciden byte por byte con los del archivo [`beans128.npz` del taller UNTRM](https://github.com/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/data/README.md). No se copiaron las imágenes a este repositorio.

| Partición | Mancha angular | Roya | Sana | Total |
|---|---:|---:|---:|---:|
| Entrenamiento | 345 | 348 | 341 | 1.034 |
| Validación | 44 | 45 | 44 | 133 |
| Prueba | 43 | 43 | 42 | 128 |

## Comprobaciones

- **1.295 de 1.295** archivos se pudieron abrir como imágenes.
- Todos tienen **128 × 128 píxeles** y modo de color **RGB**.
- Todos siguen el patrón de nombre `número_partición_clase.jpg`.
- No hay duplicados **exactos** por contenido, ni dentro de una partición ni entre particiones. Esta comprobación no descarta fotos distintas de una misma hoja.
- Las clases están aproximadamente equilibradas; una elección al azar entre las tres acertaría alrededor de un tercio de las veces.

## Ejemplos observados

Se inspeccionaron tres fotografías del grupo de entrenamiento: `0001_train_angular_leaf_spot.jpg`, `0346_train_bean_rust.jpg` y `0694_train_healthy.jpg`. La primera muestra áreas pálidas y lesiones visibles; la segunda presenta pequeñas manchas sobre una hoja verde; la tercera se ve predominantemente verde y sin lesiones evidentes en este ejemplo. Estas observaciones describen solo las fotos vistas: el aspecto de una enfermedad puede variar entre imágenes.

## Cómo repetir la revisión

Con Python y Pillow instalados:

```powershell
python 01_datos\explorar.py --datos "C:\Users\LENOVO\Downloads\beans128"
```

El programa imprime el número de imágenes por clase y partición, tamaños, modos de color, errores de lectura y duplicados exactos.

## Decisión para la etapa siguiente

Conservaremos las particiones originales. El entrenamiento aprenderá de 1.034 fotos; la validación orientará los ajustes; las 128 de prueba se usarán para evaluar el modelo elegido. La próxima etapa preparará los tensores y mostrará las transformaciones aplicadas únicamente al entrenamiento.

