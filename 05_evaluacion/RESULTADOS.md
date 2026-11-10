# Etapa 5: evaluación en imágenes de prueba

Se cargaron los pesos del modelo **sin aumento**, elegido en la [etapa 4](../04_entrenamiento/RESULTADOS.md) por su menor pérdida de validación en la época 19. El programa [`evaluar.py`](evaluar.py) clasificó las **128 fotos de prueba** sin cambiar los pesos.

## Resultado general

- **96 aciertos de 128 fotos: exactitud del 75,0 %.**
- **F1 macro: 74,1 %.** Este promedio da el mismo peso a las tres clases.
- Como referencia, elegir al azar entre tres clases equilibradas acertaría cerca del 33 %.

| Clase real | Fotos | Aciertos | Sensibilidad | Precisión | F1 |
|---|---:|---:|---:|---:|---:|
| Mancha angular | 43 | 36 | 83,7 % | 69,2 % | 75,8 % |
| Roya del frijol | 43 | 22 | **51,2 %** | 75,9 % | 61,1 % |
| Hoja sana | 42 | 38 | 90,5 % | 80,9 % | 85,4 % |

**Sensibilidad** indica cuántas fotos reales de una clase reconoce. **Precisión** indica cuántas de las fotos a las que asignó esa clase estaban bien clasificadas. La roya tiene la sensibilidad más baja: el modelo deja escapar casi la mitad de sus casos.

## Matriz de confusión

Cada fila es la **clase real** y cada columna, la **predicción**:

| Real ↓ / Predicha → | Mancha angular | Roya | Sana |
|---|---:|---:|---:|
| Mancha angular | **36** | 5 | 2 |
| Roya | **14** | **22** | 7 |
| Sana | 2 | 2 | **38** |

La confusión más frecuente fue **roya → mancha angular**: 14 fotos. También hubo 7 fotos de roya consideradas sanas. Por eso el 75 % global no describe por sí solo el desempeño en cada enfermedad.

## Revisión de algunos errores

Se inspeccionaron visualmente cuatro casos del archivo [`errores.csv`](resultados/errores.csv):

- `1218_test_bean_rust.jpg`: etiquetada como roya, predicha como mancha angular; se observan manchas y cambios de color en la hoja.
- `1232_test_bean_rust.jpg`: etiquetada como roya, predicha como sana con 98,0 % de probabilidad estimada, pese a que se aprecian zonas moteadas. Es un ejemplo de error con alta confianza.
- `1283_test_healthy.jpg`: etiquetada como sana, predicha como roya; la hoja aparece pálida en la fotografía.
- `1195_test_angular_leaf_spot.jpg`: etiquetada como mancha angular, predicha como sana; las lesiones visibles son pequeñas en esta imagen.

Estas observaciones describen las fotos revisadas. No prueban qué zona de la imagen causó cada decisión; eso se estudiará mediante mapas de características en la etapa 6.

## Archivos y reproducción

El detalle de las 128 predicciones y las métricas está en [`prueba.json`](resultados/prueba.json). La lista de las 32 fotos mal clasificadas está en [`errores.csv`](resultados/errores.csv). Se reproducen con:

```powershell
python 05_evaluacion\evaluar.py --datos "C:\Users\LENOVO\Downloads\beans128"
```

Este resultado corresponde a imágenes del mismo conjunto y sus condiciones de captura. No establece el rendimiento en otras plantas, cámaras o lugares. La siguiente etapa mostrará qué patrones activan los filtros y probará una foto aportada por el usuario.

