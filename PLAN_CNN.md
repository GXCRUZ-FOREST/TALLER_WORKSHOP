# Plan de elaboración de la CNN

## Objetivo

Construir desde cero una CNN pequeña en PyTorch para reconocer **mancha angular**, **roya del frijol** y **hoja sana**. «Desde cero» significa inicializar los pesos al azar y aprenderlos con las imágenes de entrenamiento; no se empleará transferencia de aprendizaje.

El punto de partida será la arquitectura `BeansCNN` del [código compartido del taller UNTRM](https://github.com/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/cnn_common.py) y su [práctica en Colab](https://github.com/lobaton/AI-Workshop-UNTRM/blob/main/session2_inside_the_black_box/src/CNN_colab.ipynb). El modelo entrenado antes de establecer este plan fue un ensayo exploratorio con otra arquitectura; sus cifras no se usarán para elegir variantes.

## Datos y protocolo

Los 1.295 archivos JPEG de `beans128` proporcionados para este proyecto se compararon byte por byte con el archivo `beans128.npz` del taller: **los 1.295 coinciden**. Conservaremos las particiones y etiquetas originales:

| Partición | Mancha angular | Roya | Sana | Total |
|---|---:|---:|---:|---:|
| Entrenamiento | 345 | 348 | 341 | 1.034 |
| Validación | 44 | 45 | 44 | 133 |
| Prueba | 43 | 43 | 42 | 128 |

La validación servirá para escoger época y ajustes. El conjunto de prueba se reservará para evaluar el diseño elegido. Como el ensayo exploratorio anterior ya produjo una cifra de prueba, no se usará esa cifra para tomar decisiones en este proyecto.

## Etapas

### 1. Explorar y verificar imágenes

- Mostrar ejemplos de cada clase y explicar las diferencias visibles.
- Confirmar tamaños, formato RGB, etiquetas, distribución y legibilidad de todos los archivos.
- Registrar las observaciones y las posibles fuentes de confusión entre clases.

**Entrega:** exploración reproducible y breve descripción del conjunto.

### 2. Preparar la entrada

- Cargar las imágenes de 128 × 128 y convertirlas a tensores con tres canales.
- Escalar los píxeles a `[-1, 1]`, como en el código del taller.
- Aplicar volteos y rotaciones de 90° aleatorios solo al entrenamiento; validación y prueba conservarán sus imágenes originales.
- Fijar una semilla para repetir los experimentos.

**Entrega:** cargador de datos y comprobación visual de las transformaciones.

### 3. Construir el modelo base

- Bloque 1: convolución `5 × 5`, 16 filtros, ReLU y max pooling `2 × 2`.
- Bloque 2: convolución `5 × 5`, 32 filtros, ReLU y max pooling `2 × 2`.
- Bloque 3: convolución `3 × 3`, 64 filtros, ReLU y max pooling `2 × 2`.
- Promedio global de cada mapa de características y capa lineal con tres salidas.
- Mostrar la forma de salida y el número de parámetros de cada bloque.

**Entrega:** CNN pequeña fiel al modelo del taller, con una explicación de convolución, ReLU y pooling.

### 4. Entrenar y seleccionar

- Primera ejecución con las 1.034 imágenes de entrenamiento, 16 filtros iniciales, lotes de 32, optimizador Adam y tasa de aprendizaje `0,003`.
- Empezar con 20 épocas, como en la práctica. Extender el entrenamiento solo si la curva de validación lo justifica.
- Registrar pérdida y exactitud de entrenamiento y validación por época; guardar los pesos de la mejor época según validación.
- Hacer una comparación controlada con y sin aumento de datos, cambiando únicamente ese factor y decidiendo con validación.

**Entrega:** modelo elegido, configuración, semilla y curvas de aprendizaje.

### 5. Evaluar y analizar errores

- Evaluar el modelo fijado en las 128 imágenes de prueba.
- Informar exactitud total, precisión, sensibilidad y F1 por clase, además de la matriz de confusión.
- Revisar imágenes mal clasificadas y describir patrones de error, en especial las confusiones entre las dos enfermedades.

**Entrega:** informe de resultados con cifras reproducibles y límites claros.

### 6. Interpretar y usar

- Mostrar los filtros aprendidos en la primera capa y los mapas de características de ejemplos representativos.
- Clasificar una foto aportada por el usuario con el mismo preprocesamiento.
- Explicar que una foto de otra planta o tomada en condiciones muy distintas puede recibir igualmente una de las tres etiquetas; la red no incorpora una categoría «desconocida».

**Entrega:** demostración de inferencia y explicación visual de lo que aprendió la CNN.

## Organización y almacenamiento

El repositorio contendrá el cuaderno o código, instrucciones, configuración, resultados y el modelo pequeño. El conjunto de imágenes se enlazará desde el repositorio original del taller en lugar de duplicar sus 1.295 archivos. No se subirán entornos de Python, cachés ni archivos temporales.

Cada avance terminado se registrará con un commit y se enviará a `main` para que GitHub sea la copia permanente del trabajo.

