# landmark-classifier
Clasificación de Landmarks con CNN - Maestría IA
# landmark Classifier — CNN con PyTorch

Proyecto final del módulo de Redes Neuronales — Maestría en Ciencia de Datos e IA Aplicada.

Sistema de clasificación automática de landmarks (puntos de referencia) a partir de imágenes, utilizando Redes Neuronales Convolucionales (CNN) en PyTorch. El proyecto compara una CNN diseñada desde cero contra modelos preentrenados (ResNet18 y ResNet50) aplicando Transfer Learning.

##  Objetivo

Inferir la ubicación geográfica de una fotografía identificando el landmark presente en ella, útil para servicios de organización automática de fotos cuando no hay metadatos GPS disponibles.

##  Resultados

| Modelo | Épocas | Test Accuracy | Estado |
|--------|--------|--------------|--------|
| CNN desde cero | 130 | 19.9% | Fase 2 completada 
| ResNet18 (Transfer Learning) | 35 | **77.68%** | Supera mínimo (70%) |
| ResNet50 (Transfer Learning) | 25 | **80.24%** | Supera bonus (75%) |

##  Estructura del Proyecto
landmark-classifier/
├── README.md
├── cnn_from_scratch.ipynb    # Fase 2: CNN diseñada desde cero
├── transfer_learning.ipynb   # Fase 3: Transfer Learning con ResNet
├── app.ipynb                 # Fase 4: Aplicación de predicción
├── src/
│   ├── data.py               # DataLoaders y transformaciones
│   ├── model.py              # Arquitecturas CNN y Transfer Learning
│   ├── train.py              # Loops de entrenamiento y evaluación
│   └── predictor.py          # Función predict_landmarks()
├── models/                   # Modelos .pt exportados con TorchScript
└── .gitignore

## Stack Tecnológico

- Python 3.11
- PyTorch 2.5 con CUDA 12.1
- torchvision para modelos preentrenados
- matplotlib para visualizaciones
- Jupyter Notebook para experimentación

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/pablolinares1801/landmark-classifier.git
cd landmark-classifier

# Instalar dependencias (con GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy jupyter pillow
```

## Ejecución

1. Descargar el dataset (no incluido en el repo por tamaño)
   - Estructura esperada: `landmark_images/train/` y `landmark_images/test/`
   - Cada subcarpeta corresponde a una clase (50 landmarks)

2. Ajustar la ruta del dataset en cada notebook:
```python
   DATA_DIR = 'ruta/a/landmark_images'
```

3. Ejecutar los notebooks en orden:
   - `cnn_from_scratch.ipynb` — Fase 2
   - `transfer_learning.ipynb` — Fase 3
   - `app.ipynb` — Fase 4

## Metodología

### Fase 1 — Preprocesamiento
- Resize a 256px, CenterCrop a 224px
- Normalización con media y desviación estándar de ImageNet
- Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Split 80/20 de train en train y validación

### Fase 2 — CNN desde cero
- 4 bloques convolucionales (Conv + BatchNorm + ReLU + MaxPool + Dropout)
- Clasificador fully connected con Dropout 0.5
- Optimizador Adam con weight_decay=1e-4
- Scheduler CosineAnnealingLR

### Fase 3 — Transfer Learning
- ResNet18 y ResNet50 preentrenados en ImageNet
- Congelación inicial del feature extractor
- Reemplazo de la capa FC final por un clasificador personalizado
- Fine-tuning descongelando `layer4` y `fc` con learning rate bajo

### Fase 4 — Inferencia
- Función `predict_landmarks(img_path, k)` que retorna los top-k landmarks
- Uso del modelo exportado con TorchScript
- Visualización de predicciones con barras de probabilidad

## Video Explicativo

[https://youtu.be/JVw9gyC1yOc](https://youtu.be/JVw9gyC1yOc)

## Autor

Pablo Linares — Maestría en Ciencia de Datos e IA Aplicada  
Universidad Católica Boliviana

## Licencia

Proyecto académico — uso educativo

