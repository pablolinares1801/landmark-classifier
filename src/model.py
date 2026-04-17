import torch
import torch.nn as nn


class LandmarkCNN(nn.Module):
    """
    CNN diseñada desde cero para clasificación de landmarks.
    
    Arquitectura:
        - 4 bloques convolucionales (conv + BN + ReLU + MaxPool + Dropout)
        - 2 capas fully connected
        - Salida: num_classes (50 por defecto)
    
    Justificación:
        - 4 capas conv permiten extraer features desde bordes simples
          hasta patrones complejos (arcos, cúpulas, fachadas).
        - Batch Normalization estabiliza el entrenamiento y permite
          usar learning rates más altos.
        - Dropout reduce overfitting dado el tamaño moderado del dataset.
        - MaxPooling reduce dimensionalidad progresivamente.
    """

    def __init__(self, num_classes=50):
        super(LandmarkCNN, self).__init__()

        # ── Bloque 1: 3 → 32 canales
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112
            nn.Dropout2d(0.1),
        )

        # ── Bloque 2: 32 → 64 canales
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56
            nn.Dropout2d(0.1),
        )

        # ── Bloque 3: 64 → 128 canales
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
            nn.Dropout2d(0.2),
        )

        # ── Bloque 4: 128 → 256 canales
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
            nn.Dropout2d(0.2),
        )

        # ── Clasificador fully connected
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=50, device='cpu'):
    """Instancia el modelo y lo mueve al dispositivo indicado."""
    model = LandmarkCNN(num_classes=num_classes)
    model = model.to(device)
    return model