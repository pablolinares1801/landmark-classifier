import torch
import torch.nn as nn


class LandmarkCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(LandmarkCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

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
    model = LandmarkCNN(num_classes=num_classes)
    model = model.to(device)
    return model

import torchvision.models as models

def get_transfer_model(model_name='resnet18', num_classes=50, device='cpu'):
    """
    Carga un modelo preentrenado y reemplaza la capa final.
    
    Justificación de ResNet18:
    - Preentrenado en ImageNet (1.2M imágenes, 1000 clases)
    - Ya sabe detectar bordes, texturas, formas y objetos complejos
    - Arquitectura residual evita el problema del gradiente que desaparece
    - ResNet18 es liviano y rápido, ideal para fine-tuning con dataset pequeño
    - ResNet50 es más potente pero más lento, lo usamos como segundo modelo
    
    Args:
        model_name : 'resnet18' o 'resnet50'
        num_classes: número de clases (50)
        device     : 'cuda' o 'cpu'
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Congelar todas las capas del feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la capa final con una nueva para 50 clases
    if model_name == 'resnet18':
        in_features = model.fc.in_features  # 512
    else:
        in_features = model.fc.in_features  # 2048

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)

    # Contar parámetros entrenables vs totales
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {model_name}")
    print(f"Parámetros totales    : {total:,}")
    print(f"Parámetros entrenables: {trainable:,}  ({trainable/total*100:.1f}%)")

    return model