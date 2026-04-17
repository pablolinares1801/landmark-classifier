import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter


# ─────────────────────────────────────────────
# 1. TRANSFORMACIONES
# ─────────────────────────────────────────────

def get_transforms():
    """
    Transformaciones separadas para entrenamiento y validación/test.
    - Train: incluye data augmentation
    - Val/Test: solo resize, crop y normalización (sin augmentation)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Data augmentation solo en entrenamiento
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transforms, val_test_transforms


# ─────────────────────────────────────────────
# 2. CARGA DEL DATASET Y DATALOADERS
# ─────────────────────────────────────────────
data_dir= r'C:\Users\MSI LAPTOP\Desktop\MAESRTIA CINCIENA DATOS\MODULO REDES NEURONALES\pytorch-intro\PROYECTO 1 - LANDMARKS REDES NEURONALES\landmark_images'

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, num_workers=2):
    """
    Carga el dataset desde data_dir/train y data_dir/test.
    Divide el train en train + validation (80/20 por defecto).

    Args:
        data_dir   : ruta raíz del dataset (debe contener carpetas train/ y test/)
        batch_size : tamaño de batch
        val_split  : fracción del train usada para validación
        num_workers: workers para carga paralela (0 en Colab si hay problemas)

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transforms, val_test_transforms = get_transforms()

    # Dataset completo de entrenamiento (con augmentation)
    full_train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )

    # Dataset de entrenamiento sin augmentation (para el split de validación)
    full_val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=val_test_transforms
    )

    # Split train / validation
    n_total = len(full_train_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    indices = list(range(n_total))
    torch.manual_seed(42)
    train_indices, val_indices = random_split(indices, [n_train, n_val])

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(full_val_dataset,   val_indices)

    # Dataset de test
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_test_transforms
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    class_names = full_train_dataset.classes

    print(f"Clases encontradas : {len(class_names)}")
    print(f"Imágenes en train  : {len(train_dataset)}")
    print(f"Imágenes en val    : {len(val_dataset)}")
    print(f"Imágenes en test   : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────
# 3. VISUALIZACIÓN DE EJEMPLOS
# ─────────────────────────────────────────────

def imshow(img, mean, std):
    """Desnormaliza y muestra una imagen tensor."""
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean          # desnormalizar
    img = np.clip(img, 0, 1)
    return img


def visualize_samples(loader, class_names, n=5):
    """
    Muestra n imágenes de ejemplo con sus etiquetas.
    Requerimiento Fase 1, punto 2.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    images, labels = next(iter(loader))

    fig, axes = plt.subplots(1, n, figsize=(20, 4))
    for i in range(n):
        ax = axes[i]
        ax.imshow(imshow(images[i], mean, std))
        ax.set_title(class_names[labels[i]], fontsize=9)
        ax.axis('off')
    plt.suptitle("Ejemplos del dataset", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 4. DISTRIBUCIÓN DE CLASES
# ─────────────────────────────────────────────

def plot_class_distribution(data_dir, class_names):
    """
    Gráfico de barras con cantidad de imágenes por landmark.
    Requerimiento Fase 1, punto 3.
    """
    train_path = os.path.join(data_dir, 'train')
    counts = []
    for cls in class_names:
        cls_path = os.path.join(train_path, cls)
        counts.append(len(os.listdir(cls_path)))

    sorted_pairs = sorted(zip(counts, class_names), reverse=True)
    counts_sorted, names_sorted = zip(*sorted_pairs)

    plt.figure(figsize=(20, 6))
    plt.bar(range(len(names_sorted)), counts_sorted, color='steelblue')
    plt.xticks(range(len(names_sorted)), names_sorted, rotation=90, fontsize=7)
    plt.xlabel("Landmark")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución de imágenes por clase (train)")
    plt.tight_layout()
    plt.show()

    print(f"\nClase con más imágenes  : {names_sorted[0]}  ({counts_sorted[0]} imgs)")
    print(f"Clase con menos imágenes: {names_sorted[-1]} ({counts_sorted[-1]} imgs)")