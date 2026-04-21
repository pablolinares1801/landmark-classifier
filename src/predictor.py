import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def get_inference_transforms():
    """Transformaciones para inferencia (igual que val/test)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def predict_landmarks(img_path, k, model, class_names, device):
    """
    Predice los top-k landmarks más probables de una imagen.

    Args:
        img_path   : ruta de la imagen
        k          : número de predicciones a retornar
        model      : modelo cargado (TorchScript o normal)
        class_names: lista de nombres de clases
        device     : 'cuda' o 'cpu'

    Returns:
        topk_names : lista con los k nombres de landmarks
        topk_probs : lista con las k probabilidades
    """
    # Cargar y transformar imagen
    transform = get_inference_transforms()
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # añadir dimensión batch

    # Inferencia
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)
        topk_probs, topk_indices = torch.topk(probs, k, dim=1)

    # Convertir a listas
    topk_probs   = topk_probs.squeeze().cpu().tolist()
    topk_indices = topk_indices.squeeze().cpu().tolist()

    if k == 1:
        topk_probs   = [topk_probs]
        topk_indices = [topk_indices]

    topk_names = [class_names[i] for i in topk_indices]

    return topk_names, topk_probs


def show_prediction(img_path, k, model, class_names, device):
    """
    Muestra la imagen junto a las top-k predicciones en un gráfico.
    """
    topk_names, topk_probs = predict_landmarks(
        img_path, k, model, class_names, device
    )

    img = Image.open(img_path).convert('RGB')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Imagen
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(os.path.basename(img_path), fontsize=10)

    # Barras de probabilidad
    colores = ['#2ecc71' if i == 0 else '#3498db' for i in range(k)]
    bars = ax2.barh(
        range(k),
        [p * 100 for p in topk_probs],
        color=colores,
        edgecolor='black'
    )

    ax2.set_yticks(range(k))
    ax2.set_yticklabels(
        [name.split('.')[-1].replace('_', ' ') for name in topk_names],
        fontsize=9
    )
    ax2.invert_yaxis()
    ax2.set_xlabel('Probabilidad (%)')
    ax2.set_title(f'Top-{k} Predicciones')
    ax2.set_xlim(0, 100)

    for bar, prob in zip(bars, topk_probs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=9)

    plt.suptitle('Landmark Classifier — Predicción', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return topk_names, topk_probs