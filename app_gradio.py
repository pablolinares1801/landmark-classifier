import sys
sys.path.append('C:/Users/MSI LAPTOP/landmark-classifier')

import torch
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np

from src.model import get_transfer_model
from src.data import get_data_loaders
from src.predictor import predict_landmarks

# ── Configuración ──────────────────────────
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'C:/Users/MSI LAPTOP/landmark_images/landmark_images'
MODEL_PATH = 'C:/Users/MSI LAPTOP/landmark-classifier/models/best_resnet50.pt'

# ── Cargar class_names ─────────────────────
print("Cargando dataset...")
_, _, _, class_names = get_data_loaders(
    data_dir=DATA_DIR,
    batch_size=32,
    val_split=0.2,
    num_workers=0
)

# ── Cargar modelo ──────────────────────────
print("Cargando modelo...")
model = get_transfer_model(
    model_name  = 'resnet50',
    num_classes = 50,
    device      = DEVICE
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Listo!")

# ── Función de predicción para Gradio ──────
def classify_landmark(image, top_k):
    """
    Recibe una imagen PIL y retorna un gráfico con las top-k predicciones.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)
        topk_probs, topk_indices = torch.topk(probs, int(top_k), dim=1)

    topk_probs   = topk_probs.squeeze().cpu().tolist()
    topk_indices = topk_indices.squeeze().cpu().tolist()

    if int(top_k) == 1:
        topk_probs   = [topk_probs]
        topk_indices = [topk_indices]

    nombres = [class_names[i].split('.')[-1].replace('_', ' ')
               for i in topk_indices]
    probs_pct = [p * 100 for p in topk_probs]

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    colores = ['#2ecc71' if i == 0 else '#3498db'
               for i in range(int(top_k))]
    bars = ax.barh(range(int(top_k)), probs_pct,
                   color=colores, edgecolor='black')
    ax.set_yticks(range(int(top_k)))
    ax.set_yticklabels(nombres, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilidad (%)', fontsize=11)
    ax.set_title('Prediccion de Landmark', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)

    for bar, prob in zip(bars, probs_pct):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()

    # Resultado como texto
    resultado = ""
    for nombre, prob in zip(nombres, probs_pct):
        resultado += f"{nombre}: {prob:.1f}%\n"

    return fig, resultado


# ── Interfaz Gradio ────────────────────────
with gr.Blocks(title="Landmark Classifier") as demo:

    gr.Markdown("# Landmark Classifier")
    gr.Markdown(
        "Sube una foto y el modelo identificará el landmark que contiene. "
        "Usa ResNet50 entrenado con Transfer Learning — 80.24% de test accuracy."
    )

    with gr.Row():
        with gr.Column():
            imagen_input = gr.Image(type="pil", label="Sube tu imagen")
            topk_slider  = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Cuantos landmarks mostrar (Top-K)"
            )
            btn = gr.Button("Clasificar", variant="primary")

        with gr.Column():
            grafico_output = gr.Plot(label="Predicciones")
            texto_output   = gr.Textbox(label="Resultados", lines=6)

    btn.click(
        fn=classify_landmark,
        inputs=[imagen_input, topk_slider],
        outputs=[grafico_output, texto_output]
    )

    gr.Markdown("### Ejemplos de landmarks en el dataset")
    gr.Markdown(
        "Haleakala National Park, Mount Rainier, Niagara Falls, "
        "Sydney Opera House, Matterhorn, Machu Picchu, entre otros."
    )

if __name__ == "__main__":
    demo.launch(share=False)