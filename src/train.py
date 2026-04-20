import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import os


def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=0.001, save_path='models/best_cnn_scratch.pt'):
    """
    Loop de entrenamiento completo con:
        - Registro de loss y accuracy por época (train y val)
        - Guardado del mejor modelo (menor val_loss)
        - Retorna historial para graficar curvas

    Args:
        model       : modelo CNN a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader  : DataLoader de validación
        device      : 'cuda' o 'cpu'
        epochs      : número de épocas
        lr          : learning rate inicial
        save_path   : ruta donde guardar el mejor modelo

    Returns:
        history: dict con listas de loss y accuracy de train y val
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):

        # ── ENTRENAMIENTO ──────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # ── VALIDACIÓN ────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        # ── MÉTRICAS ──────────────────────────────
        epoch_train_loss = train_loss / train_total
        epoch_val_loss   = val_loss   / val_total
        epoch_train_acc  = train_correct / train_total * 100
        epoch_val_acc    = val_correct   / val_total   * 100

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Guardar mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

        print(f"Época [{epoch:02d}/{epochs}] "
              f"| Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.1f}% "
              f"| Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.1f}%"
              + (" ✓ mejor modelo" if epoch_val_loss == best_val_loss else ""))

    # Restaurar mejores pesos
    model.load_state_dict(best_weights)
    print(f"\nEntrenamiento finalizado. Mejor val_loss: {best_val_loss:.4f}")
    return history


def plot_training_curves(history):
    """Grafica curvas de loss y accuracy (train vs val)."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', markersize=3, label='Val Loss')
    ax1.set_title('Curva de Pérdida')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   'r-o', markersize=3, label='Val Acc')
    ax2.set_title('Curva de Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('CNN desde Cero — Entrenamiento', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device):
    """Evalúa el modelo en el conjunto de test y retorna el accuracy."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy