import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from config import DEVICE, GESTURE_CLASSES, MODEL_PATH, PLOTS_DIR
from dataset import download_dataset, get_data_loaders
from model import build_model

def train():
    """Train the gesture recognition model."""

    # Download data if needed
    download_dataset()

    # Load data
    train_loader, val_loader, _, train_ds, val_ds, _ = get_data_loaders()

    # Build model
    model = build_model()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # Training loop
    train_losses, val_losses, val_f1s = [], [], []

    print(f'\nTraining on {DEVICE} for {10} epochs...\n')

    for epoch in range(1, 10 + 1):
        # Train
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_loss += criterion(out, labels).item() * imgs.size(0)
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_ds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(macro_f1)
        scheduler.step()

        print(f'Epoch {epoch:>2}/{10} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Macro F1: {macro_f1:.4f}')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': GESTURE_CLASSES,
        'macro_f1': val_f1s[-1],
    }, MODEL_PATH)
    print(f'\nModel saved to {MODEL_PATH}')

    # Plot training curves
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()

    ax2.plot(val_f1s, color='green', label='Val Macro F1')
    ax2.axhline(0.9, color='red', linestyle='--', label='Target (0.90)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.set_title('Validation Macro F1')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150)
    plt.show()
    print(f'Training curves saved to {PLOTS_DIR}/training_curves.png')

    return model


if __name__ == '__main__':
    train()