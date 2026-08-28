import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from dataset import download_dataset, get_data_loaders
from model import build_model

from config import (
    DEVICE, GESTURE_CLASSES, MODEL_PATH, PLOTS_DIR, NUM_EPOCHS,
    LEARNING_RATE, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
    LOG_EVERY_N_BATCHES, ARCHITECTURE)

def train(architecture=ARCHITECTURE, num_epochs=NUM_EPOCHS):
    torch.manual_seed(42)

    download_dataset()
    train_loader, val_loader, _, train_ds, val_ds, _ = get_data_loaders()

    model = build_model(architecture=architecture)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Training loop
    model_path = MODEL_PATH.replace('.pth', f'_{architecture}.pth') # for diff architectures
    train_losses, val_losses, val_f1s = [], [], []
    best_f1 = 0.0

    print(f'\nTraining on {DEVICE} for {num_epochs} epochs...\n')

    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

            if i % LOG_EVERY_N_BATCHES == 0:
                print(f'  epoch {epoch} | batch {i}/{len(train_loader)} | loss: {loss.item():.4f}')

        train_loss = running_loss / len(train_ds)

        # validate
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

        print(f'Epoch {epoch:>2}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Macro F1: {macro_f1:.4f}')

        # save curr best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': GESTURE_CLASSES,
                'architecture': architecture,
                'macro_f1': macro_f1,
                'epoch': epoch
            }, model_path)
            print(f'  → New best model saved (F1={macro_f1:.4f})')

    print(f'\nTraining complete. Best Val Macro F1: {best_f1:.4f}')
    print(f'Model saved to {model_path}')

    # Plot training curves
    plot_path = os.path.join(PLOTS_DIR, f'training_curves_{architecture}.png')
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
    train(architecture='mobilenet_v2')