import torch.nn as nn
from torchvision import models
from config import GESTURE_CLASSES, DEVICE

NUM_CLASSES = len(GESTURE_CLASSES)

def build_model():
    """Build MobileNetV2 with transfer learning for gesture classification"""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # Freeze all layers, then unfreeze the last feature blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,} | Trainable: {trainable:,}')

    return model


def load_model(model_path):
    """Load a saved model checkpoint"""
    import torch
    model = models.mobilenet_v2(weights=None)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f'Model loaded from {model_path}')
    print(f'  Macro F1: {checkpoint.get("macro_f1", "N/A")}')
    print(f'  FPS: {checkpoint.get("fps", "N/A")}')

    return model, checkpoint