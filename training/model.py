import torch
import torch.nn as nn
from torchvision import models
from config import GESTURE_CLASSES, DEVICE

NUM_CLASSES = len(GESTURE_CLASSES)

def build_model(architecture='mobilenet_v2', unfreeze_last_n=5, dropout=0.2):
    if architecture == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        feature_blocks = model.features
        in_features = model.last_channel

    elif architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        feature_blocks = model.features
        in_features = model.classifier[0].in_features

    elif architecture == 'mobilenet_v3_large':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        feature_blocks = model.features
        in_features = model.classifier[0].in_features

    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        feature_blocks = model.features
        in_features = model.classifier[1].in_features

    else:
        raise ValueError(f'Unknown architecture: {architecture}')

    # freeze then unfreeze last n feature blocks
    for param in model.parameters():
        param.requires_grad = False
    for param in feature_blocks[-unfreeze_last_n:].parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES)
    )

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{architecture}] Total params: {total_params:,} | Trainable: {trainable:,}')

    return model

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = build_model(architecture=checkpoint['architecture'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f'Model loaded from {model_path}')
    print(f'  Architecture: {checkpoint["architecture"]}')
    print(f'  Macro F1: {checkpoint.get("macro_f1", "N/A")}')
    print(f'  FPS: {checkpoint.get("fps", "N/A")}')

    return model, checkpoint