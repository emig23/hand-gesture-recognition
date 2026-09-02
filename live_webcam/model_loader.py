import torch
import torch.nn as nn
from torchvision import models

def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    architecture = checkpoint.get("architecture", "mobilenet_v2")

    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.last_channel
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[0].in_features
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {architecture}")

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded model  : {checkpoint_path}")
    print(f"Architecture  : {architecture}")
    print(f"Classes       : {classes}")
    if "macro_f1" in checkpoint:
        print(f"Saved F1      : {checkpoint['macro_f1']:.4f}")
    return model, classes

