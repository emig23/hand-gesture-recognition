IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

GESTURE_ACTIONS = {
    "like":    ("Like (Volume Up)",      "volume_up"),
    "dislike": ("Dislike (Volume Down)", "volume_down"),
}

HOLD_SECONDS    = 0.5
COOLDOWN        = 1.0
VOLUME_STEP     = 0.05
BRIGHTNESS_STEP = 10

DEFAULT_MODEL_PATH = "../training/content/gesture_model_mobilenet_v2.pth"