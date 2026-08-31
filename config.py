import torch

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = './content/hagrid_sample'
MODEL_PATH = './content/gesture_model.pth'
PLOTS_DIR = './content/plots'

# Dataset
GESTURE_CLASSES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute',
    'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock',
    'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted'
]

# Live Cam
GESTURE_ACTIONS = {
    "like":    ("Like (Volume Up)",      "volume_up"),
    "dislike": ("Dislike (Volume Down)", "volume_down"),
    "ok":      ("Ok (Play/Pause)",       "play_pause"),
    "call":    ("Call (Screenshot)",     "screenshot"),
    "peace":   ("Peace (Prev Track)",    "prev_track"),
}

HOLD_SECONDS    = 0.5
COOLDOWN        = 1.0
VOLUME_STEP     = 0.05
BRIGHTNESS_STEP = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Data
IMAGES_PER_CLASS = 6850

# Training hyperparams
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5
LOG_EVERY_N_BATCHES = 200
ARCHITECTURE = 'mobilenet_v2'

# Evaluation
N_LATENCY_FRAMES = 1000
TARGET_F1 = 0.90
TARGET_LATENCY_MS = 100
TARGET_FPS = 20