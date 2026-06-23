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