IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

GESTURE_ACTIONS = {
    "two_up": ("Two Up (Volume Up)", "volume_up"),
    "two_up_inverted": ("Two Up Inv. (Volume Down)", "volume_down"),
    "mute": ("Mute (Toggle Mute)", "mute"),
    "rock": ("Rock (Play/Pause)", "play_pause"),
    "peace": ("Peace (Next Track)", "next_track"),
    "peace_inverted": ("Peace Inv. (Prev. Track)", "prev_track"),
    "like": ("Like (Brightness Up)", "brightness_up"),
    "dislike": ("Dislike (Brightness Down)", "brightness_down"),
    "one": ("One (Open Google)", "open_browser"),
    "fist": ("Fist (Minimize Window)", "minimize_window")
}

HOLD_SECONDS    = 0.5
COOLDOWN        = 1.0
VOLUME_STEP     = 0.05
BRIGHTNESS_STEP = 10

DEFAULT_MODEL_PATH = "../training/content/gesture_model_mobilenet_v2.pth"