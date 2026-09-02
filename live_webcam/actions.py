from config import VOLUME_STEP

VOL_AVAILABLE = False

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOL_AVAILABLE = True
except ImportError:
    print("WARNING: pycaw / comtypes not found.")

def get_volume_interface():
    if not VOL_AVAILABLE:
        return None

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = cast(interface, POINTER(IAudioEndpointVolume))
        _ = vol.GetMasterVolumeLevelScalar()
        print("Volume control : OK (pycaw)")
        return vol
    except Exception as e:
        print(f"Volume control: FAILED ({e})")
        return None

def execute_action(action_key: str, vol_interface, screenshot_dir: str) -> str:
    if vol_interface is None:
        print(f"[ACTION] {action_key} (no volume interface)")
        return action_key.replace("_", " ").title()

    try:
        cur = vol_interface.GetMasterVolumeLevelScalar()
        if action_key == "volume_up":
            new = min(1.0, cur + VOLUME_STEP)
        else:
            new = max(0.0, cur - VOLUME_STEP)
        vol_interface.SetMasterVolumeLevelScalar(new, None)
        return f"Volume {int(new * 100)}%"
    except Exception as e:
        return f"Vol error: {e}"