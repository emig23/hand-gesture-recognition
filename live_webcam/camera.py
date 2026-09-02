import time
import cv2
import torch

from config import GESTURE_ACTIONS, HOLD_SECONDS, COOLDOWN
from preprocessing import preprocess_frame
from actions import get_volume_interface, execute_action

def run_camera(model, classes, device, camera_index=0,
               conf_threshold=0.5, screenshot_dir="screenshots"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    vol_interface = get_volume_interface()

    hold_gesture = None
    hold_start = 0.0
    last_triggered = {}
    status_text = ""
    status_until = 0.0

    C_GREEN  = (0, 220, 80)
    C_ORANGE = (0, 165, 255)
    C_YELLOW = (0, 215, 255)

    print("\nHold a gesture to trigger its action.")
    print(f"Recognized action: {list(GESTURE_ACTIONS.keys())}")
    print("Press Q to quit.\n")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera disconnected.")
                break

            now = time.perf_counter()

            tensor = preprocess_frame(frame).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            conf, pred_idx = probs.max(0)
            conf = conf.item()
            label = classes[pred_idx.item()]

            is_mapped = label in GESTURE_ACTIONS
            is_confident = conf >= conf_threshold

            if is_mapped and is_confident:

                if hold_gesture == label:
                    held_for = now - hold_start
                    cooldown_ok = (now - last_triggered.get(label, 0)) >= COOLDOWN

                    if held_for >= HOLD_SECONDS and cooldown_ok:
                        action_label, action_key = GESTURE_ACTIONS[label]
                        status_text  = execute_action(action_key, vol_interface, screenshot_dir)
                        status_until = now + 2.5
                        last_triggered[label] = now
                        hold_start = now
                        print(f"[TRIGGERED] {action_label}  ->  {status_text}")
                else:
                    hold_gesture = label
                    hold_start   = now
            else:
                hold_gesture = None

            # Minimal UI 
            color = C_GREEN if is_confident else C_ORANGE
            display = f"{label} ({conf:.0%})" if is_confident else f"{label} (?)"
            cv2.putText(frame, display, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            if is_mapped and is_confident and hold_gesture == label:
                progress = min(1.0, (now - hold_start) / HOLD_SECONDS)
                bar_color = C_GREEN if progress >= 1.0 else C_YELLOW
                cv2.rectangle(frame, (10, 55), (10 + int(200 * progress), 70), bar_color, -1)

            if now < status_until:
                cv2.putText(frame, f">> {status_text}", (12, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2, cv2.LINE_AA)

            cv2.imshow("Hand Gesture Recognition | Q to quit", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")