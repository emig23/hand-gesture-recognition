import argparse
import torch

from config import DEFAULT_MODEL_PATH
from model_loader import load_model
from camera import run_camera

def main():
    parser = argparse.ArgumentParser(
        description="Live hand gesture recognition with system controls")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--screenshots", default="screenshots")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    model, classes = load_model(args.model, device)
    run_camera(model, classes, device,
               camera_index=args.camera,
               conf_threshold=args.conf,
               screenshot_dir=args.screenshots)


if __name__ == "__main__":
    main()