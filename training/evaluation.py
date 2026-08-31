import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from dataset import get_data_loaders, apply_clahe, eval_transform
from model import load_model

from config import (
    MODEL_PATH, GESTURE_CLASSES, DEVICE, PLOTS_DIR, ARCHITECTURE,
    N_LATENCY_FRAMES, TARGET_F1, TARGET_LATENCY_MS, TARGET_FPS)

def evaluate_model(architecture=ARCHITECTURE):
    model_path = MODEL_PATH.replace('.pth', f'_{architecture}.pth')

    # load model
    model, checkpoint = load_model(model_path)
    classes = checkpoint.get('classes', GESTURE_CLASSES)

    # load test data
    _, _, test_loader, _, _, test_ds = get_data_loaders()


    # test evaluation
    print('\n--- Test Evaluation ---')
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'Test Macro F1: {macro_f1:.4f}')
    print(f'Target:        0.9000  ->  {"PASS" if macro_f1 >= 0.9 else "NEAR TARGET"}')
    print()
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    # Confusion matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix — Test Set ({architecture})')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{architecture}.png')
    plt.savefig(cm_path, dpi=150)
    print(f'Confusion matrix saved to {cm_path}')

    # Latency benchmark
    print('\n--- Latency Benchmark ---')
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    for _ in range(N_LATENCY_FRAMES)]

    latencies = []
    with torch.no_grad():
        for frame in dummy_frames:
            t0 = time.perf_counter()
            img = apply_clahe(frame)
            img = Image.fromarray(img)
            tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
            _ = model(tensor).argmax(1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    mean_lat = latencies.mean()
    p95_lat = np.percentile(latencies, 95)
    fps = 1000 / mean_lat

    print(f'Mean Latency:   {mean_lat:.2f} ms  (target < 100 ms)  -> {"PASS" if mean_lat < 100 else "FAIL"}')
    print(f'P95 Latency:    {p95_lat:.2f} ms')
    print(f'Throughput:     {fps:.1f} FPS  (target >= 20 FPS)  -> {"PASS" if fps >= 20 else "FAIL"}')

    # Latency plot
    plt.figure(figsize=(8, 4))
    plt.hist(latencies, bins=40, color='steelblue', edgecolor='white')
    plt.axvline(mean_lat, color='orange', linestyle='--', label=f'Mean: {mean_lat:.1f} ms')
    plt.axvline(p95_lat, color='red', linestyle='--', label=f'P95: {p95_lat:.1f} ms')
    plt.axvline(100, color='black', linestyle=':', label='Target: 100 ms')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Count')
    plt.title(f'End-to-End Inference Latency Distribution ({N_LATENCY_FRAMES} frames, {architecture})')
    plt.legend()
    plt.tight_layout()
    lat_path = os.path.join(PLOTS_DIR, f'latency_distribution_{architecture}.png')
    plt.savefig(lat_path, dpi=150)
    print(f'Latency plot saved to {lat_path}')

    # Update saved model with benchmark results
    checkpoint['macro_f1'] = macro_f1
    checkpoint['mean_latency_ms'] = mean_lat
    checkpoint['fps'] = fps
    torch.save(checkpoint, model_path)
    print(f'\nUpdated {model_path} with benchmark metrics')

    # Summary
    print('\n' + '=' * 50)
    print('         RESULTS SUMMARY')
    print('=' * 50)
    print(f'  Test Macro F1 : {macro_f1:.4f}  (target >= 0.90)')
    print(f'  Mean Latency  : {mean_lat:.2f} ms  (target < 100)')
    print(f'  P95 Latency   : {p95_lat:.2f} ms')
    print(f'  Throughput    : {fps:.1f} FPS  (target >= 20)')
    print('=' * 50)


if __name__ == '__main__':
    evaluate_model()