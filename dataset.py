import os
import shutil
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from config import DATA_ROOT, GESTURE_CLASSES

IMAGES_PER_CLASS = 6850

def apply_clahe(img_rgb):
    """Apply CLAHE in LAB color space for better contrast under varied lighting"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def download_dataset(images_per_class=IMAGES_PER_CLASS):
    if os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) == len(GESTURE_CLASSES):
        print(f'Dataset already exists at {DATA_ROOT}')
        return

    print('Streaming HaGRID dataset from HuggingFace...')
    ds = load_dataset(
        "cj-mills/hagrid-classification-512p-no-gesture-150k",
        split="train",
    )

    label_names = ds.features['label'].names
    missing = [c for c in GESTURE_CLASSES if c not in label_names]
    if missing:
        raise ValueError(
            f'These GESTURE_CLASSES classes are not in the dataset: {missing}\n'
            f'Dataset classes are: {label_names}'
        )

    if os.path.exists(DATA_ROOT):
        shutil.rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT)
    for cls in GESTURE_CLASSES:
        os.makedirs(os.path.join(DATA_ROOT, cls), exist_ok=True)

    counts = {cls: 0 for cls in GESTURE_CLASSES}
    total_needed = images_per_class * len(GESTURE_CLASSES)
    print(f'Extracting {images_per_class} images per class ({total_needed} total)...')

    for example in ds:
        gesture = label_names[example['label']]

        if gesture not in GESTURE_CLASSES or counts[gesture] >= images_per_class:
            continue

        img = example['image']  # PIL Image
        out_path = os.path.join(DATA_ROOT, gesture, f'{gesture}_{counts[gesture]:05d}.jpg')
        img.convert('RGB').save(out_path, 'JPEG')
        counts[gesture] += 1

        collected = sum(counts.values())
        if collected % 20 == 0:
            print(f'  ...{collected}/{total_needed} collected')

        if all(c >= images_per_class for c in counts.values()):
            break

    print('\nDataset ready. Image counts:')
    for cls in GESTURE_CLASSES:
        n = len(os.listdir(os.path.join(DATA_ROOT, cls)))
        print(f'  {cls}: {n} images')

def get_data_loaders():
    train_ds = GestureDataset(DATA_ROOT, GESTURE_CLASSES, train_transform, 'train')
    val_ds = GestureDataset(DATA_ROOT, GESTURE_CLASSES, eval_transform, 'val')
    test_ds = GestureDataset(DATA_ROOT, GESTURE_CLASSES, eval_transform, 'test')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

class GestureDataset(Dataset):
    def __init__(self, root, classes, transform=None, split='train',
                 train_ratio=0.76, val_ratio=0.09):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            files = sorted([
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            if split == 'train':
                subset = files[:n_train]
            elif split == 'val':
                subset = files[n_train:n_train + n_val]
            else:
                subset = files[n_train + n_val:]

            self.samples.extend([(f, self.class_to_idx[cls]) for f in subset])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.resize(img, (224, 224))
        img = apply_clahe(img)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label
    
# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])