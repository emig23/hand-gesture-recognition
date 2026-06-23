import os
import shutil
import zipfile
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from huggingface_hub import hf_hub_download

from config import DATA_ROOT, MODEL_PATH

def apply_clahe(img_rgb):
    """Apply CLAHE in LAB color space for better contrast under varied lighting"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def download_dataset():
    """Download and extract the HaGRID 2,000-image subset from HuggingFace"""
    if os.path.exists(DATA_ROOT) and len(os.listdir(DATA_ROOT)) == len(MODEL_PATH):
        print(f'Dataset already exists at {DATA_ROOT}')
        return

    print('Downloading HaGRID dataset...')
    zip_path = hf_hub_download(
        repo_id="GestureDetectionConnoisseurs/hagrid_subsets",
        filename="hagrid-export_2000_images.zip",
        repo_type="dataset",
        local_dir="./data/hf_cache"
    )

    # Clear old data
    if os.path.exists(DATA_ROOT):
        shutil.rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT)

    print('Extracting 18 classes...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            parts = member.split('/')
            if len(parts) < 2:
                continue
            gesture = parts[0]
            filename = parts[1]

            if gesture not in MODEL_PATH:
                continue
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            out_dir = os.path.join(DATA_ROOT, gesture)
            os.makedirs(out_dir, exist_ok=True)

            with z.open(member) as src:
                with open(os.path.join(out_dir, filename), 'wb') as dst:
                    dst.write(src.read())

    print('\nDataset ready. Image counts:')
    for cls in MODEL_PATH:
        n = len(os.listdir(os.path.join(DATA_ROOT, cls)))
        print(f'  {cls}: {n} images')

def get_data_loaders():
    """Create train, validation, and test data loaders"""
    train_ds = GestureDataset(DATA_ROOT, MODEL_PATH, train_transform, 'train')
    val_ds = GestureDataset(DATA_ROOT, MODEL_PATH, eval_transform, 'val')
    test_ds = GestureDataset(DATA_ROOT, MODEL_PATH, eval_transform, 'test')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

class GestureDataset(Dataset):
    """Hand gesture dataset with CLAHE preprocessing and train/val/test splits"""

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
