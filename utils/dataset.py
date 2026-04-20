"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for VinDr-CXR PNG images.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED,
)
from utils.image_utils import load_png


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VinDrDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_index: dict,
        split: str = "train",
        transform=None,
        binary_mode: bool = False,
    ):
        self.df          = df.reset_index(drop=True)
        self.image_index = image_index
        self.split       = split
        self.transform   = transform if transform is not None else get_transforms(split)
        self.binary_mode = binary_mode

        self.label_cols  = [c for c in df.columns if c.startswith("label_")]
        self.num_classes = 1 if binary_mode else len(self.label_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = row["image_id"]
        path     = self.image_index.get(image_id)

        if path:
            img = load_png(path)
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        tensor = self.transform(img)
        raw    = row[self.label_cols].values.astype(np.float32)

        if self.binary_mode:
            labels = torch.tensor([float(raw.max())], dtype=torch.float32)
        else:
            labels = torch.tensor(raw, dtype=torch.float32)

        return tensor, labels, image_id


# ─── Feature extraction dataset ───────────────────────────────────────────────

class VinDrFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_index: dict):
        self.df          = df.reset_index(drop=True)
        self.image_index = image_index
        self.transform   = get_transforms("val")
        self.label_cols  = [c for c in df.columns if c.startswith("label_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = row["image_id"]
        path     = self.image_index.get(image_id)
        img      = load_png(path) if path else Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        tensor   = self.transform(img)
        labels   = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return tensor, labels, image_id


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    image_index: dict,
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = PIN_MEMORY,
    binary_mode: bool = False,
):
    train_ds = VinDrDataset(train_df, image_index, split="train", binary_mode=binary_mode)
    val_ds   = VinDrDataset(val_df,   image_index, split="val",   binary_mode=binary_mode)
    test_ds  = VinDrDataset(test_df,  image_index, split="test",  binary_mode=binary_mode)

    # ── Weighted sampler ──────────────────────────────────────────────────────
    label_cols   = [c for c in train_df.columns if c.startswith("label_")]
    label_matrix = train_df[label_cols].values.astype(np.float32)

    if binary_mode:
        binary_labels = label_matrix.max(axis=1)
        pos_rate = binary_labels.mean()
        neg_rate = 1.0 - pos_rate
        weights  = np.where(binary_labels == 1,
                            1.0 / (pos_rate + 1e-6),
                            1.0 / (neg_rate + 1e-6))
    else:
        class_freq = label_matrix.mean(axis=0)
        def sample_weight(row_labels):
            pos_idx = np.where(row_labels > 0)[0]
            return (1.0 / (class_freq[pos_idx].mean() + 1e-6)
                    if len(pos_idx) > 0 else 1.0 / (class_freq.max() + 1e-6))
        weights = np.array([sample_weight(label_matrix[i]) for i in range(len(train_df))])

    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(weights), len(weights), replacement=True
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=g, persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_dl, val_dl, test_dl
