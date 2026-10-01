"""
Shared dataset utilities for the WEX 428 handwritten math symbol project.

Used by train_baseline.py, train_augmented.py, and later scripts so the
train/val/test split and class list stay IDENTICAL across all three model
approaches — this is required for a fair comparison later.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_image_dataset(data_dir: Path) -> pd.DataFrame:
    records = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for f in class_dir.glob("*"):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                records.append({"filepath": str(f), "label": label})
    if not records:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return pd.DataFrame(records)


def get_or_create_split(data_dir: Path, splits_dir: Path, seed: int = 42):
    """Returns (train_df, val_df, test_df, class_to_idx).

    The split is computed ONCE and cached to splits_dir/split.csv +
    class_to_idx.json, so every training script (baseline, augmented,
    self-supervised) trains/evaluates on the exact same split — required
    for a fair Model Comparison dashboard later.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_csv = splits_dir / "split.csv"
    classes_json = splits_dir / "class_to_idx.json"

    if split_csv.exists() and classes_json.exists():
        df = pd.read_csv(split_csv)
        with open(classes_json) as f:
            class_to_idx = json.load(f)
    else:
        df = load_image_dataset(data_dir)
        classes = sorted(df["label"].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}

        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df["label"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
        )
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        df.to_csv(split_csv, index=False)
        with open(classes_json, "w") as f:
            json.dump(class_to_idx, f, indent=2)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, class_to_idx


class MathSymbolDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label
