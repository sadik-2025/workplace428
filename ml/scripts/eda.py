"""
Exploratory Data Analysis for the Handwritten Math Symbols dataset.

Usage:
    python eda.py --data_dir ../../data/raw --out_dir ../../docs/eda_output

Expected input layout (typical Kaggle "Handwritten Math Symbols" structure):
    data/raw/
        0/
            img001.png
            img002.png
            ...
        1/
            ...
        +/
            ...
        -/
            ...
        ...

If your downloaded dataset has a different layout (e.g. a single CSV with
pixel columns, or a flat folder with a labels.csv), adjust `load_image_dataset`
below accordingly — the rest of the script only needs a DataFrame with
columns ["filepath", "label"].
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_image_dataset(data_dir: Path) -> pd.DataFrame:
    """Walk a directory of class-named subfolders and build a filepath/label table."""
    records = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for f in class_dir.glob("*"):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                records.append({"filepath": str(f), "label": label})
    if not records:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Check the dataset layout and "
            "update load_image_dataset() if it doesn't match class-per-folder."
        )
    return pd.DataFrame(records)


def class_distribution(df: pd.DataFrame, out_dir: Path):
    counts = df["label"].value_counts().sort_index()
    print("\n=== Class distribution ===")
    print(counts)
    print(f"\nTotal images: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print(f"Min class count: {counts.min()} | Max class count: {counts.max()}")

    plt.figure(figsize=(12, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Symbol class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()

    counts.to_csv(out_dir / "class_counts.csv")


def image_size_stats(df: pd.DataFrame, sample_size: int = 500):
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    sizes = []
    modes = []
    for fp in sample["filepath"]:
        with Image.open(fp) as img:
            sizes.append(img.size)  # (width, height)
            modes.append(img.mode)

    widths, heights = zip(*sizes)
    print("\n=== Image size / mode stats (sample) ===")
    print(f"Width  -> min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
    print(f"Height -> min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
    print(f"Modes seen: {set(modes)}")

    if len(set(sizes)) > 1:
        print(
            "NOTE: images are not all the same size — you'll need a resize step "
            "in your preprocessing pipeline (e.g. resize to 32x32 or 45x45)."
        )
    else:
        print(f"All sampled images share the same size: {sizes[0]}")


def show_sample_grid(df: pd.DataFrame, out_dir: Path, n_per_class: int = 3, max_classes: int = 12):
    classes = sorted(df["label"].unique())[:max_classes]
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 1.5, len(classes) * 1.5))
    for i, cls in enumerate(classes):
        subset = df[df["label"] == cls].sample(min(n_per_class, len(df[df["label"] == cls])), random_state=1)
        for j, fp in enumerate(subset["filepath"]):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            img = Image.open(fp)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, rotation=0, labelpad=20, fontsize=8)
    plt.suptitle("Sample images per class")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_grid.png", dpi=150)
    plt.close()


def train_val_test_split_preview(df: pd.DataFrame):
    """Just prints proposed split sizes — doesn't write files, since actual
    splitting should happen inside your training script (with stratification)."""
    n = len(df)
    print("\n=== Suggested split (stratified) ===")
    print(f"Train (70%): ~{int(n * 0.7)} images")
    print(f"Val   (15%): ~{int(n * 0.15)} images")
    print(f"Test  (15%): ~{int(n * 0.15)} images")
    print(
        "Use sklearn.model_selection.train_test_split with stratify=df['label'] "
        "for both splits to keep class balance."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/raw")
    parser.add_argument("--out_dir", type=str, default="../../docs/eda_output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_image_dataset(data_dir)
    df.to_csv(out_dir / "dataset_index.csv", index=False)

    class_distribution(df, out_dir)
    image_size_stats(df)
    show_sample_grid(df, out_dir)
    train_val_test_split_preview(df)

    print(f"\nAll EDA artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
