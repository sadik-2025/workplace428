"""
Phase 3 — Data Augmentation training script.

Same architecture as the baseline (BaselineCNN), same train/val/test split
(loaded via get_or_create_split, so it's identical to Phase 2's split) —
the only difference is the TRAINING data now goes through augmentation
transforms (rotation, blur, noise, elastic distortion, contrast jitter).

To actually test whether augmentation improves GENERALIZATION (not just
training accuracy), this script evaluates on:
  1. The standard test set (same clean images baseline used)
  2. An "unseen domain" test set — the same test images, but pushed through
     transformations NOT used during training (heavier rotation, inversion,
     salt-and-pepper style noise) to simulate a different visual domain.

Usage:
    python train_augmented.py --data_dir ../../data/raw --epochs 15
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import MathSymbolDataset, get_or_create_split
from model_baseline import BaselineCNN

IMG_SIZE = 45


class AddGaussianNoise:
    """Simple additive Gaussian noise, applied to a tensor after ToTensor()."""

    def __init__(self, std: float = 0.08):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, -1.0, 1.0)


def get_train_transforms():
    """Augmentations used DURING TRAINING — moderate, meant to make the model
    robust to normal handwriting/style variation."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomApply([AddGaussianNoise(std=0.06)], p=0.3),
        ]
    )


def get_clean_transforms():
    """No augmentation — used for the standard val/test set, same as baseline,
    so results are directly comparable."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_unseen_domain_transforms():
    """Transformations NOT used during training — simulates a previously
    unseen visual domain. Heavier rotation, inversion, and stronger noise."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=35),
            transforms.RandomInvert(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            AddGaussianNoise(std=0.15),
        ]
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def measure_inference_time(model, sample_batch, device, n_runs: int = 50):
    model.eval()
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(sample_batch)
        start = time.time()
        for _ in range(n_runs):
            model(sample_batch)
        elapsed = time.time() - start
    return (elapsed / n_runs) * 1000


def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/raw")
    parser.add_argument("--splits_dir", type=str, default="../../data/processed/splits")
    parser.add_argument("--out_dir", type=str, default="../models/augmented")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # SAME split as baseline (cached in splits_dir) -> fair comparison
    train_df, val_df, test_df, class_to_idx = get_or_create_split(data_dir, splits_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Classes ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = MathSymbolDataset(train_df, class_to_idx, transform=get_train_transforms())
    val_ds = MathSymbolDataset(val_df, class_to_idx, transform=get_clean_transforms())
    test_ds = MathSymbolDataset(test_df, class_to_idx, transform=get_clean_transforms())
    unseen_ds = MathSymbolDataset(test_df, class_to_idx, transform=get_unseen_domain_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BaselineCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best val_acc {val_acc:.4f}, saved checkpoint.")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    # Standard test set (clean, same as baseline's test set)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    # Unseen-domain test set (transformations not seen during training)
    unseen_loss, unseen_acc, unseen_preds, unseen_labels = evaluate(
        model, unseen_loader, criterion, device
    )
    u_precision, u_recall, u_f1, _ = precision_recall_fscore_support(
        unseen_labels, unseen_preds, average="macro", zero_division=0
    )

    sample_batch, _ = next(iter(test_loader))
    inference_ms = measure_inference_time(model, sample_batch, device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = model_size_mb(model)

    metrics = {
        "model_name": "Augmented CNN",
        "test_accuracy": test_acc,
        "top1_accuracy": test_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "unseen_domain_accuracy": unseen_acc,
        "unseen_domain_precision_macro": u_precision,
        "unseen_domain_recall_macro": u_recall,
        "unseen_domain_f1_macro": u_f1,
        "generalization_gap": test_acc - unseen_acc,
        "inference_time_ms_per_batch": inference_ms,
        "batch_size_used_for_timing": sample_batch.size(0),
        "num_parameters": n_params,
        "model_size_mb": size_mb,
        "epochs_trained": args.epochs,
        "best_val_accuracy": best_val_acc,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== Standard test set results ===")
    print(f"  accuracy: {test_acc:.4f} | precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")
    print("\n=== Unseen-domain test set results ===")
    print(f"  accuracy: {unseen_acc:.4f} | precision: {u_precision:.4f} | recall: {u_recall:.4f} | f1: {u_f1:.4f}")
    print(f"\nGeneralization gap (standard - unseen): {test_acc - unseen_acc:.4f}")

    # Training curve
    epochs_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="train")
    axes[1].plot(epochs_range, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close()

    # Confusion matrix (standard test set)
    cm = confusion_matrix(labels, preds)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Augmented CNN (standard test set)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Bar chart comparing standard vs unseen-domain accuracy
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Standard test", "Unseen domain"], [test_acc, unseen_acc], color=["#4c72b0", "#dd8452"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Augmented CNN: standard vs unseen-domain accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "domain_robustness.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
