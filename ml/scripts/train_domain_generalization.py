"""
Phase 4b — Domain-Adversarial fine-tuning (DANN-style), built on top of the
self-supervised rotation-pretrained encoder from pretrain_rotation.py.

How this differs from Phase 3 (plain augmentation):
  Phase 3 just threw augmented images at the model and hoped it generalized.
  Here, each training image is randomly assigned to one of several SYNTHETIC
  "domains" (clean / inverted+blur / heavy-rotation+noise / contrast-jitter).
  A domain discriminator tries to guess which domain a feature vector came
  from; a gradient-reversal layer (GRL) flips that signal on the way back
  into the encoder, so the encoder is explicitly pushed to produce features
  that look the SAME regardless of domain — i.e. domain-invariant.

Evaluated the same way as Phase 3 (standard test set + unseen-domain test
set) so the three models are directly comparable.

Usage:
    python train_domain_generalization.py --data_dir ../../data/raw --epochs 12
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import MathSymbolDataset, get_or_create_split
from model_ssl import ClassifierHead, DomainDiscriminator, Encoder

IMG_SIZE = 45
DOMAIN_NAMES = ["clean", "inverted_blur", "heavy_rotation_noise", "contrast_jitter"]
NUM_DOMAINS = len(DOMAIN_NAMES)


class AddGaussianNoise:
    def __init__(self, std: float = 0.08):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, -1.0, 1.0)


def get_clean_transform():
    return transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )


def get_unseen_domain_transform():
    """Same recipe as Phase 3's unseen-domain test — kept identical so the
    three models' unseen-domain scores are directly comparable."""
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


class MultiDomainDataset(Dataset):
    """Each __getitem__ randomly assigns ONE of NUM_DOMAINS synthetic domain
    transforms to that sample, and returns (image, class_label, domain_label)."""

    def __init__(self, df, class_to_idx, img_size: int = IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    def __len__(self):
        return len(self.df)

    def _apply_domain(self, img, domain_idx):
        if domain_idx == 0:  # clean
            return img
        elif domain_idx == 1:  # inverted + mild blur
            from PIL import ImageOps, ImageFilter
            img = ImageOps.invert(img)
            if random.random() < 0.6:
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
            return img
        elif domain_idx == 2:  # heavy rotation + noise (applied post-tensor)
            angle = random.uniform(-25, 25)
            img = img.rotate(angle)
            return img
        else:  # contrast jitter
            from torchvision.transforms import functional as F
            img = F.adjust_contrast(img, random.uniform(0.4, 1.8))
            img = F.adjust_brightness(img, random.uniform(0.6, 1.4))
            return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("L")
        img = self.resize(img)

        domain_idx = random.randint(0, NUM_DOMAINS - 1)
        img = self._apply_domain(img, domain_idx)
        tensor = self.to_tensor_norm(img)

        if domain_idx == 2:  # add extra noise for the "heavy" domain
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.1, -1.0, 1.0)

        label = self.class_to_idx[row["label"]]
        return tensor, label, domain_idx


def measure_inference_time(encoder, classifier, sample_batch, device, n_runs: int = 50):
    encoder.eval()
    classifier.eval()
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        for _ in range(5):
            classifier(encoder(sample_batch))
        start = time.time()
        for _ in range(n_runs):
            classifier(encoder(sample_batch))
        elapsed = time.time() - start
    return (elapsed / n_runs) * 1000


def params_and_size(modules):
    n_params = sum(p.numel() for m in modules for p in m.parameters())
    param_bytes = sum(p.numel() * p.element_size() for m in modules for p in m.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for m in modules for b in m.buffers())
    return n_params, (param_bytes + buffer_bytes) / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/raw")
    parser.add_argument("--splits_dir", type=str, default="../../data/processed/splits")
    parser.add_argument("--out_dir", type=str, default="../models/ssl_domain_gen")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default="../models/ssl_domain_gen/encoder_pretrained.pt",
        help="Path to weights from pretrain_rotation.py. If missing, trains from scratch.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df, class_to_idx = get_or_create_split(data_dir, splits_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Classes ({num_classes}): {list(class_to_idx.keys())}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = MultiDomainDataset(train_df, class_to_idx)
    val_ds = MathSymbolDataset(val_df, class_to_idx, transform=get_clean_transform())
    test_ds = MathSymbolDataset(test_df, class_to_idx, transform=get_clean_transform())
    unseen_ds = MathSymbolDataset(test_df, class_to_idx, transform=get_unseen_domain_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = Encoder(img_size=IMG_SIZE).to(device)
    pretrained_path = Path(args.pretrained_encoder)
    if pretrained_path.exists():
        encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded self-supervised pretrained encoder from {pretrained_path}")
    else:
        print("WARNING: pretrained encoder not found — training encoder from scratch. "
              "Run pretrain_rotation.py first for the intended Phase 4 pipeline.")

    classifier = ClassifierHead(encoder.out_dim, num_classes).to(device)
    domain_disc = DomainDiscriminator(encoder.out_dim, NUM_DOMAINS).to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()) + list(domain_disc.parameters()),
        lr=args.lr,
    )

    history = {"train_class_acc": [], "train_domain_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        domain_disc.train()

        # DANN-style lambda schedule: ramps 0 -> 1 over training so early
        # epochs focus on learning the symbols before the adversarial
        # domain-invariance pressure kicks in hard.
        p = epoch / args.epochs
        lambd = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

        class_correct, domain_correct, total = 0, 0, 0
        for imgs, labels, domains in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            imgs, labels, domains = imgs.to(device), labels.to(device), domains.to(device)

            optimizer.zero_grad()
            feats = encoder(imgs)
            class_logits = classifier(feats)
            domain_logits = domain_disc(feats, lambd=lambd)

            loss = class_criterion(class_logits, labels) + domain_criterion(domain_logits, domains)
            loss.backward()
            optimizer.step()

            class_correct += (class_logits.argmax(dim=1) == labels).sum().item()
            domain_correct += (domain_logits.argmax(dim=1) == domains).sum().item()
            total += imgs.size(0)

        train_class_acc = class_correct / total
        train_domain_acc = domain_correct / total

        # Validation (clean images, classifier only)
        encoder.eval()
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = classifier(encoder(imgs)).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        history["train_class_acc"].append(train_class_acc)
        history["train_domain_acc"].append(train_domain_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | lambda={lambd:.3f} | "
            f"train_class_acc={train_class_acc:.4f} train_domain_acc={train_domain_acc:.4f} "
            f"(lower domain acc = more domain-invariant) | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), out_dir / "best_encoder.pt")
            torch.save(classifier.state_dict(), out_dir / "best_classifier.pt")
            print(f"  -> New best val_acc {val_acc:.4f}, saved checkpoint.")

    # Reload best checkpoint
    encoder.load_state_dict(torch.load(out_dir / "best_encoder.pt", map_location=device))
    classifier.load_state_dict(torch.load(out_dir / "best_classifier.pt", map_location=device))
    encoder.eval()
    classifier.eval()

    @torch.no_grad()
    def eval_on(loader):
        all_preds, all_labels = [], []
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = classifier(encoder(imgs)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return np.array(all_preds), np.array(all_labels)

    preds, labels = eval_on(test_loader)
    test_acc = (preds == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

    unseen_preds, unseen_labels = eval_on(unseen_loader)
    unseen_acc = (unseen_preds == unseen_labels).mean()
    u_precision, u_recall, u_f1, _ = precision_recall_fscore_support(
        unseen_labels, unseen_preds, average="macro", zero_division=0
    )

    sample_batch, _ = next(iter(test_loader))
    inference_ms = measure_inference_time(encoder, classifier, sample_batch, device)
    n_params, size_mb = params_and_size([encoder, classifier])  # domain_disc excluded (inference doesn't use it)

    metrics = {
        "model_name": "SSL + Domain-Adversarial CNN",
        "test_accuracy": float(test_acc),
        "top1_accuracy": float(test_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "unseen_domain_accuracy": float(unseen_acc),
        "unseen_domain_precision_macro": float(u_precision),
        "unseen_domain_recall_macro": float(u_recall),
        "unseen_domain_f1_macro": float(u_f1),
        "generalization_gap": float(test_acc - unseen_acc),
        "inference_time_ms_per_batch": inference_ms,
        "batch_size_used_for_timing": sample_batch.size(0),
        "num_parameters": n_params,
        "model_size_mb": size_mb,
        "epochs_trained": args.epochs,
        "best_val_accuracy": best_val_acc,
        "used_pretrained_ssl_encoder": pretrained_path.exists(),
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
    axes[0].plot(epochs_range, history["train_class_acc"], label="train class acc")
    axes[0].plot(epochs_range, history["val_acc"], label="val class acc")
    axes[0].set_title("Classification accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_domain_acc"], color="orange")
    axes[1].set_title("Domain discriminator accuracy (lower = more domain-invariant)")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
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
    ax.set_title("Confusion Matrix — SSL + Domain-Adversarial CNN")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Domain robustness bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Standard test", "Unseen domain"], [test_acc, unseen_acc], color=["#4c72b0", "#dd8452"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("SSL + Domain-Adversarial CNN: standard vs unseen-domain")
    plt.tight_layout()
    plt.savefig(out_dir / "domain_robustness.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
