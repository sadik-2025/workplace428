"""
Phase 4a — Self-supervised pretraining via rotation prediction.

No labels are used here at all — the pretext task is: given an image
rotated by 0, 90, 180, or 270 degrees, predict which rotation was applied.
This forces the encoder to learn features about symbol SHAPE and
STRUCTURE (since it must notice orientation-dependent cues) before it ever
sees a single class label. The trained encoder is then reused (and
fine-tuned) in train_domain_generalization.py.

Usage:
    python pretrain_rotation.py --data_dir ../../data/raw --epochs 8
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import get_or_create_split
from model_ssl import Encoder, RotationHead

IMG_SIZE = 45
ROTATIONS = [0, 90, 180, 270]


class RotationDataset(Dataset):
    """Wraps the (unlabeled, for this purpose) training images: each __getitem__
    picks a random rotation, applies it, and returns (image, rotation_index)."""

    def __init__(self, df, img_size: int = IMG_SIZE):
        self.filepaths = df["filepath"].tolist()
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("L")
        img = self.resize(img)
        rot_idx = random.randint(0, 3)
        angle = ROTATIONS[rot_idx]
        if angle != 0:
            img = img.rotate(angle)
        img = self.to_tensor(img)
        return img, rot_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/raw")
    parser.add_argument("--splits_dir", type=str, default="../../data/processed/splits")
    parser.add_argument("--out_dir", type=str, default="../models/ssl_domain_gen")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only use the TRAIN split's images (no labels used) — keeps val/test
    # completely untouched by pretraining, same as Phase 2/3.
    train_df, _, _, _ = get_or_create_split(data_dir, splits_dir)
    print(f"Pretraining on {len(train_df)} unlabeled images (rotation prediction).")

    dataset = RotationDataset(train_df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    encoder = Encoder(img_size=IMG_SIZE).to(device)
    rot_head = RotationHead(encoder.out_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(rot_head.parameters()), lr=args.lr
    )

    history = {"loss": [], "acc": []}

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        rot_head.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, rot_labels in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            imgs, rot_labels = imgs.to(device), rot_labels.to(device)
            optimizer.zero_grad()
            feats = encoder(imgs)
            outputs = rot_head(feats)
            loss = criterion(outputs, rot_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == rot_labels).sum().item()
            total += imgs.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)
        print(f"Epoch {epoch}/{args.epochs} | rotation_loss={epoch_loss:.4f} rotation_acc={epoch_acc:.4f}")

    torch.save(encoder.state_dict(), out_dir / "encoder_pretrained.pt")
    print(f"\nSaved pretrained encoder to {out_dir / 'encoder_pretrained.pt'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, args.epochs + 1), history["loss"])
    axes[0].set_title("Rotation pretext loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(range(1, args.epochs + 1), history["acc"])
    axes[1].set_title("Rotation pretext accuracy")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "pretrain_curve.png", dpi=150)
    plt.close()

    print(f"Pretraining curve saved to {out_dir / 'pretrain_curve.png'}")


if __name__ == "__main__":
    main()
