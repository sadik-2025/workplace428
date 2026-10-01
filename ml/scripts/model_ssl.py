"""
Phase 4 building blocks — shared by pretrain_rotation.py and
train_domain_generalization.py.

Architecture choices (kept deliberately close to the Phase 2/3 BaselineCNN
so parameter counts / inference times are genuinely comparable):
  - Encoder: same 3-conv-block backbone as BaselineCNN, but exposed as a
    reusable feature extractor (no final classifier baked in).
  - RotationHead: 4-way classifier (0/90/180/270 degrees) used ONLY during
    self-supervised pretraining — thrown away afterwards.
  - ClassifierHead: the real symbol classifier, trained on labeled data.
  - DomainDiscriminator + GradReverse: implements a DANN-style
    domain-adversarial setup. The discriminator tries to guess which
    synthetic "domain" (augmentation family) a feature vector came from;
    the gradient reversal layer flips the sign of its gradient on the way
    back into the encoder, so the encoder is pushed to produce features
    that DON'T reveal the domain — i.e. domain-invariant features.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class Encoder(nn.Module):
    def __init__(self, img_size: int = 45):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced = img_size // 8
        self.out_dim = 128 * reduced * reduced

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class RotationHead(nn.Module):
    """4-way rotation classifier (0/90/180/270) — self-supervised pretext task."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.fc(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Predicts which synthetic domain a feature vector came from.
    Trained adversarially via gradient reversal against the encoder."""

    def __init__(self, in_dim: int, num_domains: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_domains),
        )

    def forward(self, x, lambd: float = 1.0):
        x = grad_reverse(x, lambd)
        return self.fc(x)
