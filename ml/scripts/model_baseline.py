"""
Simple CNN used as the baseline model (Phase 2) and reused with data
augmentation (Phase 3). Deliberately small/fast so it trains quickly even
on CPU — good enough as a reference point for comparing against the
self-supervised + domain generalization model in Phase 4.
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 45):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 8
        )
        reduced = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
