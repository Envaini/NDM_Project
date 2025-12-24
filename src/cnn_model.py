import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Input:  (B, 1, 64, ~15)  (mel_bins=64, frames≈15)
    Output: logits (B, n_classes)
    Embedding: forward_features(x) -> (B, emb_dim)
    """
    def __init__(self, n_classes: int, emb_dim: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),   # (64,15) -> (32,7)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),   # (32,7) -> (16,7)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))        # -> (64,4,4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, emb_dim)
        )

        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        emb = self.fc(z)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits
