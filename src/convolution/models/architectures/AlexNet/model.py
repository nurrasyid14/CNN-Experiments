"""AlexNet image classifier and feature extractor."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import LocalResponseNormalization
from .config import AlexNetConfig


class AlexNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 4096,
        num_classes: int = 10,
        config: AlexNetConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = AlexNetConfig(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
            )
        if embedding_dim != config.embedding_dim or num_classes != config.num_classes:
            config = AlexNetConfig(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                in_channels=config.in_channels,
            )

        self.config = config
        self.embedding_dim = config.embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNormalization(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNormalization(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.embedding_dim, config.num_classes)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        return self.embedding(features.flatten(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(images))


class AlexNetFeatureExtractor(AlexNet):
    """Descriptive alias for embedding pipelines."""
