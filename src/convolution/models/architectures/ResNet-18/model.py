"""ResNet-18 image classifier and feature extractor."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import BasicBlock
from .config import ResNet18Config


class ResNet18(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 10,
        config: ResNet18Config | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = ResNet18Config(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
            )
        if embedding_dim != config.embedding_dim or num_classes != config.num_classes:
            config = ResNet18Config(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                in_channels=config.in_channels,
            )

        self.config = config
        self.embedding_dim = config.embedding_dim

        self.conv1 = nn.Conv2d(
            config.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = (
            nn.Identity()
            if self.embedding_dim == 512
            else nn.Linear(512, self.embedding_dim)
        )
        self.classifier = nn.Linear(self.embedding_dim, config.num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        layers.extend(
            BasicBlock(out_channels, out_channels)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images)
        features = self.maxpool(self.relu(self.bn1(features)))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features).flatten(1)
        return self.embedding(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(images))


class ResNet18FeatureExtractor(ResNet18):
    """Backward-compatible descriptive name for embedding pipelines."""