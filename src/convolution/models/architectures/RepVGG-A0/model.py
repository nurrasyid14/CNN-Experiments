"""RepVGG-A0 image classifier and feature extractor."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import RepVGGBlock
from .config import RepVGGA0Config


class RepVGGA0(nn.Module):
    """RepVGG-A0 architecture implementation.

    Stages:
        Stage 0: 1 layer, stride 2, channels = 48
        Stage 1: 2 layers, stride 2 (first block), channels = 48
        Stage 2: 4 layers, stride 2 (first block), channels = 96
        Stage 3: 14 layers, stride 2 (first block), channels = 192
        Stage 4: 1 layer, stride 2, channels = 1280
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        num_classes: int = 10,
        config: RepVGGA0Config | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = RepVGGA0Config(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
            )
        if embedding_dim != config.embedding_dim or num_classes != config.num_classes:
            config = RepVGGA0Config(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                in_channels=config.in_channels,
                deploy=config.deploy,
            )

        self.config = config
        self.embedding_dim = config.embedding_dim
        self.deploy = config.deploy

        # RepVGG-A0 multipliers: a = 0.75, b = 2.5
        # Base channels: [64, 64, 128, 256, 512]
        # Multiplied: [48, 48, 96, 192, 1280]
        # Number of blocks: [1, 2, 4, 14, 1]
        num_blocks = [1, 2, 4, 14, 1]
        channels = [48, 48, 96, 192, 1280]

        self.stage0 = RepVGGBlock(
            in_channels=config.in_channels,
            out_channels=channels[0],
            stride=2,
            deploy=self.deploy,
        )

        self.stage1 = self._make_stage(
            in_channels=channels[0],
            out_channels=channels[1],
            num_blocks=num_blocks[1],
            stride=2,
        )
        self.stage2 = self._make_stage(
            in_channels=channels[1],
            out_channels=channels[2],
            num_blocks=num_blocks[2],
            stride=2,
        )
        self.stage3 = self._make_stage(
            in_channels=channels[2],
            out_channels=channels[3],
            num_blocks=num_blocks[3],
            stride=2,
        )
        self.stage4 = self._make_stage(
            in_channels=channels[3],
            out_channels=channels[4],
            num_blocks=num_blocks[4],
            stride=2,
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = (
            nn.Identity()
            if self.embedding_dim == channels[4]
            else nn.Linear(channels[4], self.embedding_dim)
        )
        self.dropout = nn.Dropout(p=getattr(config, "dropout", 0.2))
        self.classifier = nn.Linear(self.embedding_dim, config.num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=s,
                    deploy=self.deploy,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def switch_to_deploy(self) -> None:
        """Switch all blocks in the model to single 3x3 conv deployment mode."""
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        self.deploy = True

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        out = self.stage0(images)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out).flatten(1)
        return self.embedding(out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.embed(images)
        return self.classifier(self.dropout(features))


# Aliases
RepVGG_A0 = RepVGGA0


class RepVGGA0FeatureExtractor(RepVGGA0):
    """Backward-compatible descriptive name for embedding pipelines."""


RepVGG_A0_FeatureExtractor = RepVGGA0FeatureExtractor

