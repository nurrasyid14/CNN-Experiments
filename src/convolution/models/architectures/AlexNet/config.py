"""Configuration for the AlexNet architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlexNetConfig:
    num_classes: int = 10
    embedding_dim: int = 4096
    in_channels: int = 3
