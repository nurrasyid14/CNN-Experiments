"""Configuration for the ResNet-18 architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResNet18Config:
    num_classes: int = 10
    embedding_dim: int = 512
    in_channels: int = 3