"""Configuration for the RepVGG-A0 architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RepVGGA0Config:
    num_classes: int = 10
    embedding_dim: int = 1280
    in_channels: int = 3
    deploy: bool = False
    dropout: float = 0.4

