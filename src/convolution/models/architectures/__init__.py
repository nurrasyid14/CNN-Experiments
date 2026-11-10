"""Model architecture registry and factory for CNN-Experiments."""

from __future__ import annotations

import importlib
from typing import Any
import torch
from torch import nn

_ARCHITECTURES = {
    "repvgg-a0": "convolution.models.architectures.RepVGG-A0.model:RepVGGA0",
    "repvgg_a0": "convolution.models.architectures.RepVGG-A0.model:RepVGGA0",
    "repvgg": "convolution.models.architectures.RepVGG-A0.model:RepVGGA0",
    "resnet-18": "convolution.models.architectures.ResNet-18.model:ResNet18",
    "resnet18": "convolution.models.architectures.ResNet-18.model:ResNet18",
    "alexnet": "convolution.models.architectures.AlexNet.model:AlexNet",
    "cnn": "convolution.models.cnn.classifier:CNNFeatureExtractor",
}


def get_architecture_class(name: str) -> type[nn.Module]:
    canonical = name.strip().lower()
    if canonical not in _ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{name}'. Supported architectures: "
            f"{', '.join(sorted(_ARCHITECTURES.keys()))}"
        )
    module_path, class_name = _ARCHITECTURES[canonical].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_model(
    architecture: str,
    num_classes: int = 10,
    embedding_dim: int = 512,
    in_channels: int = 3,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a neural network model architecture by name."""
    cls = get_architecture_class(architecture)
    try:
        return cls(embedding_dim=embedding_dim, num_classes=num_classes, **kwargs)
    except TypeError:
        return cls(embedding_dim=embedding_dim, num_classes=num_classes)


__all__ = ["build_model", "get_architecture_class", "_ARCHITECTURES"]

