"""Supporting layers used by AlexNet."""

import torch
from torch import nn


class LocalResponseNormalization(nn.Module):
    def __init__(
        self,
        size: int = 5,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 2.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squared = inputs * inputs
        channels = inputs.shape[1]
        scale = inputs.new_zeros(inputs.shape)

        half_size = self.size // 2
        for offset in range(-half_size, half_size + 1):
            start = max(0, -offset)
            end = min(channels, channels - offset)
            source_start = max(0, offset)
            source_end = min(channels, channels + offset)
            scale[:, start:end] += squared[:, source_start:source_end]

        scale = self.k + (self.alpha / self.size) * scale
        return inputs / torch.pow(scale, self.beta)
