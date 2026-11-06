from __future__ import annotations

import torch
from torch import nn


class CNNLSTMHead(nn.Module):
    """LSTM classification head for CNN embeddings represented as sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim % sequence_length != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by "
                f"sequence_length ({sequence_length})"
            )
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            input_size=input_dim // sequence_length,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequences = embeddings.reshape(
            embeddings.shape[0],
            self.sequence_length,
            embeddings.shape[1] // self.sequence_length,
        )
        _, (hidden, _) = self.lstm(sequences)
        return self.classifier(hidden[-1])
