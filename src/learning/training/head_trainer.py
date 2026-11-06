from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_embedding_head(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> list[dict]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    best_accuracy = -1.0
    best_state = None
    history = []

    validation_tensor = torch.from_numpy(
        validation_features.astype(np.float32, copy=False)
    )
    validation_targets = torch.from_numpy(
        validation_labels.astype(np.int64, copy=False)
    ).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_labels.size(0)
            total += batch_labels.size(0)
            correct += (logits.argmax(1) == batch_labels).sum().item()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor.to(device))
            validation_predictions = validation_logits.argmax(1)
            validation_accuracy = (
                validation_predictions == validation_targets
            ).float().mean().item()
        entry = {
            "epoch": epoch,
            "loss": float(total_loss / max(total, 1)),
            "accuracy": float(correct / max(total, 1)),
            "validation_accuracy": float(validation_accuracy),
        }
        history.append(entry)
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"{model.__class__.__name__} epoch={epoch}/{epochs} "
            f"loss={entry['loss']:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_embedding_head(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = logits.argmax(1).cpu().numpy()
    return predictions.astype(np.int64), probabilities
