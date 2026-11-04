from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    output_path: Path,
    history: list[dict],
) -> None:
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    accuracies = [row["accuracy"] for row in history]
    validation_accuracies = [row["validation_accuracy"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set(title="CNN training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, accuracies, marker="o", label="training")
    axes[1].plot(epochs, validation_accuracies, marker="o", label="validation")
    axes[1].set(title="CNN accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
