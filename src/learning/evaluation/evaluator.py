from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from utils.visualisation import plot_training_curves


def analyze_experiment(
    output_dir: Path,
    experiment: str,
    class_names: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    cnn_history: list[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    class_ids = np.arange(len(class_names))
    matrix = confusion_matrix(targets, predictions, labels=class_ids)
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        xticks_rotation="vertical",
        cmap="Blues",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    auc = {}
    pr_auc = {}
    if probabilities is not None and probabilities.shape[1] == len(class_names):
        binary_targets = np.eye(len(class_names))[targets]
        for class_index, class_name in enumerate(class_names):
            y_true = binary_targets[:, class_index]
            if len(np.unique(y_true)) < 2:
                continue
            y_score = probabilities[:, class_index]
            auc[class_name] = float(
                roc_auc_score(y_true, y_score)
            )
            pr_auc[class_name] = float(
                average_precision_score(y_true, y_score)
            )
            display = RocCurveDisplay.from_predictions(y_true, y_score)
            display.ax_.set_title(f"ROC - {class_name}")
            plt.tight_layout()
            plt.savefig(figures_dir / f"roc_{class_name.replace(' ', '_')}.png", dpi=160)
            plt.close()
            PrecisionRecallDisplay.from_predictions(y_true, y_score)
            plt.title(f"Precision-Recall - {class_name}")
            plt.tight_layout()
            plt.savefig(
                figures_dir / f"pr_{class_name.replace(' ', '_')}.png",
                dpi=160,
            )
            plt.close()

    if any(np.asarray([row.get("gradient_norm", 0.0) for row in cnn_history]) > 0):
        gradient_norms = [row["gradient_norm"] for row in cnn_history]
        plt.plot(range(1, len(gradient_norms) + 1), gradient_norms, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Mean gradient norm")
        plt.title("CNN gradient-descent stability")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "gradient_norms.png", dpi=160)
        plt.close()

    plot_training_curves(figures_dir / "training_curves.png", cnn_history)

    summary = {
        "experiment": experiment,
        "roc_auc_by_class": auc,
        "pr_auc_by_class": pr_auc,
        "macro_roc_auc": float(np.mean(list(auc.values()))) if auc else None,
        "macro_pr_auc": float(np.mean(list(pr_auc.values()))) if pr_auc else None,
        "cnn_history": cnn_history,
        "confusion_matrix": matrix.tolist(),
        "classes": class_names,
        "target_count": int(len(targets)),
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, indent=2)
    return summary
