from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cnn import CNNFeatureExtractor
from training.data import ManifestDataset, encode_labels, load_manifest
from analysis.report import analyze_experiment


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


def make_loader(
    records: list[dict],
    batch_size: int,
    training: bool,
    image_size: int = 224,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = ManifestDataset(
        records,
        image_size=(image_size, image_size),
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def configure_acceleration(training_config: dict) -> None:
    cudnn_config = training_config.get("cudnn", {})
    torch.backends.cudnn.benchmark = bool(cudnn_config.get("benchmark", False))
    torch.backends.cudnn.deterministic = bool(cudnn_config.get("deterministic", False))


def train_cnn(
    model: CNNFeatureExtractor,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: Path,
    experiment: str,
) -> list[dict]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"loss={total_loss / max(total, 1):.4f} "
            f"accuracy={correct / max(total, 1):.4f}"
        )
        gradient_norms = [
            parameter.grad.detach().norm().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "experiment": experiment,
            },
            checkpoint_dir / "latest_model.pt",
        )

        validation_correct = 0
        validation_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(1)
                validation_correct += (predictions == labels).sum().item()
                validation_total += labels.size(0)
        validation_accuracy = validation_correct / max(validation_total, 1)
        model.train()
        print(f"validation accuracy={validation_accuracy:.4f}")
        history.append({
            "epoch": epoch + 1,
            "loss": float(total_loss / max(total, 1)),
            "accuracy": float(correct / max(total, 1)),
            "validation_accuracy": float(validation_accuracy),
            "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        })
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "experiment": experiment,
                },
                checkpoint_dir / "best_model.pt",
            )
    return history


def extract_features(
    model: CNNFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            embeddings = model.embed(images.to(device)).cpu().numpy()
            features.append(embeddings)
            targets.append(labels.numpy())
    return np.concatenate(features), np.concatenate(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["simplicity", "aerial", "animal", "mnist"], default="simplicity")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    args = parser.parse_args()

    root = ROOT
    datasets = load_config(root / "configs/datasets.yaml")["datasets"]
    training_config = load_config(root / "configs/training.yaml")["training"]
    seed_everything(int(training_config["seed"]))
    configure_acceleration(training_config)

    records = load_manifest(root / datasets[args.experiment]["manifest"])
    records, class_names, _ = encode_labels(records)
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    missing = [name for name, values in split_records.items() if name in ("train", "test") and not values]
    if missing:
        raise ValueError(f"No {args.experiment} records for splits: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    dataloader_config = training_config["dataloader"]
    loader_options = {
        "num_workers": int(dataloader_config["num_workers"]),
        "persistent_workers": bool(dataloader_config["persistent_workers"]),
    }
    model = CNNFeatureExtractor(args.embedding_dim, len(class_names)).to(device)
    checkpoint_dir = root / "checkpoints" / "experiments" / args.experiment
    cnn_history = train_cnn(
        model,
        make_loader(split_records["train"], batch_size, training=True, image_size=args.image_size, **loader_options),
        make_loader(split_records["validation"], batch_size, training=False, image_size=args.image_size, **loader_options),
        device,
        args.epochs,
        float(training_config["learning_rate"]),
        checkpoint_dir,
        args.experiment,
    )
    best_checkpoint = torch.load(
        checkpoint_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_features, train_labels = extract_features(
        model,
        make_loader(split_records["train"], batch_size, training=False, image_size=args.image_size, **loader_options),
        device,
    )
    test_features, test_labels = extract_features(
        model,
        make_loader(split_records["test"], batch_size, training=False, image_size=args.image_size, **loader_options),
        device,
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=1000,
        random_state=int(training_config["seed"]),
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    probabilities = (
        classifier.predict_proba(test_features)
        if hasattr(classifier, "predict_proba")
        else None
    )
    if probabilities is not None:
        aligned_probabilities = np.zeros(
            (probabilities.shape[0], len(class_names)),
            dtype=float,
        )
        for column, class_id in enumerate(classifier.classes_):
            aligned_probabilities[:, int(class_id)] = probabilities[:, column]
        probabilities = aligned_probabilities

    report = classification_report(
        test_labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "experiment": args.experiment,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "classes": class_names,
        "device": str(device),
        "cnn_embedding_dimensions": args.embedding_dim,
        "classification_report": report,
    }
    output_dir = root / "reports" / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "embedding_dim": args.embedding_dim,
        },
        output_dir / "cnn_features.pt",
    )
    joblib.dump(classifier, checkpoint_dir / "sklearn_mlp.joblib")
    analysis = analyze_experiment(
        output_dir,
        args.experiment,
        class_names,
        test_labels,
        predictions,
        probabilities,
        cnn_history,
    )
    metrics["roc_auc"] = analysis["macro_roc_auc"]
    metrics["pr_auc"] = analysis["macro_pr_auc"]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
    print(report)
    print(f"test accuracy: {metrics['accuracy']:.4f}")
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())