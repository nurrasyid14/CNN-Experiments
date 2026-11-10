from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from convolution.models.architectures import build_model
from convolution.models.cnn.classifier import CNNFeatureExtractor
from learning.checkpoints.manager import save_torch_artifact
from learning.evaluation.evaluator import analyze_experiment
from learning.models.lstm import CNNLSTMHead
from learning.models.mlp import CNNMLPHead
from learning.models.rnn import CNNRNNHead
from learning.monitoring.logger import MonitoringLogger
from learning.reports.writer import write_run_manifest
from learning.training.head_trainer import predict_embedding_head, train_embedding_head
from preprocessing.python.dataloader import ManifestDataset, encode_labels, load_manifest
from utils.metrics import (
    benchmark_inference,
    compare_model_execution,
    classification_metrics,
    count_model_parameters,
    estimate_model_flops,
    peak_gpu_memory_mib,
    reset_peak_gpu_memory,
)


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
    model: nn.Module,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: Path,
    experiment: str,
    weight_decay: float = 0.05,
) -> list[dict]:
    model.train()
    reset_peak_gpu_memory(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    best_accuracy = -1.0
    history = []
    for epoch in range(epochs):
        epoch_started = perf_counter()
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
        scheduler.step()
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
                "architecture": getattr(model, "__class__", type(model)).__name__,
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
        epoch_elapsed = perf_counter() - epoch_started
        model.train()
        print(f"validation accuracy={validation_accuracy:.4f}")
        history.append({
            "epoch": epoch + 1,
            "loss": float(total_loss / max(total, 1)),
            "accuracy": float(correct / max(total, 1)),
            "validation_accuracy": float(validation_accuracy),
            "gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "training_images_per_second": float(total / max(epoch_elapsed, 1e-9)),
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
                    "architecture": getattr(model, "__class__", type(model)).__name__,
                },
                checkpoint_dir / "best_model.pt",
            )
    return history


def extract_features(
    model: nn.Module,
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
    parser.add_argument("--architecture", "--model", dest="architecture", default="repvgg-a0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=1280)
    parser.add_argument("--head", choices=["mlp", "lstm", "rnn"], default="mlp")
    parser.add_argument("--run-id", type=str, default=None, help="Custom ID; defaults to <YYYYMMDD_HHMMSS>_<dataset>")
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

    # Each experiment run is labelled with datetime followed with dataset name as its ID
    run_id = args.run_id if args.run_id else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.experiment}"
    print(f"=== Starting Experiment: {run_id} (Architecture: {args.architecture}, Device: {device}) ===")

    model = build_model(
        args.architecture,
        num_classes=len(class_names),
        embedding_dim=args.embedding_dim,
    ).to(device)

    # Directories for checkpoints and outputs
    checkpoint_dir = root / "outputs" / "checkpoints" / args.experiment / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = root / "experiments" / "results" / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir = root / "outputs" / "reports" / args.experiment / run_id
    report_output_dir.mkdir(parents=True, exist_ok=True)

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
    validation_features, validation_labels = extract_features(
        model,
        make_loader(split_records["validation"], batch_size, training=False, image_size=args.image_size, **loader_options),
        device,
    )

    if args.head == "lstm":
        sequence_length = 8 if args.embedding_dim % 8 == 0 else 1
        classifier = CNNLSTMHead(
            args.embedding_dim,
            len(class_names),
            sequence_length=sequence_length,
        )
    elif args.head == "rnn":
        sequence_length = 8 if args.embedding_dim % 8 == 0 else 1
        classifier = CNNRNNHead(
            args.embedding_dim,
            len(class_names),
            sequence_length=sequence_length,
        )
    else:
        classifier = CNNMLPHead(args.embedding_dim, len(class_names))

    head_history = train_embedding_head(
        classifier,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        device,
        args.epochs,
        float(training_config["learning_rate"]),
        batch_size=batch_size,
    )
    monitoring_dir = root / "outputs" / "monitoring" / args.experiment / run_id
    monitoring = MonitoringLogger(monitoring_dir)
    for entry in head_history:
        monitoring.log_epoch(entry)
    predictions, probabilities = predict_embedding_head(
        classifier,
        test_features,
        device,
    )

    report = classification_report(
        test_labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "experiment": args.experiment,
        "run_id": run_id,
        "architecture": args.architecture,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "classes": class_names,
        "device": str(device),
        "cnn_embedding_dimensions": args.embedding_dim,
        "classification_report": report,
        "classification_metrics": classification_metrics(
            test_labels,
            predictions,
            class_names,
        ),
        "model": {
            "architecture": args.architecture,
            "parameters": count_model_parameters(model),
            "flops_per_inference": estimate_model_flops(
                model,
                torch.zeros(1, 3, args.image_size, args.image_size, device=device),
            ),
            "peak_vram_mib": peak_gpu_memory_mib(device),
            "inference": benchmark_inference(
                model,
                torch.zeros(1, 3, args.image_size, args.image_size, device=device),
            ),
            "execution_device": str(device),
            "execution_comparison": compare_model_execution(
                model,
                (1, 3, args.image_size, args.image_size),
            ),
        },
        "training": {
            "epochs": [
                {
                    "epoch": entry["epoch"],
                    "images_per_second": entry["training_images_per_second"],
                    "loss": entry["loss"],
                    "accuracy": entry["accuracy"],
                    "validation_accuracy": entry["validation_accuracy"],
                }
                for entry in cnn_history
            ],
            "peak_vram_mib": peak_gpu_memory_mib(device),
        },
    }

    # Features bundle
    features_payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "embedding_dim": args.embedding_dim,
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "validation_features": validation_features,
        "validation_labels": validation_labels,
    }

    # Save outputs into both experiments/results/<run_id> and outputs/reports/<experiment>/<run_id>
    for target_dir in (experiment_dir, report_output_dir):
        # 1. Best model pt file
        shutil.copy2(checkpoint_dir / "best_model.pt", target_dir / "best_model.pt")
        if (checkpoint_dir / "latest_model.pt").is_file():
            shutil.copy2(checkpoint_dir / "latest_model.pt", target_dir / "latest_model.pt")

        # 2. Features pt file (both feats.pt and cnn_features.pt)
        torch.save(features_payload, target_dir / "feats.pt")
        torch.save(features_payload, target_dir / "cnn_features.pt")

    save_torch_artifact(
        checkpoint_dir,
        f"best_{args.head}",
        {
            "model_state_dict": classifier.state_dict(),
            "head": args.head,
            "class_names": class_names,
            "embedding_dim": args.embedding_dim,
        },
        {"experiment": args.experiment, "run_id": run_id, "head": args.head},
    )

    # 3. Generate figures
    analysis = analyze_experiment(
        experiment_dir,
        args.experiment,
        class_names,
        test_labels,
        predictions,
        probabilities,
        cnn_history,
    )
    # Mirror figures to report_output_dir as well
    if (experiment_dir / "figures").is_dir():
        shutil.copytree(experiment_dir / "figures", report_output_dir / "figures", dirs_exist_ok=True)

    metrics["roc_auc"] = analysis.get("macro_roc_auc")
    metrics["pr_auc"] = analysis.get("macro_pr_auc")

    # 4. Decode metrics into json and log
    for target_dir in (experiment_dir, report_output_dir):
        with (target_dir / "metrics.json").open("w", encoding="utf-8") as target:
            json.dump(metrics, target, indent=2)

        # Write human-readable metrics.log
        with (target_dir / "metrics.log").open("w", encoding="utf-8") as log_file:
            log_file.write(f"=====================================================\n")
            log_file.write(f"Experiment Run ID : {run_id}\n")
            log_file.write(f"Dataset           : {args.experiment}\n")
            log_file.write(f"Architecture      : {args.architecture}\n")
            log_file.write(f"Head Classifier   : {args.head}\n")
            log_file.write(f"Execution Device  : {device}\n")
            log_file.write(f"Timestamp         : {datetime.now().isoformat()}\n")
            log_file.write(f"Classes ({len(class_names)}): {', '.join(class_names)}\n")
            log_file.write(f"=====================================================\n\n")

            log_file.write("--- Model Architecture & Complexity ---\n")
            params_info = metrics['model']['parameters']
            if isinstance(params_info, dict):
                total_params = params_info.get("total", 0)
                trainable_params = params_info.get("trainable", 0)
                param_str = f"{total_params:,} (trainable: {trainable_params:,})"
            else:
                param_str = f"{params_info:,}"
            log_file.write(f"Total Parameters    : {param_str}\n")
            log_file.write(f"Estimated FLOPs     : {metrics['model']['flops_per_inference']}\n")
            log_file.write(f"Peak VRAM Usage     : {metrics['model']['peak_vram_mib']:.2f} MiB\n")
            log_file.write(f"Embedding Dimension : {args.embedding_dim}\n\n")

            log_file.write("--- Classification Report ---\n")
            log_file.write(f"{report}\n\n")

            log_file.write("--- Summary Evaluation ---\n")
            log_file.write(f"Test Accuracy       : {metrics['accuracy']:.4f}\n")
            if metrics.get("roc_auc") is not None:
                log_file.write(f"Macro ROC-AUC       : {metrics['roc_auc']:.4f}\n")
            if metrics.get("pr_auc") is not None:
                log_file.write(f"Macro PR-AUC        : {metrics['pr_auc']:.4f}\n")
            log_file.write("\n--- Training History ---\n")
            for entry in cnn_history:
                log_file.write(
                    f"Epoch {entry['epoch']:02d}/{args.epochs}: "
                    f"Loss={entry['loss']:.4f} | "
                    f"Train Acc={entry['accuracy']:.4f} | "
                    f"Val Acc={entry['validation_accuracy']:.4f} | "
                    f"Grad Norm={entry['gradient_norm']:.4f} | "
                    f"Throughput={entry['training_images_per_second']:.1f} img/s\n"
                )
            log_file.write("\n--- Generated Artifacts ---\n")
            log_file.write(f"Best Weights       : {target_dir / 'best_model.pt'}\n")
            log_file.write(f"Features           : {target_dir / 'feats.pt'}\n")
            log_file.write(f"Decoded Metrics    : {target_dir / 'metrics.json'}\n")
            log_file.write(f"Figures Directory  : {target_dir / 'figures'}\n")

    write_run_manifest(
        report_output_dir,
        args.experiment,
        run_id,
        {
            "head": args.head,
            "architecture": args.architecture,
            "metrics": metrics,
            "monitoring_dir": str(monitoring_dir.relative_to(root)),
        },
    )

    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETE: {run_id}")
    print("=" * 50)
    print(report)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Best model pt : {experiment_dir / 'best_model.pt'}")
    print(f"Features pt   : {experiment_dir / 'feats.pt'}")
    print(f"Metrics json  : {experiment_dir / 'metrics.json'}")
    print(f"Metrics log   : {experiment_dir / 'metrics.log'}")
    print(f"Figures dir   : {experiment_dir / 'figures'}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
