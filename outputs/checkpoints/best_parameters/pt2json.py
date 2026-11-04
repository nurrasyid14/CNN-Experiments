from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "experiments"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "checkpoints" / "best_parameters"


def encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "values": tensor.tolist(),
    }


def encode_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return encode_tensor(value)
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def convert_checkpoint(checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint: {checkpoint_path}")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")

    encoded_parameters = {
        name: encode_tensor(value) if isinstance(value, torch.Tensor) else encode_value(value)
        for name, value in model_state.items()
    }
    report = {
        "source": str(checkpoint_path.resolve()),
        "experiment": checkpoint.get("experiment", checkpoint_path.parent.name),
        "epoch": checkpoint.get("epoch"),
        "validation_accuracy": checkpoint.get("validation_accuracy"),
        "parameter_count": sum(
            value.numel() for value in model_state.values() if isinstance(value, torch.Tensor)
        ),
        "parameters": encoded_parameters,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def resolve_checkpoint(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode PyTorch best-model checkpoints as JSON parameter reports."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Checkpoint path(s). Repeat to convert multiple files.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Experiment name(s) under checkpoints/experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated JSON reports.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if args.checkpoint:
        checkpoint_paths = [resolve_checkpoint(ROOT, value) for value in args.checkpoint]
    elif args.experiment:
        checkpoint_paths = [
            DEFAULT_CHECKPOINT_DIR / experiment / "best_model.pt"
            for experiment in args.experiment
        ]
    else:
        checkpoint_paths = sorted(DEFAULT_CHECKPOINT_DIR.glob("*/best_model.pt"))

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        output_path = output_dir / f"{checkpoint_path.parent.name}.json"
        report = convert_checkpoint(checkpoint_path, output_path)
        print(
            f"{checkpoint_path.parent.name}: "
            f"{report['parameter_count']} parameters -> {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())