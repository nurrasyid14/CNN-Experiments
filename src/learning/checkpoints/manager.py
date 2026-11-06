from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _json_value(value: torch.Tensor) -> list[object] | float:
    tensor = value.detach().cpu()
    if tensor.numel() == 1:
        return float(tensor.item())
    return tensor.tolist()


def decode_state_dict(
    checkpoint_path: Path,
    metadata: dict[str, Any] | None = None,
    json_path: Path | None = None,
    text_path: Path | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    decoded = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "values": _json_value(tensor),
        }
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
    }
    document = {"metadata": metadata or {}, "parameters": decoded}
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    if text_path is not None:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "CNN learning artifact",
            json.dumps(metadata or {}, sort_keys=True),
        ]
        lines.extend(
            f"{name}\tshape={shape['shape']}\tdtype={shape['dtype']}"
            for name, shape in decoded.items()
        )
        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return document


def save_torch_artifact(
    output_dir: Path,
    stem: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{stem}.pt"
    torch.save(payload, checkpoint_path)
    decode_state_dict(
        checkpoint_path,
        metadata,
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.txt",
    )
    return checkpoint_path
