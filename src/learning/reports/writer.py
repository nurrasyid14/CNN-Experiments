from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_manifest(
    output_dir: Path,
    experiment: str,
    run_id: str,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": experiment, "run_id": run_id, **payload}
    path = output_dir / "run.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
