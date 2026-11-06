from __future__ import annotations

import json
from pathlib import Path


class MonitoringLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_history.jsonl"
        self.history_path.write_text("", encoding="utf-8")

    def log_epoch(self, entry: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(entry) + "\n")
