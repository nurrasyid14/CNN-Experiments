"""Compatibility entry point for the CNN training pipeline."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "pipeline" / "runners" / "run_training.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    runpy.run_path(str(RUNNER), run_name="__main__")
