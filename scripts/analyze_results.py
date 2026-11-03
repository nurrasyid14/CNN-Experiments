from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report import analyze_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["simplicity", "aerial", "animal", "mnist"], required=True)
    args = parser.parse_args()

    root = ROOT
    print(
        "Existing report analysis is refreshed automatically after training. "
        f"See {root / 'reports' / args.experiment / 'analysis.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
