from __future__ import annotations

from collections import Counter
from typing import Iterable


class MetricsCounter:
    """Accumulate scalar metric observations."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._totals: dict[str, float] = {}

    def update(self, metrics: dict[str, float | int]) -> None:
        for name, value in metrics.items():
            numeric_value = float(value)
            self._counts[name] += 1
            self._totals[name] = self._totals.get(name, 0.0) + numeric_value

    def update_many(self, name: str, values: Iterable[float | int]) -> None:
        for value in values:
            self.update({name: value})

    def mean(self, name: str) -> float | None:
        count = self._counts[name]
        return self._totals[name] / count if count else None

    def totals(self) -> dict[str, float]:
        return dict(self._totals)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)
