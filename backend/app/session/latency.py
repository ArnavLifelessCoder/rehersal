"""Latency instrumentation.

The spec sets a 1.5s turn budget but has no way to know whether it holds. A
budget you don't measure is a wish, so every turn is timed per stage, compared
against the configured budget, and the breakdown is shipped to the client (it
renders in the debug panel) as well as persisted for later analysis.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TurnTimings:
    """Per-stage millisecond timings for one candidate-turn -> interviewer-speech."""

    stages: dict[str, float] = field(default_factory=dict)
    breaches: list[str] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str, budget_ms: float | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.stages[name] = round(elapsed, 1)
            if budget_ms is not None and elapsed > budget_ms:
                self.breaches.append(f"{name}: {elapsed:.0f}ms > {budget_ms:.0f}ms budget")

    def mark(self, name: str) -> None:
        self.stages[name] = round((time.perf_counter() - self._t0) * 1000.0, 1)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 1)

    def to_dict(self, budget_total_ms: float | None = None) -> dict:
        total = self.total_ms
        out = {
            "stages": self.stages,
            "total_ms": total,
            "breaches": self.breaches,
        }
        if budget_total_ms is not None:
            out["budget_ms"] = budget_total_ms
            out["within_budget"] = total <= budget_total_ms
        return out


@dataclass
class LatencyLog:
    """Session-level rollup: p50/p95 across turns, for the report."""

    turns: list[dict] = field(default_factory=list)

    def add(self, timings: dict) -> None:
        self.turns.append(timings)

    def summary(self) -> dict:
        totals = sorted(t["total_ms"] for t in self.turns if "total_ms" in t)
        if not totals:
            return {}
        def pct(p: float) -> float:
            idx = min(int(len(totals) * p), len(totals) - 1)
            return totals[idx]
        return {
            "turns_measured": len(totals),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "max_ms": totals[-1],
            "breach_count": sum(1 for t in self.turns if t.get("breaches")),
        }
