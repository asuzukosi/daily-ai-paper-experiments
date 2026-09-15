"""Loop metrics — pass rates, relative gain helpers (ProgramBench-style)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationMetrics:
    iteration: int
    plan_ok: bool
    developer_ok: bool
    qa_passed: bool
    runtime_ok: bool
    n_criteria: int = 0
    n_criteria_pass: int = 0
    n_gaps: int = 0
    schema_retries: int = 0
    notes: str = ""

    @property
    def criteria_pass_rate(self) -> float:
        if self.n_criteria <= 0:
            return 0.0
        return 100.0 * self.n_criteria_pass / self.n_criteria

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["criteria_pass_rate"] = self.criteria_pass_rate
        return d


@dataclass
class RunMetrics:
    iterations: List[IterationMetrics] = field(default_factory=list)
    vanilla_pass_rate: Optional[float] = None  # baseline if measured

    def add(self, m: IterationMetrics) -> None:
        self.iterations.append(m)

    @property
    def final_pass_rate(self) -> float:
        if not self.iterations:
            return 0.0
        return self.iterations[-1].criteria_pass_rate

    def relative_gain_pct(self, baseline: Optional[float] = None) -> Optional[float]:
        """Relative gain vs baseline (paper reports avg 52.25% across harnesses)."""
        base = baseline if baseline is not None else self.vanilla_pass_rate
        if base is None or base == 0:
            return None
        return 100.0 * (self.final_pass_rate - base) / base

    def summary(self) -> Dict[str, Any]:
        return {
            "n_iterations": len(self.iterations),
            "final_pass_rate": self.final_pass_rate,
            "vanilla_pass_rate": self.vanilla_pass_rate,
            "relative_gain_pct": self.relative_gain_pct(),
            "iterations": [m.to_dict() for m in self.iterations],
        }


def paper_headline_results() -> Dict[str, Any]:
    """Cite paper Table 1 numbers exactly (for docs / demos)."""
    return {
        "avg_relative_gain_pct": 52.25,
        "max_relative_gain_pct": 82.86,
        "gamecraft_overall": {
            "Codex": {"vanilla": 49.58, "hoh3": 71.52, "delta": 21.93},
            "OpenCode": {"vanilla": 26.90, "hoh3": 48.98, "delta": 22.08},
            "Pi": {"vanilla": 42.16, "hoh3": 58.78, "delta": 16.62},
        },
        "frontierswe_dominance": {
            "Codex": {"vanilla": 44, "hoh3": 71, "delta": 27},
            "OpenCode": {"vanilla": 25, "hoh3": 44, "delta": 19},
            "Pi": {"vanilla": 35, "hoh3": 64, "delta": 29},
        },
        "programbench_pass_rate": {
            "Codex": {"vanilla": 60.41, "hoh3": 66.50},
            "OpenCode": {"vanilla": 45.27, "hoh3": 57.56},
            "Pi": {"vanilla": 35.83, "hoh3": 52.68},
        },
        "multi_day": "70+ iterations FPS game Fusepoint",
        "harness_model_pairs": [
            "Codex+GPT-5.5",
            "OpenCode+DeepSeek-V4-Pro",
            "Pi+MiniMax-M3",
        ],
    }
