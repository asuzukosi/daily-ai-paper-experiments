"""Accuracy helpers + paper Table 1 citations (do not invent numbers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from wikiskill.schemas import IterationSummary, TraceRecord


def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    for wrap in ("$", "`", '"', "'"):
        if s.startswith(wrap) and s.endswith(wrap) and len(s) >= 2:
            s = s[1:-1].strip()
    s = " ".join(s.split())
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s.lower()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def extract_final_answer(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
        if low.startswith("answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1] if lines else ""


def accuracy(traces: List[TraceRecord]) -> float:
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.correct) / len(traces)


@dataclass
class RunMetrics:
    iterations: List[IterationSummary] = field(default_factory=list)
    final_test_acc: Optional[float] = None
    accepted: int = 0
    rejected: int = 0

    def add(self, summary: IterationSummary) -> None:
        self.iterations.append(summary)
        if summary.gate_accepted:
            self.accepted += 1
        else:
            self.rejected += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": [i.to_dict() for i in self.iterations],
            "final_test_acc": self.final_test_acc,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


def paper_headline_results() -> Dict[str, Any]:
    """Cite Table 1 averages and key text numbers only — no invented values."""
    return {
        "table1_avg": {
            "Qwen-3.5-4B": {"no_skill": 26.2, "wikiskill": 38.5},
            "Qwen-3.5-9B": {"no_skill": 29.9, "wikiskill": 47.4},
            "Qwen-3.6-27B": {"no_skill": 39.4, "wikiskill": 63.3},
            "Gemma-4-31B": {"no_skill": 41.3, "wikiskill": 54.9},
            "Gemini-3.5-Flash": {"no_skill": 49.5, "wikiskill": 68.1},
        },
        "qwen_relative_gains_points": {"4B": 12.3, "9B": 17.5, "27B": 23.9},
        "gemini_highlights": {
            "LiveMath": {"no_skill": 33.0, "wikiskill": 72.6},
            "SpreadSheet": {"no_skill": 50.5, "wikiskill": 76.6},
        },
        "transfer_alfworld": {
            "note": "Qwen-3.5-9B with 27B-evolved skill 70.2% vs self-evolved 63.4% on ALFWorld",
            "self_evolved_9b": 63.4,
            "with_27b_skills": 70.2,
        },
        "ablation": (
            "Persistent wiki is critical (Skill Proposer wiki access +15.0 Avg "
            "when Inference has no wiki)."
        ),
        "arxiv": "2608.27454",
    }



