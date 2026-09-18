"""Offline LLM refiner: contrast traces → ΔG + rejection memory (§3.3)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .graph import ProceduralGraph
from .prompts import REFINER_SYSTEM, refiner_user_prompt
from .schemas import EpisodeResult, GraphEdit


class RejectionMemory:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, edits: List[GraphEdit], val_score: float, note: str = "") -> None:
        self.entries.append(
            {
                "edits": [e.to_dict() for e in edits],
                "val_score": val_score,
                "note": note,
            }
        )

    def render(self, max_entries: int = 8) -> str:
        if not self.entries:
            return "(empty)"
        lines = []
        for i, e in enumerate(self.entries[-max_entries:], 1):
            lines.append(f"{i}. val={e['val_score']:.3f} note={e.get('note','')} edits={json.dumps(e['edits'])}")
        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)


class Refiner:
    def __init__(self, llm: Any):
        self.llm = llm

    def propose(
        self,
        graph: ProceduralGraph,
        train_results: List[EpisodeResult],
        rejection: RejectionMemory,
    ) -> List[GraphEdit]:
        ok = [r for r in train_results if r.success]
        bad = [r for r in train_results if not r.success]
        succ = self._summarize(ok) or "(none)"
        fail = self._summarize(bad) or "(none)"
        user = refiner_user_prompt(graph.render_text(), succ, fail, rejection.render())
        raw = self.llm.generate(system=REFINER_SYSTEM, user=user, role="refiner")
        return parse_edits(raw)

    @staticmethod
    def _summarize(results: List[EpisodeResult], limit: int = 6) -> str:
        lines = []
        for r in results[:limit]:
            acts = " -> ".join(s.action for s in r.steps[:8])
            lines.append(
                f"- {r.task_id} score={r.score:.0f} answer={r.answer!r} gold={r.gold!r} traj={acts}"
            )
        return "\n".join(lines)


def parse_edits(raw: str) -> List[GraphEdit]:
    raw = raw.strip()
    # extract JSON array
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    edits: List[GraphEdit] = []
    if not isinstance(data, list):
        return []
    for item in data:
        if not isinstance(item, dict) or "op" not in item:
            continue
        edits.append(GraphEdit.from_dict(item))
    return edits


class ValidationGate:
    """Commit iff S_val(G_cand) >= S_val(G_prev) (§3.3 eq. 5)."""

    @staticmethod
    def decide(prev_score: float, cand_score: float) -> bool:
        return cand_score >= prev_score - 1e-9
