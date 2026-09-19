"""Data schemas for Procedural Graphs (arXiv:2609.09153)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EdgeAttrs:
    """Φ(e): condition, guidance, pitfalls (paper §3.1)."""

    condition: str = ""
    guidance: str = ""
    pitfalls: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "EdgeAttrs":
        d = d or {}
        return cls(
            condition=str(d.get("condition", "")),
            guidance=str(d.get("guidance", "")),
            pitfalls=str(d.get("pitfalls", "")),
        )


@dataclass
class Edge:
    """Directed attributed triplet (u, r, v)."""

    source: str
    relation: str
    target: str
    attrs: EdgeAttrs = field(default_factory=EdgeAttrs)

    def key(self) -> str:
        return f"{self.source}|{self.relation}|{self.target}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "attrs": self.attrs.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(
            source=str(d["source"]),
            relation=str(d.get("relation", "LEADS_TO")),
            target=str(d["target"]),
            attrs=EdgeAttrs.from_dict(d.get("attrs") or d.get("attributes")),
        )


@dataclass
class Node:
    """Abstract procedure node: tool / reasoning step / state."""

    id: str
    kind: str = "tool"  # tool | reasoning | state
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(
            id=str(d["id"]),
            kind=str(d.get("kind", "tool")),
            description=str(d.get("description", "")),
        )


@dataclass
class GraphEdit:
    """Single edit in ΔG: add/delete node or edge (attribute = delete+re-add)."""

    op: str  # add_node | delete_node | add_edge | delete_edge
    payload: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "payload": self.payload, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdit":
        return cls(
            op=str(d["op"]),
            payload=dict(d.get("payload") or {}),
            rationale=str(d.get("rationale", "")),
        )


@dataclass
class TrajectoryStep:
    action: str
    observation: str
    thought: str = ""
    guidance: str = ""
    active_node: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    task_id: str
    query: str
    success: bool
    score: float
    answer: str
    gold: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    mode: str = "no_graph"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "success": self.success,
            "score": self.score,
            "answer": self.answer,
            "gold": self.gold,
            "steps": [s.to_dict() for s in self.steps],
            "mode": self.mode,
        }
