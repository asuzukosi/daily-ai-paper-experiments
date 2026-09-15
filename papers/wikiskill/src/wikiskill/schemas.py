"""Data schemas for WikiSkill traces, proposals, and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TaskExample:
    id: str
    question: str
    answer: str
    tags: List[str] = field(default_factory=list)
    split: str = "train"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskExample":
        return cls(
            id=str(d["id"]),
            question=str(d["question"]),
            answer=str(d["answer"]).strip(),
            tags=list(d.get("tags") or []),
            split=str(d.get("split") or "train"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRecord:
    """Immutable execution trace written to raw/ (write-once)."""

    task_id: str
    iteration: int
    split: str
    question: str
    prediction: str
    ground_truth: str
    correct: bool
    reasoning: str = ""
    skills_used: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceRecord":
        return cls(
            task_id=str(d["task_id"]),
            iteration=int(d["iteration"]),
            split=str(d["split"]),
            question=str(d["question"]),
            prediction=str(d.get("prediction") or ""),
            ground_truth=str(d.get("ground_truth") or ""),
            correct=bool(d.get("correct")),
            reasoning=str(d.get("reasoning") or ""),
            skills_used=list(d.get("skills_used") or []),
            meta=dict(d.get("meta") or {}),
        )


ProposalAction = Literal["create", "update", "noop"]


@dataclass
class SkillProposal:
    """Atomic skill proposal (one skill create/update per iteration)."""

    action: ProposalAction
    skill_name: str
    skill_md: str
    purpose_md: str
    rationale: str = ""
    target_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillProposal":
        return cls(
            action=d.get("action", "noop"),  # type: ignore[arg-type]
            skill_name=str(d.get("skill_name") or ""),
            skill_md=str(d.get("skill_md") or ""),
            purpose_md=str(d.get("purpose_md") or ""),
            rationale=str(d.get("rationale") or ""),
            target_patterns=list(d.get("target_patterns") or []),
        )


@dataclass
class GateDecision:
    accepted: bool
    val_score: float
    best_score: float
    proposal: SkillProposal
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["proposal"] = self.proposal.to_dict()
        return d


@dataclass
class IterationSummary:
    iteration: int
    train_acc: float
    val_acc: float
    test_acc: Optional[float]
    n_patterns: int
    n_skills: int
    gate_accepted: bool
    proposal_skill: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
