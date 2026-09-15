"""Constrained OUTPUT schemas for HoH roles (paper §3.4).

HoH constrains verifiable outputs, not agent workflows. Schema failures
trigger retries; agents remain free over reasoning / tool use.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DevelopmentPlan:
    """Planner output D_t — bounded, locally complete increment."""

    iteration: int
    objective: str
    scope: List[str]
    preserve: List[str]
    acceptance_criteria: List[str]
    rationale: str = ""
    repair_vs_growth: str = "balanced"  # repair | growth | balanced

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceReport:
    """QA Tester output E_t — structured evidence bundle."""

    iteration: int
    passed: bool
    criteria_results: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    runtime_check: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PLAN_REQUIRED = ("iteration", "objective", "scope", "acceptance_criteria")
EVIDENCE_REQUIRED = ("iteration", "passed", "criteria_results")


def validate_plan(data: Dict[str, Any]) -> Tuple[bool, str, Optional[DevelopmentPlan]]:
    """Validate planner JSON / dict against DevelopmentPlan schema."""
    if not isinstance(data, dict):
        return False, "plan must be a dict/object", None
    missing = [k for k in PLAN_REQUIRED if k not in data]
    if missing:
        return False, f"missing fields: {missing}", None
    if not isinstance(data["objective"], str) or not data["objective"].strip():
        return False, "objective must be a non-empty string", None
    if not isinstance(data["scope"], list) or not data["scope"]:
        return False, "scope must be a non-empty list", None
    if not isinstance(data["acceptance_criteria"], list) or not data["acceptance_criteria"]:
        return False, "acceptance_criteria must be a non-empty list", None
    try:
        plan = DevelopmentPlan(
            iteration=int(data["iteration"]),
            objective=str(data["objective"]).strip(),
            scope=[str(x) for x in data["scope"]],
            preserve=[str(x) for x in data.get("preserve", [])],
            acceptance_criteria=[str(x) for x in data["acceptance_criteria"]],
            rationale=str(data.get("rationale", "")),
            repair_vs_growth=str(data.get("repair_vs_growth", "balanced")),
        )
    except (TypeError, ValueError) as e:
        return False, f"coerce error: {e}", None
    return True, "ok", plan


def validate_evidence(data: Dict[str, Any]) -> Tuple[bool, str, Optional[EvidenceReport]]:
    """Validate QA JSON against EvidenceReport schema."""
    if not isinstance(data, dict):
        return False, "evidence must be a dict/object", None
    missing = [k for k in EVIDENCE_REQUIRED if k not in data]
    if missing:
        return False, f"missing fields: {missing}", None
    if not isinstance(data["passed"], bool):
        return False, "passed must be boolean", None
    if not isinstance(data["criteria_results"], list):
        return False, "criteria_results must be a list", None
    try:
        report = EvidenceReport(
            iteration=int(data["iteration"]),
            passed=bool(data["passed"]),
            criteria_results=list(data["criteria_results"]),
            gaps=[str(x) for x in data.get("gaps", [])],
            regressions=[str(x) for x in data.get("regressions", [])],
            runtime_check=dict(data.get("runtime_check") or {}),
            notes=str(data.get("notes", "")),
        )
    except (TypeError, ValueError) as e:
        return False, f"coerce error: {e}", None
    return True, "ok", report


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull first JSON object from model text (fenced or raw)."""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try whole text / first {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def plan_to_markdown(plan: DevelopmentPlan) -> str:
    """Human-readable planner markdown (also accepted as planner output form)."""
    lines = [
        f"# Development Plan — iteration {plan.iteration}",
        "",
        f"**Objective:** {plan.objective}",
        "",
        f"**Repair vs growth:** {plan.repair_vs_growth}",
        "",
        "## Scope",
    ]
    for s in plan.scope:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Preserve")
    if plan.preserve:
        for p in plan.preserve:
            lines.append(f"- {p}")
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append("## Acceptance criteria")
    for c in plan.acceptance_criteria:
        lines.append(f"- {c}")
    if plan.rationale:
        lines += ["", "## Rationale", plan.rationale]
    return "\n".join(lines) + "\n"


def parse_plan_markdown(text: str, iteration: int) -> Optional[Dict[str, Any]]:
    """Best-effort parse of planner markdown into a plan dict."""
    # Prefer embedded JSON if present
    obj = extract_json_object(text)
    if obj and "objective" in obj:
        obj.setdefault("iteration", iteration)
        return obj

    objective = ""
    repair = "balanced"
    scope: List[str] = []
    preserve: List[str] = []
    criteria: List[str] = []
    rationale = ""
    section = None
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("**objective:**"):
            objective = s.split(":", 1)[1].strip().strip("*").strip()
            continue
        if low.startswith("**repair vs growth:**"):
            repair = s.split(":", 1)[1].strip().strip("*").strip().lower()
            continue
        if s.startswith("## "):
            title = s[3:].strip().lower()
            if "scope" in title:
                section = "scope"
            elif "preserve" in title:
                section = "preserve"
            elif "acceptance" in title or "criteria" in title:
                section = "criteria"
            elif "rationale" in title:
                section = "rationale"
            else:
                section = None
            continue
        if s.startswith("- ") and section in ("scope", "preserve", "criteria"):
            item = s[2:].strip()
            if item.lower() in ("(none yet)", "none", ""):
                continue
            if section == "scope":
                scope.append(item)
            elif section == "preserve":
                preserve.append(item)
            else:
                criteria.append(item)
        elif section == "rationale" and s:
            rationale += s + " "

    if not objective or not scope or not criteria:
        return None
    return {
        "iteration": iteration,
        "objective": objective,
        "scope": scope,
        "preserve": preserve,
        "acceptance_criteria": criteria,
        "rationale": rationale.strip(),
        "repair_vs_growth": repair or "balanced",
    }
