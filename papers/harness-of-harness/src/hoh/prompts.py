"""Role prompt templates aligned with paper Appendix (Planner / Developer / QA)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


PLANNER_SYSTEM = """You are the Project Planner for an iterative software-development run (Harness-of-Harness).
You MUST produce a bounded, locally complete development plan for ONE iteration.
Balance repair of known gaps with a small new capability (capability growth).
You may INSPECT the artifact (read-only) but MUST NOT modify production code.
Output MUST satisfy the DevelopmentPlan JSON schema (or equivalent markdown with the same fields).
"""

DEVELOPER_SYSTEM = """You are the Developer for Harness-of-Harness.
You are the ONLY role allowed to modify the artifact (single-writer).
Implement the development plan as a small, verifiable increment.
Embed focused implementation-time tests, then stop — independent QA evaluates separately.
Return a unified diff / file patches that apply cleanly to the current artifact.
"""

QA_SYSTEM = """You are the QA Tester for Harness-of-Harness.
You receive a FROZEN, read-only candidate artifact. You MUST NOT modify it.
Evaluate against the specification S and the current development plan D_t, plus Runtime.check results.
Use complementary black-box and white-box criteria. Record gaps instead of inventing success.
Output MUST be EvidenceReport JSON with: iteration, passed, criteria_results, gaps, regressions, runtime_check, notes.
"""


def build_planner_prompt(
    *,
    loop_index: int,
    spec: str,
    evidence: Optional[Dict[str, Any]],
    artifact_index: str,
) -> str:
    evidence_block = (
        "(empty — first iteration)"
        if not evidence
        else f"```json\n{_safe_json(evidence)}\n```"
    )
    return f"""{PLANNER_SYSTEM}

You are the Project Planner for iteration {loop_index} of an iterative software-development run.

## Specification S (fixed)
{spec}

## Evidence E_{{t-1}}
{evidence_block}

## Artifact index (read-only progressive disclosure)
{artifact_index}

## Project Planner Priorities
1. Address outstanding gaps / failures from evidence when present.
2. Deliver ONE small, concrete new capability that is locally complete and testable.
3. Explicitly list behaviors that must be preserved.
4. Write checkable acceptance criteria derived from S and the chosen increment.

## Required output
Return either:
(A) a JSON object with fields: iteration, objective, scope[], preserve[], acceptance_criteria[], rationale, repair_vs_growth
or (B) markdown with the same sections (Objective / Scope / Preserve / Acceptance criteria / Rationale).
"""


def build_developer_prompt(
    *,
    loop_index: int,
    spec: str,
    plan: Dict[str, Any],
    artifact_listing: str,
) -> str:
    return f"""{DEVELOPER_SYSTEM}

Iteration: {loop_index}

## Specification S
{spec}

## Development plan D_t
```json
{_safe_json(plan)}
```

## Current artifact (you may modify)
{artifact_listing}

## Required output
Return patches as one or more blocks:
```patch path=RELATIVE/PATH
... unified diff or full file contents if creating ...
```
Only modify files needed for this increment. Prefer reuse over recreation.
"""


def build_qa_prompt(
    *,
    loop_index: int,
    spec: str,
    plan: Dict[str, Any],
    artifact_listing: str,
    runtime_check: Dict[str, Any],
) -> str:
    return f"""{QA_SYSTEM}

Iteration: {loop_index}

## Specification S
{spec}

## Development plan D_t
```json
{_safe_json(plan)}
```

## Candidate artifact (FROZEN, read-only)
{artifact_listing}

## Runtime.check(A_t)
```json
{_safe_json(runtime_check)}
```

## Required output
JSON EvidenceReport:
{{
  "iteration": {loop_index},
  "passed": true|false,
  "criteria_results": [{{"criterion": "...", "status": "pass|fail|insufficient", "detail": "..."}}],
  "gaps": ["..."],
  "regressions": ["..."],
  "runtime_check": {{...}},
  "notes": "..."
}}
"""


def _safe_json(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


def progressive_index(paths: List[str], *, max_show: int = 40) -> str:
    """Concise categorized index — progressive disclosure of artifacts."""
    if not paths:
        return "(empty artifact)"
    shown = paths[:max_show]
    lines = [f"- {p}" for p in shown]
    if len(paths) > max_show:
        lines.append(f"... and {len(paths) - max_show} more (request by path if needed)")
    return "\n".join(lines)
