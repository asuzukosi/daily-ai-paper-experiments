"""Prompt templates for Inference / Wiki Maintainer / Skill Proposer."""

from __future__ import annotations

from typing import List

from wikiskill.schemas import TraceRecord


def inference_user_prompt(question: str) -> str:
    return (
        f"Solve the following problem.\n\n"
        f"Problem:\n{question}\n\n"
        f"Show brief reasoning, then end with `Final answer: <value>` on its own line."
    )


def maintainer_prompt(
    *,
    wiki_index: str,
    sample_traces: List[TraceRecord],
    iteration: int,
) -> str:
    blocks = []
    for t in sample_traces:
        status = "PASS" if t.correct else "FAIL"
        blocks.append(
            f"### Trace {t.task_id} [{status}]\n"
            f"Q: {t.question}\n"
            f"Prediction: {t.prediction}\n"
            f"Ground truth: {t.ground_truth}\n"
            f"Reasoning: {t.reasoning[:500]}\n"
        )
    return (
        f"You are the Wiki Maintainer for WikiSkill (iteration {iteration}).\n"
        f"Perform root-cause analysis on failing traces and extract successful strategies "
        f"from passing traces. Propose NEW or UPDATED pattern pages.\n\n"
        f"Current wiki index:\n{wiki_index}\n\n"
        f"Sampled traces:\n" + "\n".join(blocks) + "\n"
        f"Respond with markdown sections of the form:\n"
        f"## PATTERN: <slug>\n"
        f"<pattern body>\n"
        f"## LOG\n"
        f"- bullet findings\n"
    )


def proposer_prompt(
    *,
    wiki_index: str,
    skill_impact: str,
    outcome_summary: str,
    active_skills: List[str],
    iteration: int,
) -> str:
    skills_blob = ", ".join(active_skills) if active_skills else "(none)"
    return (
        f"You are the Skill Proposer for WikiSkill (iteration {iteration}).\n"
        f"Operate in a ReAct style: reason about wiki patterns and outcomes, then propose "
        f"ONE atomic skill create or update.\n\n"
        f"Active skills: {skills_blob}\n\n"
        f"Wiki index:\n{wiki_index}\n\n"
        f"Skill impact history:\n{skill_impact}\n\n"
        f"Training outcomes:\n{outcome_summary}\n\n"
        f"Respond with:\n"
        f"## ACTION: create|update|noop\n"
        f"## SKILL_NAME: <slug>\n"
        f"## RATIONALE: <text>\n"
        f"## PURPOSE\n"
        f"<maps skill back to wiki patterns>\n"
        f"## SKILL\n"
        f"<full SKILL.md content>\n"
    )
