"""Gating & Rollback — accept skill only if validation improves (§3.2.4)."""

from __future__ import annotations

from typing import List, Tuple

from wikiskill.inference import InferenceAgent
from wikiskill.metrics import accuracy
from wikiskill.schemas import GateDecision, SkillProposal, TaskExample, TraceRecord
from wikiskill.skills import SkillStore
from wikiskill.wiki import WikiStore


class GatingController:
    def __init__(
        self,
        skills: SkillStore,
        wiki: WikiStore,
        inference: InferenceAgent,
    ):
        self.skills = skills
        self.wiki = wiki
        self.inference = inference
        self.best_score: float = -1.0

    def initialize_baseline(
        self,
        val_tasks: List[TaskExample],
        *,
        iteration: int = 0,
    ) -> float:
        """R_best ← R(T_val,0) with empty / current skills before loop proposals."""
        traces = self.inference.rollout(val_tasks, iteration=iteration, split="val")
        score = accuracy(traces)
        self.best_score = score
        return score

    def evaluate_and_gate(
        self,
        proposal: SkillProposal,
        val_tasks: List[TaskExample],
        *,
        iteration: int,
    ) -> Tuple[GateDecision, List[TraceRecord]]:
        if proposal.action == "noop" or not proposal.skill_name:
            traces = self.inference.rollout(val_tasks, iteration=iteration, split="val")
            score = accuracy(traces)
            decision = GateDecision(
                accepted=False,
                val_score=score,
                best_score=self.best_score,
                proposal=proposal,
                reason="noop proposal",
            )
            self.wiki.append_skill_impact(
                iteration=iteration,
                skill=proposal.skill_name or "(noop)",
                action="noop",
                val_score=score,
                decision="Rejected",
                notes="noop",
            )
            return decision, traces

        backup_tag = f"pre_iter_{iteration}"
        self.skills.backup(backup_tag)
        self.skills.apply_proposal(proposal)

        traces = self.inference.rollout(val_tasks, iteration=iteration, split="val")
        score = accuracy(traces)
        accepted = score > self.best_score

        if accepted:
            self.best_score = score
            reason = f"val {score:.3f} > best; keep skill"
            decision_label = "Accepted"
        else:
            self.skills.restore(backup_tag)
            reason = f"val {score:.3f} <= best {self.best_score:.3f}; rollback skills"
            decision_label = "Rejected"

        # Wiki persists regardless of accept/reject
        self.wiki.append_skill_impact(
            iteration=iteration,
            skill=proposal.skill_name,
            action=proposal.action,
            val_score=score,
            decision=decision_label,
            notes=reason,
        )
        self.wiki.append_log(
            iteration,
            [
                f"{decision_label}: skill '{proposal.skill_name}' "
                f"(val score = {score:.3f}, best = {self.best_score:.3f})"
            ],
        )

        decision = GateDecision(
            accepted=accepted,
            val_score=score,
            best_score=self.best_score,
            proposal=proposal,
            reason=reason,
        )
        return decision, traces
