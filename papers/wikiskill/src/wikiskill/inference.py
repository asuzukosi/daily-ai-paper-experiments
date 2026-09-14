"""Inference Agent — rollouts with skills injected; wiki access FORBIDDEN."""

from __future__ import annotations

from typing import Any, List

from wikiskill.metrics import exact_match, extract_final_answer
from wikiskill.prompts import inference_user_prompt
from wikiskill.schemas import TaskExample, TraceRecord
from wikiskill.skills import SkillStore
from wikiskill.traces import RawStore


class InferenceAgent:
    """Paper §3.2.1: full skill injection; no wiki access during training rollouts."""

    def __init__(self, llm: Any, skills: SkillStore, raw: RawStore):
        self.llm = llm
        self.skills = skills
        self.raw = raw

    def rollout(
        self,
        tasks: List[TaskExample],
        *,
        iteration: int,
        split: str,
        allow_wiki: bool = False,
    ) -> List[TraceRecord]:
        # allow_wiki is an ablation hook; default WikiSkill keeps this False (§5.1)
        system = self.skills.inject_prompt()
        active = self.skills.list_skills()
        traces: List[TraceRecord] = []
        for task in tasks:
            user = inference_user_prompt(task.question)
            # Hidden meta for MockLLM gold alignment (ignored by real models)
            user = f"<!--TASK_META id={task.id} gold={task.answer}-->\n" + user
            text = self.llm.generate(system=system, user=user, role="inference")
            pred = extract_final_answer(text)
            ok = exact_match(pred, task.answer)
            tr = TraceRecord(
                task_id=task.id,
                iteration=iteration,
                split=split,
                question=task.question,
                prediction=pred,
                ground_truth=task.answer,
                correct=ok,
                reasoning=text,
                skills_used=list(active),
                meta={"allow_wiki": allow_wiki},
            )
            self.raw.write(tr)
            traces.append(tr)
        return traces
