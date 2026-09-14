"""Skill Proposer — ReAct-style atomic create/update (§3.2.3)."""

from __future__ import annotations

import re
from typing import Any

from wikiskill.prompts import proposer_prompt
from wikiskill.schemas import SkillProposal
from wikiskill.skills import SkillStore
from wikiskill.traces import RawStore
from wikiskill.wiki import WikiStore


class SkillProposer:
    def __init__(self, llm: Any, wiki: WikiStore, skills: SkillStore, raw: RawStore):
        self.llm = llm
        self.wiki = wiki
        self.skills = skills
        self.raw = raw

    def run(self, *, iteration: int) -> SkillProposal:
        prompt = proposer_prompt(
            wiki_index=self.wiki.read_index(),
            skill_impact=self.wiki.read_skill_impact(),
            outcome_summary=self.raw.outcome_summary(iteration, "train"),
            active_skills=self.skills.list_skills(),
            iteration=iteration,
        )
        # Prefetch top pattern pages (ReAct-lite for transformers / mock)
        extra = []
        for name in self.wiki.list_patterns()[:5]:
            body = self.wiki.read_pattern(name) or ""
            extra.append(f"### patterns/{name}.md\n{body[:800]}\n")
        if extra:
            prompt += "\n\nOn-demand pattern pages (pre-fetched):\n" + "\n".join(extra)

        text = self.llm.generate(
            system="You propose a single atomic skill create/update informed by the wiki.",
            user=prompt,
            role="proposer",
        )
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> SkillProposal:
        def _field(label: str) -> str:
            m = re.search(rf"(?im)^##\s*{label}\s*:\s*(.+)$", text)
            return m.group(1).strip() if m else ""

        action = (_field("ACTION") or "noop").lower()
        if action not in ("create", "update", "noop"):
            action = "noop"
        name = _field("SKILL_NAME")
        rationale = _field("RATIONALE")

        purpose = ""
        skill = ""
        pm = re.search(r"(?im)^##\s*PURPOSE\s*$([\s\S]*?)(?=^##\s|\Z)", text)
        if pm:
            purpose = pm.group(1).strip()
        sm = re.search(r"(?im)^##\s*SKILL\s*$([\s\S]*?)(?=^##\s|\Z)", text)
        if sm:
            skill = sm.group(1).strip()

        patterns = re.findall(r"patterns/([a-zA-Z0-9_-]+)", purpose + "\n" + rationale)
        return SkillProposal(
            action=action,  # type: ignore[arg-type]
            skill_name=name if name.lower() != "none" else "",
            skill_md=skill,
            purpose_md=purpose,
            rationale=rationale,
            target_patterns=patterns,
        )
