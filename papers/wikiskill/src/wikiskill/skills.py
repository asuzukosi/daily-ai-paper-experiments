"""Skills Layer — reversible procedural instructions (SKILL.md + PURPOSE.md)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

from wikiskill.schemas import SkillProposal


class SkillStore:
    """Active skill set under skills/.

    Each skill is a directory:
      skills/<name>/SKILL.md
      skills/<name>/PURPOSE.md
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._backup_dir = self.root.parent / ".skill_backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self) -> List[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "SKILL.md").exists()
        )

    def n_skills(self) -> int:
        return len(self.list_skills())

    def skill_dir(self, name: str) -> Path:
        return self.root / name

    def read_skill(self, name: str) -> Optional[str]:
        p = self.skill_dir(name) / "SKILL.md"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def read_purpose(self, name: str) -> Optional[str]:
        p = self.skill_dir(name) / "PURPOSE.md"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def inject_prompt(self) -> str:
        """Full skill injection for the Inference Agent system prompt."""
        skills = self.list_skills()
        if not skills:
            return (
                "You are a careful reasoning agent. No specialized skills are currently active.\n"
                "Solve the problem step by step and put the final answer on the last line as "
                "`Final answer: <value>`.\n"
            )
        parts = [
            "You are a careful reasoning agent. The following procedural skills are ACTIVE "
            "and MUST be followed:\n"
        ]
        for name in skills:
            body = self.read_skill(name) or ""
            parts.append(f"\n### Skill: {name}\n{body.strip()}\n")
        parts.append("\nAlways end with `Final answer: <value>` on its own line.\n")
        return "".join(parts)

    def apply_proposal(self, proposal: SkillProposal) -> Path:
        if proposal.action == "noop" or not proposal.skill_name:
            return self.root
        d = self.skill_dir(proposal.skill_name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(proposal.skill_md.strip() + "\n", encoding="utf-8")
        (d / "PURPOSE.md").write_text(proposal.purpose_md.strip() + "\n", encoding="utf-8")
        return d

    def backup(self, tag: str) -> Path:
        dest = self._backup_dir / tag
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name in self.list_skills():
            shutil.copytree(self.skill_dir(name), dest / name)
        (dest / "_meta.json").write_text(
            json.dumps({"skills": self.list_skills(), "tag": tag}, indent=2),
            encoding="utf-8",
        )
        return dest

    def restore(self, tag: str) -> None:
        src = self._backup_dir / tag
        if not src.exists():
            raise FileNotFoundError(f"No skill backup tagged {tag!r}")
        for name in list(self.list_skills()):
            shutil.rmtree(self.skill_dir(name))
        for child in src.iterdir():
            if child.is_dir():
                shutil.copytree(child, self.skill_dir(child.name))

    def diff_summary(self, proposal: SkillProposal) -> str:
        if proposal.action == "create":
            return f"+ create skills/{proposal.skill_name}/"
        if proposal.action == "update":
            return f"~ update skills/{proposal.skill_name}/"
        return "noop"
