"""Wiki Layer — compounding structured patterns (never reset)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


class WikiStore:
    """Persistent wiki under wiki/.

    Layout:
      wiki/index.md
      wiki/logs.md
      wiki/skill-impact.md
      wiki/patterns/*.md
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.patterns_dir = self.root / "patterns"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.md"
        self.logs_path = self.root / "logs.md"
        self.skill_impact_path = self.root / "skill-impact.md"
        self._ensure_bootstrap()

    def _ensure_bootstrap(self) -> None:
        if not self.index_path.exists():
            self.index_path.write_text(
                "# Wiki Pattern Index\n\n"
                "Catalog of failure modes and successful strategies.\n"
                "Updated by the Wiki Maintainer each iteration.\n\n"
                "## Patterns\n\n_(none yet)_\n",
                encoding="utf-8",
            )
        if not self.logs_path.exists():
            self.logs_path.write_text(
                "# Evolution Log\n\n"
                "Chronological record of pattern updates and gating decisions.\n\n",
                encoding="utf-8",
            )
        if not self.skill_impact_path.exists():
            self.skill_impact_path.write_text(
                "# Skill Impact Tracker\n\n"
                "Programmatic audit trail of skill proposals and accept/reject outcomes.\n\n"
                "| Iteration | Skill | Action | Val Score | Decision | Notes |\n"
                "|-----------|-------|--------|-----------|----------|-------|\n",
                encoding="utf-8",
            )

    def pattern_path(self, name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
        return self.patterns_dir / f"{safe}.md"

    def list_patterns(self) -> List[str]:
        return sorted(p.stem for p in self.patterns_dir.glob("*.md"))

    def n_patterns(self) -> int:
        return len(self.list_patterns())

    def read_pattern(self, name: str) -> Optional[str]:
        p = self.pattern_path(name)
        if p.exists():
            return p.read_text(encoding="utf-8")
        for existing in self.patterns_dir.glob("*.md"):
            if existing.stem == name:
                return existing.read_text(encoding="utf-8")
        return None

    def write_pattern(self, name: str, content: str, *, append: bool = False) -> Path:
        path = self.pattern_path(name)
        if append and path.exists():
            prev = path.read_text(encoding="utf-8")
            path.write_text(prev.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")
        else:
            path.write_text(content.strip() + "\n", encoding="utf-8")
        self.refresh_index()
        return path

    def refresh_index(self) -> None:
        patterns = self.list_patterns()
        lines = [
            "# Wiki Pattern Index",
            "",
            "Catalog of failure modes and successful strategies.",
            "Updated by the Wiki Maintainer each iteration.",
            "",
            f"**Pattern count:** {len(patterns)}",
            "",
            "## Patterns",
            "",
        ]
        if not patterns:
            lines.append("_(none yet)_")
        else:
            for name in patterns:
                path = self.pattern_path(name)
                first = ""
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith("#"):
                        first = line.strip()[:120]
                        break
                lines.append(f"- [{name}](patterns/{name}.md) — {first}")
        lines.append("")
        self.index_path.write_text("\n".join(lines), encoding="utf-8")

    def read_index(self) -> str:
        return self.index_path.read_text(encoding="utf-8")

    def read_logs(self) -> str:
        return self.logs_path.read_text(encoding="utf-8")

    def append_log(self, iteration: int, bullet_lines: List[str]) -> None:
        block = [f"## Iteration {iteration}", ""]
        for b in bullet_lines:
            block.append(f"- {b}" if not b.startswith("-") else b)
        block.append("")
        with self.logs_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(block) + "\n")

    def append_skill_impact(
        self,
        *,
        iteration: int,
        skill: str,
        action: str,
        val_score: float,
        decision: str,
        notes: str = "",
    ) -> None:
        row = (
            f"| {iteration} | `{skill}` | {action} | {val_score:.3f} | "
            f"**{decision}** | {notes.replace('|', '/')} |\n"
        )
        with self.skill_impact_path.open("a", encoding="utf-8") as f:
            f.write(row)

    def read_skill_impact(self) -> str:
        return self.skill_impact_path.read_text(encoding="utf-8")

    def snapshot_summary(self) -> Dict[str, object]:
        return {
            "n_patterns": self.n_patterns(),
            "patterns": self.list_patterns(),
            "index_chars": len(self.read_index()),
            "logs_chars": len(self.read_logs()),
        }
