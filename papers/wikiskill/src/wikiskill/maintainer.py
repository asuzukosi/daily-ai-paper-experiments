"""Wiki Maintainer — root-cause analysis → pattern pages + logs (§3.2.2)."""

from __future__ import annotations

import re
from typing import Any, List

from wikiskill.prompts import maintainer_prompt
from wikiskill.traces import RawStore
from wikiskill.wiki import WikiStore


class WikiMaintainer:
    def __init__(self, llm: Any, wiki: WikiStore, raw: RawStore):
        self.llm = llm
        self.wiki = wiki
        self.raw = raw

    def run(self, *, iteration: int, max_fail: int = 6, max_pass: int = 4) -> List[str]:
        sample = self.raw.sample(
            iteration, "train", max_fail=max_fail, max_pass=max_pass
        )
        prompt = maintainer_prompt(
            wiki_index=self.wiki.read_index(),
            sample_traces=sample,
            iteration=iteration,
        )
        text = self.llm.generate(
            system="You maintain a persistent wiki of agent failure/success patterns.",
            user=prompt,
            role="maintainer",
        )
        return self._apply_response(text, iteration=iteration)

    def _apply_response(self, text: str, *, iteration: int) -> List[str]:
        updated: List[str] = []
        pattern_blocks = re.split(r"(?m)^##\s*PATTERN:\s*", text)
        log_bullets: List[str] = []
        for block in pattern_blocks[1:]:
            lines = block.strip().splitlines()
            if not lines:
                continue
            name = lines[0].strip().split()[0].strip("#").strip()
            body_lines: List[str] = []
            for ln in lines[1:]:
                if ln.strip().upper().startswith("## LOG"):
                    break
                body_lines.append(ln)
            body = "\n".join(body_lines).strip()
            if not name:
                continue
            header = f"# Pattern: {name}\n\n"
            evidence = f"\n\n## Evidence\n- Iteration {iteration}\n"
            existing = self.wiki.read_pattern(name)
            if existing:
                self.wiki.write_pattern(name, evidence, append=True)
            else:
                self.wiki.write_pattern(name, header + body + evidence)
            updated.append(name)

        m = re.search(r"(?im)^##\s*LOG\s*$([\s\S]*)", text)
        if m:
            for ln in m.group(1).splitlines():
                ln = ln.strip()
                if ln.startswith("-"):
                    log_bullets.append(ln.lstrip("- ").strip())
        if not log_bullets:
            log_bullets = [
                f"Updated patterns: {', '.join(updated) if updated else 'none'}"
            ]
        self.wiki.append_log(iteration, log_bullets)
        return updated
