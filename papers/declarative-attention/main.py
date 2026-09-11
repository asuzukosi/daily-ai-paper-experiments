#!/usr/bin/env python3
"""
Declarative Attention — didactic simulator

Demonstrates the CORE IDEA from Ho et al., arXiv:2609.02737
("Language Models Can Control Their Own Attention"):

  The model declares attention scope inside its chain-of-thought via
  <global> / <focus magic_chunks="K"> / <local> tags. An inference-style
  controller parses those tags (like tool calls) and restricts which
  context tokens are "read" at each decoding step.

This is NOT a transformer / NOT training — a token-counting simulator
that shows savings vs full (vanilla) attention over a mock long context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Context: scaffold + magic chunks (paper: ~2K-token addressable segments)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    name: str
    n_tokens: int
    text: str = ""


def build_demo_prompt() -> Tuple[List[Segment], List[Segment], int]:
    """Return (scaffold_segments, context_chunks, total_prompt_tokens)."""
    scaffold = [
        Segment("A_system", 180, "System: answer using Declarative Attention."),
        Segment("C_question", 50, "Q: How many years after founding did Acme go public?"),
        Segment("D_instruction", 200, "Use <global>/<focus>/<local> tags as instructed."),
    ]
    # Twelve short "magic chunks" stand in for a long document (paper Fig. 1 style).
    # Chunk sizes are small for a readable demo but preserve the structure.
    chunk_bodies = [
        "Acme facilities overview; HQ in San Jose.",
        "Acme Corp was founded in 2003 in San Jose.",  # relevant: founding
        "Early product roadmap and hiring notes.",
        "Office expansion across APAC.",
        "Partnership announcements 2008–2010.",
        "Internal wiki: culture and values.",
        "Acme Corp went public on the NYSE in 2011.",  # relevant: IPO
        "Board composition after IPO.",
        "Revenue footnotes FY2012.",
        "Customer case studies.",
        "Legal boilerplate appendix.",
        "Glossary of internal acronyms.",
    ]
    chunks = [
        Segment(f"magic_chunk_{i}", 400 + (i % 3) * 50, body)
        for i, body in enumerate(chunk_bodies, start=1)
    ]
    total = sum(s.n_tokens for s in scaffold + chunks)
    return scaffold, chunks, total


# ---------------------------------------------------------------------------
# DA state machine (paper §2.3)
# ---------------------------------------------------------------------------

class Mode(Enum):
    GLOBAL = "global"
    FOCUS = "focus"
    LOCAL = "local"


OPEN_TAG_RE = re.compile(
    r"<(global|local|focus)(?:\s+magic_chunks\s*=\s*[\"']([^\"']+)[\"'])?\s*>",
    re.IGNORECASE,
)
CLOSE_TAG_RE = re.compile(r"</(global|local|focus)\s*>", re.IGNORECASE)


@dataclass
class DAController:
    """Parses model output and maintains the current attention mode / focus set."""

    mode: Mode = Mode.GLOBAL
    focus_ids: Set[int] = field(default_factory=set)
    _buf: str = ""

    def feed_token(self, token: str) -> None:
        """Feed one generated 'token' (word/punct) and update mode on tag boundaries."""
        self._buf += token
        # Process as many complete tags as possible from the buffer.
        while True:
            open_m = OPEN_TAG_RE.search(self._buf)
            close_m = CLOSE_TAG_RE.search(self._buf)
            # Prefer whichever tag ends first in the buffer.
            candidates = []
            if open_m:
                candidates.append(("open", open_m))
            if close_m:
                candidates.append(("close", close_m))
            if not candidates:
                # Keep a short tail in case a tag is split across tokens.
                if len(self._buf) > 80:
                    self._buf = self._buf[-80:]
                break
            kind, m = min(candidates, key=lambda x: x[1].end())
            if kind == "open":
                tag = m.group(1).lower()
                if tag == "focus":
                    raw = m.group(2) or ""
                    self.focus_ids = {
                        int(x.strip()) for x in raw.split(",") if x.strip().isdigit()
                    }
                    self.mode = Mode.FOCUS
                elif tag == "local":
                    self.focus_ids = set()
                    self.mode = Mode.LOCAL
                else:
                    # <global> is prompting structure; default between spans is already global.
                    self.focus_ids = set()
                    self.mode = Mode.GLOBAL
            else:
                # Closing any mode tag reverts to global (paper §2.3).
                self.mode = Mode.GLOBAL
                self.focus_ids = set()
            self._buf = self._buf[m.end() :]

    def visible_context_tokens(
        self, scaffold: Sequence[Segment], chunks: Sequence[Segment]
    ) -> int:
        """How many *prompt context* tokens this mode would read (excl. response so far)."""
        scaffold_tok = sum(s.n_tokens for s in scaffold)
        if self.mode == Mode.GLOBAL:
            return scaffold_tok + sum(c.n_tokens for c in chunks)
        if self.mode == Mode.FOCUS:
            kept = sum(
                c.n_tokens
                for c in chunks
                if int(c.name.rsplit("_", 1)[-1]) in self.focus_ids
            )
            return scaffold_tok + kept
        # LOCAL: scaffold only (no magic chunks)
        return scaffold_tok


# ---------------------------------------------------------------------------
# Simulated generation (scripted CoT matching paper's Acme example)
# ---------------------------------------------------------------------------

DEMO_RESPONSE = """\
<global>
I need the founding year and the IPO year. The company history in Magic Chunk 2 should state the founding.
</global>
<focus magic_chunks="2">
"Acme Corp was founded in 2003 in San Jose."
</focus>
<global>
The IPO year is still missing. Magic Chunk 7 covers Acme's financial milestones.
</global>
<focus magic_chunks="7">
"Acme Corp went public on the NYSE in 2011."
</focus>
<local>
2011 - 2003 = 8 years.
</local>
<answer>8 years</answer>
"""


def tokenize_rough(text: str) -> List[str]:
    """Whitespace + keep tags glued so the controller can parse them."""
    # Split on whitespace but keep punctuation attached to words for simplicity.
    parts = text.split()
    return [p + " " for p in parts]


def simulate(response: str, scaffold: List[Segment], chunks: List[Segment]):
    tokens = tokenize_rough(response)
    ctrl = DAController()
    steps = []
    response_so_far = 0
    attended_total = 0
    vanilla_total = 0
    prompt_tok = sum(s.n_tokens for s in scaffold + chunks)

    for tok in tokens:
        ctrl.feed_token(tok)
        response_so_far += 1
        # Visible = allowed prompt tokens + all tokens generated so far (causal).
        visible_prompt = ctrl.visible_context_tokens(scaffold, chunks)
        attended = visible_prompt + response_so_far
        vanilla = prompt_tok + response_so_far
        attended_total += attended
        vanilla_total += vanilla
        steps.append(
            {
                "token": tok.strip(),
                "mode": ctrl.mode.value,
                "focus": sorted(ctrl.focus_ids),
                "attended": attended,
                "vanilla": vanilla,
                "saving_pct": 100.0 * (1.0 - attended / vanilla),
            }
        )
    return steps, attended_total, vanilla_total, prompt_tok


def mode_spans(steps: List[dict]) -> List[Tuple[str, int, Optional[List[int]], dict]]:
    """Collapse consecutive steps with the same mode; attach a mid-span sample."""
    spans = []
    if not steps:
        return spans
    start = 0
    cur_mode = steps[0]["mode"]
    cur_focus = steps[0]["focus"]
    for i, s in enumerate(steps):
        if s["mode"] != cur_mode or s["focus"] != cur_focus:
            mid = steps[start + (i - start) // 2]
            spans.append((cur_mode, i - start, cur_focus if cur_mode == "focus" else None, mid))
            start, cur_mode, cur_focus = i, s["mode"], s["focus"]
    mid = steps[start + (len(steps) - start) // 2]
    spans.append((cur_mode, len(steps) - start, cur_focus if cur_mode == "focus" else None, mid))
    return spans


def main() -> None:
    scaffold, chunks, prompt_tok = build_demo_prompt()
    print("=" * 68)
    print("Declarative Attention — didactic decoder simulator")
    print("Paper: Language Models Can Control Their Own Attention (arXiv:2609.02737)")
    print("=" * 68)
    print(f"\nPrompt: {prompt_tok} tokens "
          f"({sum(s.n_tokens for s in scaffold)} scaffold + "
          f"{sum(c.n_tokens for c in chunks)} across {len(chunks)} magic chunks)")
    print("Question: How many years after founding did Acme go public?\n")

    steps, attended_total, vanilla_total, _ = simulate(DEMO_RESPONSE, scaffold, chunks)
    n = len(steps)

    print("Mode spans in the simulated CoT (controller view):")
    for mode, count, focus, mid in mode_spans(steps):
        extra = f" chunks={focus}" if focus is not None else ""
        print(
            f"  [{mode:6s}{extra}]  {count:3d} decode steps  |  "
            f"~{mid['attended']} attended / {mid['vanilla']} vanilla "
            f"({mid['saving_pct']:.1f}% save this step)"
        )

    saving = 100.0 * (1.0 - attended_total / vanilla_total)
    print("\n" + "-" * 68)
    print(f"Decode steps:              {n}")
    print(f"Vanilla attended (sum):    {vanilla_total:,}  (full KV every step)")
    print(f"DA attended (sum):         {attended_total:,}  (mode-gated KV reads)")
    print(f"Token-read savings:        {saving:.1f}%")
    print("-" * 68)
    print(
        "\nTakeaway: global steps still scan the full context; focus/local steps "
        "skip most magic chunks — same idea as the paper's inference engine parsing "
        "declarations like tool calls."
    )
    print(
        f"\n(Paper headline, real models, 15 tasks: −52.0% attended tokens on "
        f"Gemma-4-31B, −31.1% on Qwen-3.6-27B; this demo is illustrative only.)"
    )


if __name__ == "__main__":
    main()
