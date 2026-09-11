#!/usr/bin/env python3
"""Declarative Attention — educational CPU protocol simulator (stdlib only).

Refactored from the original single-file didactic demo. Demonstrates the
CORE IDEA from Ho et al., arXiv:2609.02737:

  The model declares attention scope inside its chain-of-thought via
  <global> / <focus magic_chunks="K"> / <local> tags. An inference-style
  controller parses those tags (like tool calls) and restricts which
  context tokens are "read" at each decoding step.

This is NOT a transformer / NOT training — a token-counting simulator
that shows savings vs full (vanilla) attention over a mock long context.

Runnable as:
  python -m demos.cpu_protocol_demo
  python demos/cpu_protocol_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from da.chunking import build_demo_segments  # noqa: E402
from da.metrics import account_decode, mode_spans, summarize_accounting  # noqa: E402


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


def main() -> None:
    scaffold, chunks, prompt_tok = build_demo_segments()
    print("=" * 68)
    print("Declarative Attention — didactic decoder simulator")
    print("Paper: Language Models Can Control Their Own Attention (arXiv:2609.02737)")
    print("=" * 68)
    print(
        f"\nPrompt: {prompt_tok} tokens "
        f"({sum(s.n_tokens for s in scaffold)} scaffold + "
        f"{sum(c.n_tokens for c in chunks)} across {len(chunks)} magic chunks)"
    )
    print("Question: How many years after founding did Acme go public?\n")

    steps, attended_total, vanilla_total, _ = account_decode(
        DEMO_RESPONSE, scaffold, chunks
    )
    summary = summarize_accounting(steps, attended_total, vanilla_total)

    print("Mode spans in the simulated CoT (controller view):")
    for mode, count, focus, mid in mode_spans(steps):
        extra = f" chunks={focus}" if focus is not None else ""
        print(
            f"  [{mode:6s}{extra}]  {count:3d} decode steps  |  "
            f"~{mid.attended} attended / {mid.vanilla} vanilla "
            f"({mid.saving_pct:.1f}% save this step)"
        )

    print("\n" + "-" * 68)
    print(f"Decode steps:              {summary['decode_steps']}")
    print(f"Vanilla attended (sum):    {summary['vanilla_total']:,}  (full KV every step)")
    print(f"DA attended (sum):         {summary['attended_total']:,}  (mode-gated KV reads)")
    print(f"Token-read savings:        {summary['token_read_saving_pct']:.1f}%")
    print("-" * 68)
    print(
        "\nTakeaway: global steps still scan the full context; focus/local steps "
        "skip most magic chunks — same idea as the paper's inference engine parsing "
        "declarations like tool calls."
    )
    print(
        "\n(Paper headline, real models, 15 tasks: −52.0% attended tokens on "
        "Gemma-4-31B, −31.1% on Qwen-3.6-27B; this demo is illustrative only.)"
    )


if __name__ == "__main__":
    main()
