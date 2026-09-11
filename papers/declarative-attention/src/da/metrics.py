"""Vanilla vs DA attended-token accounting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunking import Segment
from .controller import DAController


@dataclass
class AccountingStep:
    token: str
    mode: str
    focus: List[int]
    attended: int
    vanilla: int
    saving_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tokenize_rough(text: str) -> List[str]:
    """Whitespace split; keep a trailing space so tags spanning words still parse."""
    parts = text.split()
    return [p + " " for p in parts]


def account_decode(
    response: str,
    scaffold: Sequence[Segment],
    chunks: Sequence[Segment],
    *,
    controller: Optional[DAController] = None,
) -> Tuple[List[AccountingStep], int, int, int]:
    """Walk a generated response and accumulate attended vs vanilla token reads.

    Returns (steps, attended_total, vanilla_total, prompt_tok).
    """
    tokens = tokenize_rough(response)
    ctrl = controller or DAController()
    steps: List[AccountingStep] = []
    response_so_far = 0
    attended_total = 0
    vanilla_total = 0
    prompt_tok = sum(s.n_tokens for s in list(scaffold) + list(chunks))

    for tok in tokens:
        ctrl.feed_token(tok)
        response_so_far += 1
        visible_prompt = ctrl.visible_context_tokens(scaffold, chunks)
        attended = visible_prompt + response_so_far
        vanilla = prompt_tok + response_so_far
        attended_total += attended
        vanilla_total += vanilla
        steps.append(
            AccountingStep(
                token=tok.strip(),
                mode=ctrl.mode.value,
                focus=sorted(ctrl.focus_ids),
                attended=attended,
                vanilla=vanilla,
                saving_pct=100.0 * (1.0 - attended / vanilla) if vanilla else 0.0,
            )
        )
    return steps, attended_total, vanilla_total, prompt_tok


def mode_spans(
    steps: Sequence[AccountingStep],
) -> List[Tuple[str, int, Optional[List[int]], AccountingStep]]:
    """Collapse consecutive steps with the same mode; attach a mid-span sample."""
    spans: List[Tuple[str, int, Optional[List[int]], AccountingStep]] = []
    if not steps:
        return spans
    start = 0
    cur_mode = steps[0].mode
    cur_focus = steps[0].focus
    for i, s in enumerate(steps):
        if s.mode != cur_mode or s.focus != cur_focus:
            mid = steps[start + (i - start) // 2]
            spans.append(
                (cur_mode, i - start, cur_focus if cur_mode == "focus" else None, mid)
            )
            start, cur_mode, cur_focus = i, s.mode, s.focus
    mid = steps[start + (len(steps) - start) // 2]
    spans.append(
        (cur_mode, len(steps) - start, cur_focus if cur_mode == "focus" else None, mid)
    )
    return spans


def summarize_accounting(
    steps: Sequence[AccountingStep],
    attended_total: int,
    vanilla_total: int,
) -> Dict[str, Any]:
    """Aggregate metrics suitable for JSONL logging."""
    n = len(steps)
    saving = 100.0 * (1.0 - attended_total / vanilla_total) if vanilla_total else 0.0
    mode_counts: Dict[str, int] = {}
    for s in steps:
        mode_counts[s.mode] = mode_counts.get(s.mode, 0) + 1
    return {
        "decode_steps": n,
        "attended_total": attended_total,
        "vanilla_total": vanilla_total,
        "token_read_saving_pct": round(saving, 2),
        "mode_counts": mode_counts,
        "mode_fractions": {
            k: round(v / n, 4) if n else 0.0 for k, v in mode_counts.items()
        },
        "protocol_valid": any(s.mode == "focus" for s in steps)
        or any(s.mode == "local" for s in steps),
    }
