"""Declarative Attention (DA) — protocol helpers for arXiv:2609.02737.

Zero-shot decode-time attention control via <global>/<focus>/<local> tags.
"""

from .modes import Mode, OPEN_TAG_RE, CLOSE_TAG_RE, parse_focus_ids
from .chunking import Segment, partition_into_magic_chunks, build_demo_segments
from .controller import DAController
from .metrics import AccountingStep, account_decode, summarize_accounting
from .prompting import DA_SYSTEM_PROMPT, build_da_prompt

__all__ = [
    "Mode",
    "OPEN_TAG_RE",
    "CLOSE_TAG_RE",
    "parse_focus_ids",
    "Segment",
    "partition_into_magic_chunks",
    "build_demo_segments",
    "DAController",
    "AccountingStep",
    "account_decode",
    "summarize_accounting",
    "DA_SYSTEM_PROMPT",
    "build_da_prompt",
]

__version__ = "0.1.0"
