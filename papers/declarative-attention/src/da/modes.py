"""Mode enum and tag parsing for Declarative Attention."""

from __future__ import annotations

import re
from enum import Enum
from typing import Set


class Mode(Enum):
    GLOBAL = "global"
    FOCUS = "focus"
    LOCAL = "local"


OPEN_TAG_RE = re.compile(
    r"<(global|local|focus)(?:\s+magic_chunks\s*=\s*[\"']([^\"']+)[\"'])?\s*>",
    re.IGNORECASE,
)
CLOSE_TAG_RE = re.compile(r"</(global|local|focus)\s*>", re.IGNORECASE)


def parse_focus_ids(raw: str | None) -> Set[int]:
    """Parse comma-separated magic chunk ids from a focus tag attribute."""
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}
