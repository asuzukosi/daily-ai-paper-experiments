"""Decode-time Declarative Attention state machine (paper §2.3).

Parses <global>/<focus>/<local> tags from the model's stream and maintains
the current attention mode + focus set. The inference engine would use this
to rewrite the KV block table each step; we also use it for projected savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Set

from .chunking import Segment
from .modes import CLOSE_TAG_RE, Mode, OPEN_TAG_RE, parse_focus_ids


@dataclass
class DAController:
    """Parses model output and maintains the current attention mode / focus set."""

    mode: Mode = Mode.GLOBAL
    focus_ids: Set[int] = field(default_factory=set)
    _buf: str = ""

    def reset(self) -> None:
        self.mode = Mode.GLOBAL
        self.focus_ids = set()
        self._buf = ""

    def feed_token(self, token: str) -> None:
        """Feed one generated token (or word) and update mode on tag boundaries."""
        self._buf += token
        while True:
            open_m = OPEN_TAG_RE.search(self._buf)
            close_m = CLOSE_TAG_RE.search(self._buf)
            candidates = []
            if open_m:
                candidates.append(("open", open_m))
            if close_m:
                candidates.append(("close", close_m))
            if not candidates:
                if len(self._buf) > 80:
                    self._buf = self._buf[-80:]
                break
            kind, m = min(candidates, key=lambda x: x[1].end())
            if kind == "open":
                tag = m.group(1).lower()
                if tag == "focus":
                    self.focus_ids = parse_focus_ids(m.group(2))
                    self.mode = Mode.FOCUS
                elif tag == "local":
                    self.focus_ids = set()
                    self.mode = Mode.LOCAL
                else:
                    self.focus_ids = set()
                    self.mode = Mode.GLOBAL
            else:
                # Closing any mode tag reverts to global (paper §2.3).
                self.mode = Mode.GLOBAL
                self.focus_ids = set()
            self._buf = self._buf[m.end() :]

    def feed_text(self, text: str) -> None:
        """Convenience: feed an entire string (e.g. full generation)."""
        self.feed_token(text)

    def visible_context_tokens(
        self, scaffold: Sequence[Segment], chunks: Sequence[Segment]
    ) -> int:
        """How many *prompt context* tokens this mode would read (excl. response)."""
        scaffold_tok = sum(s.n_tokens for s in scaffold)
        if self.mode == Mode.GLOBAL:
            return scaffold_tok + sum(c.n_tokens for c in chunks)
        if self.mode == Mode.FOCUS:
            kept = sum(
                c.n_tokens
                for c in chunks
                if c.chunk_id is not None and c.chunk_id in self.focus_ids
            )
            return scaffold_tok + kept
        # LOCAL: scaffold only (no magic chunks)
        return scaffold_tok

    def attended_set_description(
        self, scaffold: Sequence[Segment], chunks: Sequence[Segment]
    ) -> str:
        """Human-readable summary of what is attended in the current mode."""
        if self.mode == Mode.GLOBAL:
            return f"scaffold + all {len(chunks)} magic chunks"
        if self.mode == Mode.FOCUS:
            return f"scaffold + magic chunks {sorted(self.focus_ids)}"
        return "scaffold only (local)"
