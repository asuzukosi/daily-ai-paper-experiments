"""Magic-chunk partitioning for Declarative Attention prompts.

Paper: long context is split into ~2K-token addressable "magic chunks"
that the model can name in <focus magic_chunks="K"> tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Segment:
    """A named contiguous span of the prompt with an approximate token count."""

    name: str
    n_tokens: int
    text: str = ""

    @property
    def chunk_id(self) -> int | None:
        """Numeric id for magic_chunk_N names; None for scaffold segments."""
        if not self.name.startswith("magic_chunk_"):
            return None
        try:
            return int(self.name.rsplit("_", 1)[-1])
        except ValueError:
            return None


def _approx_tokens(text: str) -> int:
    """Rough whitespace token count (good enough for synthetic demos)."""
    return max(1, len(text.split()))


def partition_into_magic_chunks(
    text: str,
    chunk_size: int = 512,
    *,
    start_id: int = 1,
) -> List[Segment]:
    """Split *text* into successive magic chunks of ~chunk_size words.

    Real deployments use ~2K tokenizer tokens; demos may use smaller sizes.
    """
    words = text.split()
    if not words:
        return [Segment("magic_chunk_1", 1, "")]
    chunks: List[Segment] = []
    for i in range(0, len(words), chunk_size):
        piece = " ".join(words[i : i + chunk_size])
        cid = start_id + len(chunks)
        chunks.append(
            Segment(f"magic_chunk_{cid}", _approx_tokens(piece), piece)
        )
    return chunks


def format_magic_chunks(chunks: Sequence[Segment]) -> str:
    """Render chunks with explicit Magic Chunk N headers for the model."""
    parts = []
    for c in chunks:
        cid = c.chunk_id if c.chunk_id is not None else c.name
        parts.append(f"### Magic Chunk {cid}\n{c.text}")
    return "\n\n".join(parts)


def build_demo_segments(
    chunk_tokens: int = 400,
) -> Tuple[List[Segment], List[Segment], int]:
    """Return (scaffold, magic_chunks, total_prompt_tokens) for the Acme demo.

    Matches the didactic example from the paper's figure style.
    """
    scaffold = [
        Segment("A_system", 180, "System: answer using Declarative Attention."),
        Segment(
            "C_question",
            50,
            "Q: How many years after founding did Acme go public?",
        ),
        Segment(
            "D_instruction",
            200,
            "Use <global>/<focus>/<local> tags as instructed.",
        ),
    ]
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
        Segment(
            f"magic_chunk_{i}",
            chunk_tokens + (i % 3) * 50,
            body,
        )
        for i, body in enumerate(chunk_bodies, start=1)
    ]
    total = sum(s.n_tokens for s in scaffold + chunks)
    return scaffold, chunks, total


def build_synthetic_long_context(
    n_chunks: int = 16,
    words_per_chunk: int = 400,
    *,
    relevant_ids: Sequence[int] | None = None,
) -> Tuple[str, List[Segment], str]:
    """Build a long synthetic document + Acme-style question.

    Returns (full_context_text, chunk_segments, question).
    """
    if relevant_ids is None:
        relevant_ids = (2, 7)
    relevant = set(relevant_ids)
    bodies = []
    for i in range(1, n_chunks + 1):
        if i in relevant and i == 2:
            core = "Acme Corp was founded in 2003 in San Jose."
        elif i in relevant and i == 7:
            core = "Acme Corp went public on the NYSE in 2011."
        else:
            core = f"Filler section {i}: operations notes, metrics, and boilerplate."
        # Pad to approximate word budget.
        pad = " ".join(f"word{j}" for j in range(max(0, words_per_chunk - len(core.split()))))
        bodies.append(f"{core} {pad}".strip())
    chunks = [
        Segment(f"magic_chunk_{i}", _approx_tokens(b), b)
        for i, b in enumerate(bodies, start=1)
    ]
    text = format_magic_chunks(chunks)
    question = "How many years after founding did Acme go public?"
    return text, chunks, question
