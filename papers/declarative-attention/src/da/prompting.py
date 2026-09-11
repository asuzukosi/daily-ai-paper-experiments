"""DA system / instruction templates for the base model."""

from __future__ import annotations

from typing import Sequence

from .chunking import Segment, format_magic_chunks

DA_SYSTEM_PROMPT = """\
You are an assistant that uses Declarative Attention (DA) while reasoning.

While solving the user's question over a long document split into numbered \
"Magic Chunks", you MUST declare attention scope inside your chain-of-thought \
using exactly these tags:

1. <global>...</global>
   Use when navigating / locating which chunks matter. The engine attends to the \
full context.

2. <focus magic_chunks="K">...</focus>
   or <focus magic_chunks="K,M">...</focus>
   Use when reading specific Magic Chunk id(s). The engine attends only to those \
chunks (plus the scaffold and your response so far).

3. <local>...</local>
   Use for arithmetic, synthesis, or writing that does not need the long document. \
The engine attends only to the scaffold and your response so far.

Rules:
- Start by briefly scanning in <global> to find relevant chunk ids.
- Switch to <focus> to quote or extract facts from those chunks.
- Use <local> for final computation and put the final answer in <answer>...</answer>.
- Only reference Magic Chunk ids that exist in the document.
- Do not invent other tag names.
"""

DA_INSTRUCTION_BLOCK = """\
Follow the Declarative Attention protocol above. Emit <global>, <focus magic_chunks="...">, \
and <local> spans in your reasoning, then finish with <answer>...</answer>.
"""


def build_da_prompt(
    context_chunks: Sequence[Segment],
    question: str,
    *,
    system: str = DA_SYSTEM_PROMPT,
    instruction: str = DA_INSTRUCTION_BLOCK,
) -> str:
    """Assemble system + magic-chunk document + question + DA instruction."""
    doc = format_magic_chunks(context_chunks)
    return (
        f"{system.strip()}\n\n"
        f"## Document\n\n{doc}\n\n"
        f"## Question\n{question.strip()}\n\n"
        f"## Instructions\n{instruction.strip()}\n"
    )


def scaffold_segments_from_prompt_parts(
    system: str,
    question: str,
    instruction: str,
) -> list[Segment]:
    """Approximate scaffold token counts for accounting (non-chunk parts)."""

    def _tok(t: str) -> int:
        return max(1, len(t.split()))

    return [
        Segment("A_system", _tok(system), system),
        Segment("C_question", _tok(question), question),
        Segment("D_instruction", _tok(instruction), instruction),
    ]
