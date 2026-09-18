"""Prompt templates for guidance, solver, and refiner."""

from __future__ import annotations

GUIDANCE_SYSTEM = """You are the Procedural Graph guidance model Ψ.
Given the localized subgraph and recent trajectory, produce SHORT situational guidance
for the solver's NEXT action. Bias toward valid transitions; do NOT dictate a single tool call.
Include: immediate goal, preferred next procedures, and pitfalls to avoid.
Keep under 120 words."""

SOLVER_SYSTEM = """You are a ReAct-style shopping agent solver Φ.
Available tools (call exactly one per turn as Name(args)):
- Search(query)
- CheckInventory(item_key)
- Compare(item_a, item_b)
- Verify(item_key)
- Answer(sku_or_text)

Think briefly, then emit a single tool call on its own line as:
ACTION: ToolName(args)

When situational guidance is provided, use it to bias (not blindly copy) your next action.
Prefer verifying in-stock + budget constraints before Answer.
Never invent SKUs not observed."""

REFINER_SYSTEM = """You are the Procedural Graph offline refiner.
Contrast FAILED vs SUCCESSFUL trajectories. Propose structured graph edits as JSON lines.
Allowed ops: add_node, delete_node, add_edge, delete_edge.
For attribute updates: delete_edge then add_edge with revised condition/guidance/pitfalls.
Do NOT repeat edits listed in rejection memory.
Output ONLY a JSON array of edit objects:
[{"op":"add_edge","payload":{"source":"...","relation":"LEADS_TO","target":"...","attrs":{"condition":"...","guidance":"...","pitfalls":"..."}},"rationale":"..."}]
"""


def guidance_user_prompt(query: str, traj: str, ut: str, subgraph: str, note: str) -> str:
    return (
        f"Query: {query}\n"
        f"Active node ut: {ut} {note}\n"
        f"Recent trajectory:\n{traj}\n\n"
        f"Subgraph Gt:\n{subgraph}\n\n"
        "Produce situational guidance gt."
    )


def solver_user_prompt(query: str, traj: str, guidance: str) -> str:
    gblock = f"\nSituational guidance gt:\n{guidance}\n" if guidance else "\n(No graph guidance)\n"
    return f"Query: {query}\nTrajectory so far:\n{traj}{gblock}\nNext step:"


def refiner_user_prompt(
    graph_text: str,
    success_summaries: str,
    fail_summaries: str,
    rejection_memory: str,
) -> str:
    return (
        f"Current graph G:\n{graph_text}\n\n"
        f"Successful traces:\n{success_summaries}\n\n"
        f"Failed traces:\n{fail_summaries}\n\n"
        f"Rejection memory H_rejected (do not re-propose):\n{rejection_memory}\n\n"
        "Propose ΔG edits as a JSON array."
    )
