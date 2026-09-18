"""Online generative guidance: Match → extract N_h → Ψ (§3.2)."""

from __future__ import annotations

from typing import Any, Optional

from .graph import ProceduralGraph
from .prompts import GUIDANCE_SYSTEM, guidance_user_prompt


class GuidanceEngine:
    def __init__(self, llm: Any, *, hop: int = 2, history_window: int = 3):
        self.llm = llm
        self.hop = hop
        self.history_window = history_window

    def localize(self, graph: ProceduralGraph, last_action: Optional[str]) -> str:
        ut = graph.match_active(last_action)
        return ut if ut is not None else ProceduralGraph.START

    def subgraph(self, graph: ProceduralGraph, ut: Optional[str]) -> ProceduralGraph:
        if ut is None or ut not in graph.nodes:
            return graph
        return graph.hop_neighborhood(ut, h=self.hop)

    def generate(
        self,
        graph: ProceduralGraph,
        query: str,
        trajectory_text: str,
        last_action: Optional[str],
    ) -> tuple[str, str, str]:
        """Return (active_node, guidance_text, subgraph_text)."""
        ut = self.localize(graph, last_action)
        Gt = self.subgraph(graph, ut if ut != "" else None)
        if ut is None or (ut not in graph.nodes) or (not Gt.edges and graph.edges):
            ut = ut or ProceduralGraph.START
            Gt = graph
            sub_note = "(fallback: full graph)"
        else:
            sub_note = f"(localized {self.hop}-hop from {ut})"
        sub_text = Gt.render_text()
        user = guidance_user_prompt(query, trajectory_text, ut, sub_text, sub_note)
        gt = self.llm.generate(system=GUIDANCE_SYSTEM, user=user, role="guidance")
        return ut or ProceduralGraph.START, gt.strip(), sub_text
