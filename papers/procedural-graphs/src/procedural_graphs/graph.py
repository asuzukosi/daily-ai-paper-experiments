"""Procedural Graph store G = (V, R, E, Φ) with hop neighborhoods and edits."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import Edge, EdgeAttrs, GraphEdit, Node


class ProceduralGraph:
    """Directed attributed procedural graph (§3.1)."""

    START = "Start"
    END = "End"

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.ensure_node(self.START, kind="state", description="Episode start marker")
        self.ensure_node(self.END, kind="state", description="Terminal / answer emitted")

    def ensure_node(self, nid: str, kind: str = "tool", description: str = "") -> None:
        if nid not in self.nodes:
            self.nodes[nid] = Node(id=nid, kind=kind, description=description)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        condition: str = "",
        guidance: str = "",
        pitfalls: str = "",
    ) -> Edge:
        self.ensure_node(source)
        self.ensure_node(target)
        e = Edge(
            source=source,
            relation=relation,
            target=target,
            attrs=EdgeAttrs(condition=condition, guidance=guidance, pitfalls=pitfalls),
        )
        self.edges[e.key()] = e
        return e

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        key = f"{source}|{relation}|{target}"
        return self.edges.pop(key, None) is not None

    def remove_node(self, nid: str) -> None:
        if nid in (self.START, self.END):
            return
        self.nodes.pop(nid, None)
        dead = [k for k, e in self.edges.items() if e.source == nid or e.target == nid]
        for k in dead:
            del self.edges[k]

    def node_ids(self) -> Set[str]:
        return set(self.nodes)

    def outgoing(self, nid: str) -> List[Edge]:
        return [e for e in self.edges.values() if e.source == nid]

    def hop_neighborhood(self, ut: str, h: int = 2) -> "ProceduralGraph":
        """N_h(ut): ut plus outgoing transitions within h hops (§3.2)."""
        sub = ProceduralGraph()
        # clear auto Start/End for clean sub; re-add needed
        sub.nodes.clear()
        sub.edges.clear()
        if ut not in self.nodes:
            return sub
        frontier = {ut}
        visited: Set[str] = {ut}
        sub.nodes[ut] = copy.deepcopy(self.nodes[ut])
        for _ in range(max(0, h)):
            nxt: Set[str] = set()
            for u in frontier:
                for e in self.outgoing(u):
                    sub.edges[e.key()] = copy.deepcopy(e)
                    if e.target not in sub.nodes and e.target in self.nodes:
                        sub.nodes[e.target] = copy.deepcopy(self.nodes[e.target])
                    if e.target not in visited:
                        visited.add(e.target)
                        nxt.add(e.target)
            frontier = nxt
        return sub

    def match_active(self, last_action: Optional[str]) -> Optional[str]:
        """Match(a_{t-1}, V): exact match of latest procedure to a node (§3.2)."""
        if last_action is None or last_action == "":
            return self.START
        # Normalize: tool name before '(' or whitespace
        name = last_action.strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if name in self.nodes:
            return name
        # case-insensitive fallback
        lower = {k.lower(): k for k in self.nodes}
        return lower.get(name.lower())

    def apply_edits(self, edits: List[GraphEdit]) -> Tuple["ProceduralGraph", List[str]]:
        """Return G ⊕ ΔG on a deep copy; list structural warnings."""
        g = self.clone()
        warnings: List[str] = []
        for ed in edits:
            op = ed.op.lower()
            p = ed.payload
            if op == "add_node":
                nid = str(p.get("id") or p.get("node_id") or "")
                if not nid:
                    warnings.append("add_node missing id")
                    continue
                g.ensure_node(nid, kind=str(p.get("kind", "tool")), description=str(p.get("description", "")))
            elif op == "delete_node":
                nid = str(p.get("id") or p.get("node_id") or "")
                if nid in (g.START, g.END):
                    warnings.append(f"refuse delete protected node {nid}")
                    continue
                if nid not in g.nodes:
                    warnings.append(f"delete_node unknown {nid}")
                    continue
                g.remove_node(nid)
            elif op == "add_edge":
                src = str(p.get("source", ""))
                rel = str(p.get("relation", "LEADS_TO"))
                tgt = str(p.get("target", ""))
                if not src or not tgt:
                    warnings.append("add_edge missing endpoints")
                    continue
                attrs = p.get("attrs") or {}
                g.add_edge(
                    src,
                    rel,
                    tgt,
                    condition=str(attrs.get("condition", p.get("condition", ""))),
                    guidance=str(attrs.get("guidance", p.get("guidance", ""))),
                    pitfalls=str(attrs.get("pitfalls", p.get("pitfalls", ""))),
                )
            elif op == "delete_edge":
                src = str(p.get("source", ""))
                rel = str(p.get("relation", "LEADS_TO"))
                tgt = str(p.get("target", ""))
                if not g.remove_edge(src, rel, tgt):
                    warnings.append(f"delete_edge missing {src}|{rel}|{tgt}")
            else:
                warnings.append(f"unknown op {ed.op}")
        g.repair_orphan_start()
        return g, warnings

    def repair_orphan_start(self) -> None:
        """Ensure Start has at least one outgoing edge if other tool nodes exist."""
        tools = [n for n in self.nodes if n not in (self.START, self.END)]
        if tools and not self.outgoing(self.START):
            self.add_edge(
                self.START,
                "LEADS_TO",
                tools[0],
                condition="episode begins",
                guidance=f"Begin with {tools[0]}",
                pitfalls="Do not skip the first procedural step",
            )

    def structural_ok(self) -> Tuple[bool, str]:
        if self.START not in self.nodes:
            return False, "missing Start"
        if not self.edges and len(self.nodes) > 2:
            return False, "nodes without edges"
        # detect simple invalid self-only loops without End reachability soft-check
        for e in self.edges.values():
            if e.source not in self.nodes or e.target not in self.nodes:
                return False, f"dangling edge {e.key()}"
        return True, "ok"

    def render_text(self) -> str:
        lines = ["# Procedural Graph", "## Nodes"]
        for n in sorted(self.nodes.values(), key=lambda x: x.id):
            lines.append(f"- {n.id} ({n.kind}): {n.description}")
        lines.append("## Edges (procedure, relation, procedure) + Φ")
        for e in sorted(self.edges.values(), key=lambda x: x.key()):
            a = e.attrs
            lines.append(
                f"- ({e.source}) -[{e.relation}]-> ({e.target})\n"
                f"    condition: {a.condition}\n"
                f"    guidance: {a.guidance}\n"
                f"    pitfalls: {a.pitfalls}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    def clone(self) -> "ProceduralGraph":
        g = ProceduralGraph.from_dict(self.to_dict())
        return g

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProceduralGraph":
        g = cls()
        g.nodes.clear()
        g.edges.clear()
        for nd in d.get("nodes") or []:
            n = Node.from_dict(nd)
            g.nodes[n.id] = n
        if cls.START not in g.nodes:
            g.ensure_node(cls.START, kind="state", description="Episode start marker")
        if cls.END not in g.nodes:
            g.ensure_node(cls.END, kind="state", description="Terminal")
        for ed in d.get("edges") or []:
            e = Edge.from_dict(ed)
            g.edges[e.key()] = e
            g.ensure_node(e.source)
            g.ensure_node(e.target)
        return g

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ProceduralGraph":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def minimal_skeleton() -> ProceduralGraph:
    """Mode 4/5 scratch skeleton: Search → Check → Answer (paper-inspired)."""
    g = ProceduralGraph()
    g.ensure_node("Search", kind="tool", description="Search catalog / facts")
    g.ensure_node("CheckInventory", kind="tool", description="Verify stock and constraints")
    g.ensure_node("Compare", kind="reasoning", description="Compare candidate options")
    g.ensure_node("Verify", kind="reasoning", description="Verify answer before commit")
    g.ensure_node("Answer", kind="tool", description="Emit final answer / checkout")
    g.add_edge(
        "Start", "LEADS_TO", "Search",
        condition="new query",
        guidance="Search for the requested item or fact first.",
        pitfalls="Do not answer without searching when tools are available.",
    )
    g.add_edge(
        "Search", "LEADS_TO", "CheckInventory",
        condition="search returned candidates",
        guidance="Check inventory/price constraints on the top hit.",
        pitfalls="Avoid repeating Search with the same query.",
    )
    g.add_edge(
        "CheckInventory", "LEADS_TO", "Compare",
        condition="multiple viable candidates or need ranking",
        guidance="Compare candidates against budget and requirements.",
        pitfalls="Do not buy before comparing when alternatives exist.",
    )
    g.add_edge(
        "Compare", "LEADS_TO", "Verify",
        condition="preferred candidate selected",
        guidance="Verify the chosen item matches the query constraints.",
        pitfalls="Skipping verify causes wrong checkouts.",
    )
    g.add_edge(
        "Verify", "LEADS_TO", "Answer",
        condition="verification passed",
        guidance="Emit the final answer / product id.",
        pitfalls="Do not invent SKUs not seen in observations.",
    )
    g.add_edge(
        "Answer", "LEADS_TO", "End",
        condition="answer emitted",
        guidance="Stop.",
        pitfalls="",
    )
    # recovery edge
    g.add_edge(
        "Search", "FALLBACK", "Search",
        condition="empty results",
        guidance="Broaden query once, then move on.",
        pitfalls="Do not loop Search more than twice.",
    )
    return g


def flawed_expert_prior() -> ProceduralGraph:
    """Mode 1-style flawed prior: Answer immediately after Search (skips verify)."""
    g = ProceduralGraph()
    g.ensure_node("Search", kind="tool", description="Search")
    g.ensure_node("Answer", kind="tool", description="Answer too early")
    g.add_edge(
        "Start", "LEADS_TO", "Search",
        condition="always",
        guidance="Search then answer immediately.",
        pitfalls="",
    )
    g.add_edge(
        "Search", "LEADS_TO", "Answer",
        condition="any hit",
        guidance="Pick the first search hit and answer.",
        pitfalls="This prior skips inventory and verification.",
    )
    g.add_edge("Answer", "LEADS_TO", "End", condition="done", guidance="Stop.", pitfalls="")
    return g
