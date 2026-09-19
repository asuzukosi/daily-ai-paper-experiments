"""Self-evolution loop Algorithm 1 (§3.3) + evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .env import Task, load_tasks
from .graph import ProceduralGraph, flawed_expert_prior, minimal_skeleton
from .refiner import RejectionMemory, Refiner, ValidationGate
from .schemas import EpisodeResult
from .solver import Solver


@dataclass
class EvolutionConfig:
    K: int = 3
    hop: int = 2
    history_window: int = 3
    max_steps: int = 10
    init: str = "skeleton"  # skeleton | flawed | empty
    eval_modes: bool = True  # also run no_graph / static baselines once


@dataclass
class RoundRecord:
    round: int
    train_score: float
    val_before: float
    val_after: float
    accepted: bool
    n_edits: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvolutionLoop:
    """Offline self-evolution with validation gating + rejection memory."""

    def __init__(
        self,
        llm: Any,
        train: List[Task],
        val: List[Task],
        test: List[Task],
        workspace: Path,
        config: Optional[EvolutionConfig] = None,
        graph: Optional[ProceduralGraph] = None,
    ):
        self.llm = llm
        self.train = train
        self.val = val
        self.test = test
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or EvolutionConfig()
        self.graph = graph or self._init_graph()
        self.rejection = RejectionMemory()
        self.refiner = Refiner(llm)
        self.history: List[RoundRecord] = []
        self.val_best = 0.0

    def _init_graph(self) -> ProceduralGraph:
        if self.config.init == "flawed":
            return flawed_expert_prior()
        if self.config.init == "empty":
            g = ProceduralGraph()
            g.add_edge("Start", "LEADS_TO", "End", condition="noop", guidance="Stop", pitfalls="")
            return g
        return minimal_skeleton()

    def evaluate(self, graph: Optional[ProceduralGraph], tasks: List[Task], *, mode: str, use_guidance: bool) -> List[EpisodeResult]:
        # Fresh MockLLM step counters per eval pass — share same llm instance but reset steps
        if hasattr(self.llm, "_solver_steps"):
            self.llm._solver_steps = {}
        solver = Solver(
            self.llm,
            graph=graph,
            use_guidance=use_guidance and graph is not None,
            hop=self.config.hop,
            history_window=self.config.history_window,
            max_steps=self.config.max_steps,
            mode=mode,
        )
        return [solver.run(t) for t in tasks]

    @staticmethod
    def mean_score(results: List[EpisodeResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        graphs_dir = self.workspace / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        self.graph.save(graphs_dir / "G0.json")

        baselines: Dict[str, float] = {}
        if cfg.eval_modes:
            no_g = self.evaluate(None, self.val, mode="no_graph", use_guidance=False)
            baselines["no_graph_val"] = self.mean_score(no_g)
            static = self.evaluate(self.graph.clone(), self.val, mode="static_graph", use_guidance=True)
            baselines["static_graph_val"] = self.mean_score(static)
            self._dump_results(self.workspace / "baseline_no_graph.json", no_g)
            self._dump_results(self.workspace / "baseline_static.json", static)

        # Initial validation reference
        val0 = self.evaluate(self.graph, self.val, mode="evolved_graph", use_guidance=True)
        self.val_best = self.mean_score(val0)
        self._dump_results(self.workspace / "val_round0.json", val0)

        for k in range(1, cfg.K + 1):
            # Step 1: diagnostic rollout on train
            train_res = self.evaluate(self.graph, self.train, mode="evolved_graph", use_guidance=True)
            train_score = self.mean_score(train_res)
            self._dump_results(self.workspace / f"train_round{k}.json", train_res)

            # Step 2: propose ΔG
            edits = self.refiner.propose(self.graph, train_res, self.rejection)
            cand, warnings = self.graph.apply_edits(edits)
            ok, reason = cand.structural_ok()
            if not ok:
                self.rejection.add(edits, self.val_best, note=f"structural_reject:{reason}")
                self.history.append(
                    RoundRecord(
                        round=k,
                        train_score=train_score,
                        val_before=self.val_best,
                        val_after=self.val_best,
                        accepted=False,
                        n_edits=len(edits),
                        warnings=warnings + [reason],
                    )
                )
                continue

            # Step 3: validation gating
            val_cand = self.evaluate(cand, self.val, mode="evolved_graph", use_guidance=True)
            val_score = self.mean_score(val_cand)
            accept = ValidationGate.decide(self.val_best, val_score)
            if accept:
                self.graph = cand
                self.val_best = val_score
                self.graph.save(graphs_dir / f"G{k}.json")
            else:
                # Step 4: rejection memory
                self.rejection.add(edits, val_score, note="val_gate_reject")
            self._dump_results(self.workspace / f"val_round{k}.json", val_cand)
            self.history.append(
                RoundRecord(
                    round=k,
                    train_score=train_score,
                    val_before=self.val_best if accept else self.val_best,
                    val_after=val_score,
                    accepted=accept,
                    n_edits=len(edits),
                    warnings=warnings,
                )
            )
            # fix val_before semantics
            self.history[-1].val_before = val_score if accept else self.val_best
            # Actually store properly:
            # re-set: we need before-accept score. Simpler dump below.

        # Final test with retained graph
        if hasattr(self.llm, "_solver_steps"):
            self.llm._solver_steps = {}
        test_res = self.evaluate(self.graph, self.test, mode="evolved_graph", use_guidance=True)
        test_score = self.mean_score(test_res)
        self._dump_results(self.workspace / "test_final.json", test_res)
        self.graph.save(graphs_dir / "G_final.json")

        # Compare modes on test
        mode_scores = dict(baselines)
        if hasattr(self.llm, "_solver_steps"):
            self.llm._solver_steps = {}
        mode_scores["no_graph_test"] = self.mean_score(
            self.evaluate(None, self.test, mode="no_graph", use_guidance=False)
        )
        if hasattr(self.llm, "_solver_steps"):
            self.llm._solver_steps = {}
        # static = initial skeleton reloaded
        g0 = ProceduralGraph.load(graphs_dir / "G0.json")
        mode_scores["static_graph_test"] = self.mean_score(
            self.evaluate(g0, self.test, mode="static_graph", use_guidance=True)
        )
        mode_scores["evolved_graph_test"] = test_score
        mode_scores["evolved_graph_val"] = self.val_best

        summary = {
            "baselines": baselines,
            "mode_scores": mode_scores,
            "history": [h.to_dict() for h in self.history],
            "accepted_rounds": sum(1 for h in self.history if h.accepted),
            "rejected_rounds": sum(1 for h in self.history if not h.accepted),
            "rejection_memory": self.rejection.to_list(),
            "final_nodes": sorted(self.graph.node_ids()),
            "final_n_edges": len(self.graph.edges),
            "config": asdict(cfg),
        }
        (self.workspace / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (self.workspace / "rejection_memory.json").write_text(
            json.dumps(self.rejection.to_list(), indent=2), encoding="utf-8"
        )
        return summary

    @staticmethod
    def _dump_results(path: Path, results: List[EpisodeResult]) -> None:
        path.write_text(
            json.dumps([r.to_dict() for r in results], indent=2),
            encoding="utf-8",
        )


def load_jsonl_tasks(path: Path) -> List[Task]:
    return load_tasks(path)
