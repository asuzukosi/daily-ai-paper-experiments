#!/usr/bin/env python3
"""Procedural Graphs — CPU demo with MockLLM (no torch).

Compares no-graph / static-graph / evolved-graph on tasks/mini_shop and runs
K=3 self-evolution with validation gating + rejection memory.

  python demos/cpu_pg_demo.py
  bash experiments/run_cpu_demo.sh
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from procedural_graphs.loop import EvolutionConfig, EvolutionLoop, load_jsonl_tasks  # noqa: E402
from procedural_graphs.metrics import paper_headline_results  # noqa: E402
from procedural_graphs.runtime import MockLLM  # noqa: E402

TASK = _ROOT / "tasks" / "mini_shop"
OUT = _ROOT / "artifacts" / "cpu_demo"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    train = load_jsonl_tasks(TASK / "train.jsonl")
    val = load_jsonl_tasks(TASK / "val.jsonl")
    test = load_jsonl_tasks(TASK / "test.jsonl")

    llm = MockLLM()
    loop = EvolutionLoop(
        llm=llm,
        train=train,
        val=val,
        test=test,
        workspace=OUT,
        config=EvolutionConfig(K=3, init="skeleton", eval_modes=True),
    )
    summary = loop.run()
    ms = summary["mode_scores"]

    print("=" * 60)
    print("Procedural Graphs CPU demo — MockLLM, K=3, mini_shop")
    print("=" * 60)
    print(f"workspace: {OUT}")
    print(f"no_graph_val:      {ms.get('no_graph_val', 'n/a')}")
    print(f"static_graph_val:  {ms.get('static_graph_val', 'n/a')}")
    print(f"evolved_graph_val: {ms.get('evolved_graph_val', 'n/a')}")
    print(f"no_graph_test:     {ms.get('no_graph_test', 'n/a')}")
    print(f"static_graph_test: {ms.get('static_graph_test', 'n/a')}")
    print(f"evolved_graph_test:{ms.get('evolved_graph_test', 'n/a')}")
    print(f"accepted/rejected rounds: {summary['accepted_rounds']}/{summary['rejected_rounds']}")
    print(f"final nodes: {summary['final_nodes']}")
    print(f"final n_edges: {summary['final_n_edges']}")
    for h in summary["history"]:
        print(
            f"  round {h['round']}: train={h['train_score']:.3f} "
            f"val_after={h['val_after']:.3f} "
            f"{'ACCEPT' if h['accepted'] else 'reject'} edits={h['n_edits']}"
        )
    print()
    print("Paper headline citations (not mini-bench):")
    print(json.dumps(paper_headline_results()["highlights"], indent=2))
    print()
    print(f"Artifacts under {OUT}")
    assert (OUT / "run_summary.json").exists()
    assert (OUT / "graphs" / "G_final.json").exists()
    # Evolved should beat or match no-graph on test for MockLLM scripted path
    assert ms["evolved_graph_test"] >= ms["no_graph_test"] - 1e-9
    print("\nCPU demo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
