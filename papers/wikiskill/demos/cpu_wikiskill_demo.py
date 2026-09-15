#!/usr/bin/env python3
"""WikiSkill — CPU demo with MockLLM (no torch).

Runs K=3 evolutionary iterations on tasks/mini_livemath:
  Inference → Wiki Maintainer → Skill Proposer → Gating & Rollback

MockLLM is scripted so format / arithmetic / unit failure modes clear as
skills are accepted — wiki patterns and skills visibly evolve under
artifacts/cpu_demo/.

Runnable as:
  python demos/cpu_wikiskill_demo.py
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

from wikiskill.loop import LoopConfig, WikiSkillLoop, load_jsonl  # noqa: E402
from wikiskill.metrics import paper_headline_results  # noqa: E402
from wikiskill.runtime import MockLLM  # noqa: E402


TASK = _ROOT / "tasks" / "mini_livemath"
OUT = _ROOT / "artifacts" / "cpu_demo"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    train = load_jsonl(TASK / "train.jsonl")
    val = load_jsonl(TASK / "val.jsonl")
    test = load_jsonl(TASK / "test.jsonl")

    llm = MockLLM()
    loop = WikiSkillLoop(
        workspace=OUT,
        llm=llm,
        train=train,
        val=val,
        test=test,
        config=LoopConfig(K=3, eval_test_each_iter=True),
    )
    result = loop.run()
    m = result.metrics

    print("=" * 60)
    print("WikiSkill CPU demo — MockLLM, K=3, mini_livemath")
    print("=" * 60)
    print(f"workspace: {result.workspace}")
    print(f"baseline val (iter 0, empty skills): {result.history[0]['val_acc']:.3f}")
    for it in m.iterations:
        ta = f"{it.test_acc:.3f}" if it.test_acc is not None else "n/a"
        print(
            f"  iter {it.iteration}: train={it.train_acc:.3f} val={it.val_acc:.3f} "
            f"test={ta} skills={it.n_skills} patterns={it.n_patterns} "
            f"gate={'ACCEPT' if it.gate_accepted else 'reject'} "
            f"proposal={it.proposal_skill}"
        )
    if m.final_test_acc is not None:
        print(f"final test acc: {m.final_test_acc:.3f}")
    print(f"accepted/rejected proposals: {m.accepted}/{m.rejected}")
    print(f"active skills: {loop.skills.list_skills()}")
    print(f"wiki patterns: {loop.wiki.list_patterns()}")
    print()
    print("Paper Table 1 averages (citation only):")
    print(json.dumps(paper_headline_results()["table1_avg"], indent=2))
    print()
    print(f"Artifacts written under {OUT}")
    print("  raw/  wiki/  skills/  run_summary.json")

    assert m.accepted >= 1, "expected at least one accepted skill"
    assert loop.wiki.n_patterns() >= 1, "expected wiki patterns"
    assert (OUT / "run_summary.json").exists()
    assert (OUT / "wiki" / "logs.md").exists()
    assert (OUT / "wiki" / "skill-impact.md").exists()
    print("\nCPU demo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
