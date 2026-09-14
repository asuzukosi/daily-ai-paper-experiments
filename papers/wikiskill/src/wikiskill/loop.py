"""WikiSkill outer evolutionary loop (Algorithm 1 / §3.2).

Each iteration k:
  1. Inference Agent  — rollouts on D_train with S_{k-1}; write raw/
  2. Wiki Maintainer  — consolidate sampled traces → wiki/ (never reset)
  3. Skill Proposer   — ReAct-style atomic create/update from wiki + outcomes
  4. Gating & Rollback — eval on D_val; accept iff R > R_best; wiki always kept
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from wikiskill.gating import GatingController
from wikiskill.inference import InferenceAgent
from wikiskill.maintainer import WikiMaintainer
from wikiskill.metrics import RunMetrics, accuracy
from wikiskill.proposer import SkillProposer
from wikiskill.schemas import IterationSummary, TaskExample
from wikiskill.skills import SkillStore
from wikiskill.traces import RawStore
from wikiskill.wiki import WikiStore


@dataclass
class LoopConfig:
    K: int = 3
    max_fail_sample: int = 6
    max_pass_sample: int = 4
    eval_test_each_iter: bool = True
    early_stop_at_perfect_val: bool = True


@dataclass
class LoopResult:
    workspace: Path
    metrics: RunMetrics
    history: List[Dict[str, Any]] = field(default_factory=list)


def load_jsonl(path: Path) -> List[TaskExample]:
    items: List[TaskExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(TaskExample.from_dict(json.loads(line)))
    return items


class WikiSkillLoop:
    """Orchestrates the four-component WikiSkill iteration."""

    def __init__(
        self,
        *,
        workspace: Path,
        llm: Any,
        train: List[TaskExample],
        val: List[TaskExample],
        test: Optional[List[TaskExample]] = None,
        config: Optional[LoopConfig] = None,
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.train = train
        self.val = val
        self.test = test or []
        self.config = config or LoopConfig()

        self.raw = RawStore(self.workspace / "raw")
        self.wiki = WikiStore(self.workspace / "wiki")
        self.skills = SkillStore(self.workspace / "skills")

        self.inference = InferenceAgent(llm, self.skills, self.raw)
        self.maintainer = WikiMaintainer(llm, self.wiki, self.raw)
        self.proposer = SkillProposer(llm, self.wiki, self.skills, self.raw)
        self.gating = GatingController(self.skills, self.wiki, self.inference)

    def run(self) -> LoopResult:
        cfg = self.config
        metrics = RunMetrics()
        history: List[Dict[str, Any]] = []

        baseline = self.gating.initialize_baseline(self.val, iteration=0)
        history.append(
            {
                "iteration": 0,
                "phase": "baseline",
                "val_acc": baseline,
                "n_skills": self.skills.n_skills(),
                "n_patterns": self.wiki.n_patterns(),
            }
        )

        for k in range(1, cfg.K + 1):
            train_traces = self.inference.rollout(
                self.train, iteration=k, split="train", allow_wiki=False
            )
            train_acc = accuracy(train_traces)

            patterns = self.maintainer.run(
                iteration=k,
                max_fail=cfg.max_fail_sample,
                max_pass=cfg.max_pass_sample,
            )

            proposal = self.proposer.run(iteration=k)

            decision, val_traces = self.gating.evaluate_and_gate(
                proposal, self.val, iteration=k
            )
            val_acc = accuracy(val_traces)

            test_acc: Optional[float] = None
            if cfg.eval_test_each_iter and self.test:
                test_traces = self.inference.rollout(
                    self.test, iteration=k, split="test", allow_wiki=False
                )
                test_acc = accuracy(test_traces)

            summary = IterationSummary(
                iteration=k,
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc,
                n_patterns=self.wiki.n_patterns(),
                n_skills=self.skills.n_skills(),
                gate_accepted=decision.accepted,
                proposal_skill=proposal.skill_name or "(noop)",
                notes=decision.reason,
            )
            metrics.add(summary)
            history.append(
                {
                    **summary.to_dict(),
                    "proposal": proposal.to_dict(),
                    "patterns_updated": patterns,
                    "best_val": self.gating.best_score,
                }
            )

            if cfg.early_stop_at_perfect_val and self.gating.best_score >= 1.0 - 1e-9:
                break

        if self.test:
            final_test = self.inference.rollout(
                self.test, iteration=cfg.K + 1, split="test", allow_wiki=False
            )
            metrics.final_test_acc = accuracy(final_test)
        else:
            metrics.final_test_acc = None

        payload = {
            "metrics": metrics.to_dict(),
            "history": history,
            "skills": self.skills.list_skills(),
            "wiki": self.wiki.snapshot_summary(),
            "baseline_val": baseline,
        }
        (self.workspace / "run_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return LoopResult(workspace=self.workspace, metrics=metrics, history=history)
