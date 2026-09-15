"""Algorithm 1 — Harness-of-Harness outer loop.

  E0 ← ∅
  for t = 1..T:
    Dt ← ProjectPlanner(S, E_{t-1}; read_only(A_{t-1}))
    At ← Developer(A_{t-1}; S, Dt)
    Et ← QATester(read_only(At); S, Dt, Runtime.check(At))
  return A_T
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hoh.artifacts import ArtifactStore
from hoh.metrics import IterationMetrics, RunMetrics
from hoh.roles import DeveloperRole, LLMFn, PlannerRole, QATesterRole
from hoh.runtime import runtime_check
from hoh.schemas import DevelopmentPlan, EvidenceReport, plan_to_markdown


@dataclass
class LoopConfig:
    T: int = 3
    max_schema_retries: int = 3
    primary_module: str = "todo.py"
    tests_dir: Optional[Path] = None
    snapshot_each: bool = True


@dataclass
class LoopResult:
    final_workspace: Path
    metrics: RunMetrics
    plans: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class HarnessOfHarness:
    """Outer HoH loop wrapping an LLM-backed coding harness (roles)."""

    def __init__(
        self,
        *,
        spec: str,
        store: ArtifactStore,
        llm: LLMFn,
        config: Optional[LoopConfig] = None,
    ):
        self.spec = spec
        self.store = store
        self.config = config or LoopConfig()
        self.planner = PlannerRole(llm, max_retries=self.config.max_schema_retries)
        self.developer = DeveloperRole(llm, max_retries=self.config.max_schema_retries)
        self.qa = QATesterRole(llm, max_retries=self.config.max_schema_retries)

    def run(self) -> LoopResult:
        cfg = self.config
        metrics = RunMetrics()
        plans: List[Dict[str, Any]] = []
        evidence_list: List[Dict[str, Any]] = []

        # E0 ← ∅
        evidence: Optional[Dict[str, Any]] = None
        if cfg.snapshot_each:
            self.store.snapshot(0, label="A0_initial")

        for t in range(1, cfg.T + 1):
            schema_retries = 0

            # Planner — read_only(A_{t-1})
            with tempfile.TemporaryDirectory(prefix=f"hoh_plan_ro_{t}_") as tmp:
                ro = self.store.read_only_copy(Path(tmp) / "A")
                plan_res = self.planner.run(
                    iteration=t,
                    spec=self.spec,
                    evidence=evidence,
                    store=self.store,
                    readonly_root=ro,
                )
            schema_retries += plan_res.retries
            if not plan_res.ok:
                metrics.add(
                    IterationMetrics(
                        iteration=t,
                        plan_ok=False,
                        developer_ok=False,
                        qa_passed=False,
                        runtime_ok=False,
                        schema_retries=schema_retries,
                        notes=f"planner failed: {plan_res.error}",
                    )
                )
                break
            plan: DevelopmentPlan = plan_res.payload
            plan_dict = plan.to_dict()
            self.store.save_plan(t, plan_dict)
            # Also persist markdown form for progressive disclosure
            (self.store.plans_dir / f"D_{t}.md").write_text(
                plan_to_markdown(plan), encoding="utf-8"
            )
            plans.append(plan_dict)

            # Developer — sole writer
            dev_res = self.developer.run(
                iteration=t, spec=self.spec, plan=plan, store=self.store
            )
            schema_retries += dev_res.retries
            if not dev_res.ok:
                metrics.add(
                    IterationMetrics(
                        iteration=t,
                        plan_ok=True,
                        developer_ok=False,
                        qa_passed=False,
                        runtime_ok=False,
                        schema_retries=schema_retries,
                        notes=f"developer failed: {dev_res.error}",
                    )
                )
                break
            if cfg.snapshot_each:
                self.store.snapshot(t, label=f"A{t}_after_developer")

            # Runtime.check(A_t)
            rt = runtime_check(
                self.store.workspace,
                tests_dir=cfg.tests_dir,
                primary_module=cfg.primary_module,
            )

            # QA — read_only(A_t)
            with tempfile.TemporaryDirectory(prefix=f"hoh_qa_ro_{t}_") as tmp:
                ro = self.store.read_only_copy(Path(tmp) / "A")
                qa_res = self.qa.run(
                    iteration=t,
                    spec=self.spec,
                    plan=plan,
                    store=self.store,
                    readonly_root=ro,
                    runtime=rt,
                )
            schema_retries += qa_res.retries
            if not qa_res.ok:
                metrics.add(
                    IterationMetrics(
                        iteration=t,
                        plan_ok=True,
                        developer_ok=True,
                        qa_passed=False,
                        runtime_ok=bool(rt.get("ok")),
                        schema_retries=schema_retries,
                        notes=f"qa schema failed: {qa_res.error}",
                    )
                )
                break
            report: EvidenceReport = qa_res.payload
            ev = report.to_dict()
            self.store.save_evidence(t, ev)
            evidence_list.append(ev)
            evidence = ev  # E_t → next Planner

            n_crit = len(report.criteria_results)
            n_pass = sum(
                1
                for c in report.criteria_results
                if str(c.get("status", "")).lower() in ("pass", "passed", "ok", "true")
            )
            metrics.add(
                IterationMetrics(
                    iteration=t,
                    plan_ok=True,
                    developer_ok=True,
                    qa_passed=bool(report.passed),
                    runtime_ok=bool(rt.get("ok")),
                    n_criteria=n_crit,
                    n_criteria_pass=n_pass,
                    n_gaps=len(report.gaps),
                    schema_retries=schema_retries,
                    notes=report.notes[:200],
                )
            )

        # Persist run summary
        summary_path = self.store.root / "run_summary.json"
        summary_path.write_text(
            json.dumps(metrics.summary(), indent=2), encoding="utf-8"
        )

        return LoopResult(
            final_workspace=self.store.workspace,
            metrics=metrics,
            plans=plans,
            evidence=evidence_list,
            history=list(self.store.history),
        )
