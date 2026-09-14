#!/usr/bin/env python3
"""Harness-of-Harness — CPU loop demo with MockLLM (no torch).

Demonstrates Algorithm 1 over the mini_cli_todo task for T=3 iterations:
  E0=∅; for t=1..T: Planner → Developer → QA(+Runtime.check)

MockLLM emits valid planner markdown / developer file patches / QA JSON
so the schema validators and single-writer artifact store are exercised
end-to-end on CPU.

Runnable as:
  python demos/cpu_loop_demo.py
  bash experiments/run_cpu_demo.sh
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hoh.artifacts import ArtifactStore  # noqa: E402
from hoh.loop import HarnessOfHarness, LoopConfig  # noqa: E402
from hoh.metrics import paper_headline_results  # noqa: E402
from hoh.roles import assert_read_only_roles_cannot_write  # noqa: E402


TASK = _ROOT / "tasks" / "mini_cli_todo"
SPEC = (TASK / "SPEC.md").read_text(encoding="utf-8")


# Progressive implementations emitted by MockLLM Developer across iterations.
_IMPL_T1 = '''\
"""TodoStore — iteration 1: add + list (in-memory)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Todo:
    id: int
    title: str
    done: bool = False


@dataclass
class TodoStore:
    items: List[Todo] = field(default_factory=list)
    _next_id: int = 1
    path: str = "todos.json"

    def add(self, title: str) -> Todo:
        title = (title or "").strip()
        if not title:
            raise ValueError("title must be non-empty")
        t = Todo(id=self._next_id, title=title, done=False)
        self._next_id += 1
        self.items.append(t)
        return t

    def list(self) -> List[Todo]:
        return list(self.items)

    def complete(self, todo_id: int) -> Todo:
        raise NotImplementedError("complete not implemented yet")

    def remove(self, todo_id: int) -> None:
        raise NotImplementedError("remove not implemented yet")

    def save(self) -> None:
        raise NotImplementedError("save not implemented yet")

    def load(self) -> None:
        raise NotImplementedError("load not implemented yet")


def main(argv: Optional[List[str]] = None) -> int:
    import sys as _sys
    argv = list(argv if argv is not None else _sys.argv[1:])
    store = TodoStore()
    if not argv or argv[0] == "list":
        for t in store.list():
            print(f"[{t.id}] {'x' if t.done else ' '} {t.title}")
        return 0
    if argv[0] == "add":
        store.add(" ".join(argv[1:]))
        return 0
    print("usage: add|list|complete|remove", file=_sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

_IMPL_T2 = '''\
"""TodoStore — iteration 2: complete + remove."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Todo:
    id: int
    title: str
    done: bool = False


@dataclass
class TodoStore:
    items: List[Todo] = field(default_factory=list)
    _next_id: int = 1
    path: str = "todos.json"

    def add(self, title: str) -> Todo:
        title = (title or "").strip()
        if not title:
            raise ValueError("title must be non-empty")
        t = Todo(id=self._next_id, title=title, done=False)
        self._next_id += 1
        self.items.append(t)
        return t

    def list(self) -> List[Todo]:
        return list(self.items)

    def _find(self, todo_id: int) -> Todo:
        for t in self.items:
            if t.id == todo_id:
                return t
        raise KeyError(f"todo id {todo_id} not found")

    def complete(self, todo_id: int) -> Todo:
        t = self._find(todo_id)
        t.done = True
        return t

    def remove(self, todo_id: int) -> None:
        t = self._find(todo_id)
        self.items.remove(t)

    def save(self) -> None:
        raise NotImplementedError("save not implemented yet")

    def load(self) -> None:
        raise NotImplementedError("load not implemented yet")


def main(argv: Optional[List[str]] = None) -> int:
    import sys as _sys
    argv = list(argv if argv is not None else _sys.argv[1:])
    store = TodoStore()
    if not argv or argv[0] == "list":
        for t in store.list():
            print(f"[{t.id}] {'x' if t.done else ' '} {t.title}")
        return 0
    if argv[0] == "add":
        store.add(" ".join(argv[1:]))
        return 0
    if argv[0] == "complete":
        store.complete(int(argv[1]))
        return 0
    if argv[0] == "remove":
        store.remove(int(argv[1]))
        return 0
    print("usage: add|list|complete|remove", file=_sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

_IMPL_T3 = '''\
"""TodoStore — iteration 3: JSON persistence + full CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Todo:
    id: int
    title: str
    done: bool = False


@dataclass
class TodoStore:
    items: List[Todo] = field(default_factory=list)
    _next_id: int = 1
    path: str = "todos.json"

    def add(self, title: str) -> Todo:
        title = (title or "").strip()
        if not title:
            raise ValueError("title must be non-empty")
        t = Todo(id=self._next_id, title=title, done=False)
        self._next_id += 1
        self.items.append(t)
        return t

    def list(self) -> List[Todo]:
        return list(self.items)

    def _find(self, todo_id: int) -> Todo:
        for t in self.items:
            if t.id == todo_id:
                return t
        raise KeyError(f"todo id {todo_id} not found")

    def complete(self, todo_id: int) -> Todo:
        t = self._find(todo_id)
        t.done = True
        return t

    def remove(self, todo_id: int) -> None:
        t = self._find(todo_id)
        self.items.remove(t)

    def save(self) -> None:
        payload = {
            "next_id": self._next_id,
            "items": [asdict(t) for t in self.items],
        }
        Path(self.path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            self.items = []
            self._next_id = 1
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            self.items = [Todo(**x) for x in data]
            self._next_id = max((t.id for t in self.items), default=0) + 1
        else:
            self.items = [Todo(**x) for x in data.get("items", [])]
            self._next_id = int(data.get("next_id", max((t.id for t in self.items), default=0) + 1))


def main(argv: Optional[List[str]] = None) -> int:
    import sys as _sys
    argv = list(argv if argv is not None else _sys.argv[1:])
    store = TodoStore()
    try:
        store.load()
    except Exception:
        pass
    if not argv or argv[0] == "list":
        for t in store.list():
            print(f"[{t.id}] {'x' if t.done else ' '} {t.title}")
        return 0
    if argv[0] == "add":
        store.add(" ".join(argv[1:]))
        store.save()
        return 0
    if argv[0] == "complete":
        store.complete(int(argv[1]))
        store.save()
        return 0
    if argv[0] == "remove":
        store.remove(int(argv[1]))
        store.save()
        return 0
    print("usage: add|list|complete|remove", file=_sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


class MockLLM:
    """Deterministic role simulator — no network / no torch."""

    def __init__(self) -> None:
        self.calls = 0
        self._plan_n = 0
        self._dev_n = 0
        self._qa_n = 0

    def __call__(self, role: str, prompt: str) -> str:
        self.calls += 1
        role = role.lower().strip()
        if role == "planner":
            return self._plan(prompt)
        if role == "developer":
            return self._dev(prompt)
        if role == "qa":
            return self._qa(prompt)
        return f"unknown role {role}"

    def _plan(self, prompt: str) -> str:
        self._plan_n += 1
        t = self._plan_n
        plans = {
            1: dict(
                objective="Implement add() and list() with non-empty title validation",
                scope=["Implement TodoStore.add", "Keep TodoStore.list working", "Stub CLI list/add"],
                preserve=["Module imports", "Todo dataclass fields"],
                acceptance_criteria=[
                    "add returns Todo with unique id",
                    "empty title raises",
                    "list returns added items",
                ],
                rationale="First increment: core create/read path (capability growth).",
                repair_vs_growth="growth",
            ),
            2: dict(
                objective="Implement complete() and remove() with missing-id errors",
                scope=["TodoStore.complete", "TodoStore.remove", "CLI complete/remove"],
                preserve=["add/list behavior from iteration 1"],
                acceptance_criteria=[
                    "complete marks done",
                    "remove deletes item",
                    "missing id raises",
                ],
                rationale="Grow mutation APIs while preserving add/list (balanced).",
                repair_vs_growth="balanced",
            ),
            3: dict(
                objective="Add JSON persistence (save/load) and wire CLI to persist",
                scope=["TodoStore.save", "TodoStore.load", "CLI save after mutations"],
                preserve=["add/list/complete/remove semantics"],
                acceptance_criteria=[
                    "save writes todos.json",
                    "load restores items",
                    "CLI list returns 0 on empty store",
                ],
                rationale="Close remaining SPEC gaps: persistence + CLI stability.",
                repair_vs_growth="growth",
            ),
        }
        p = plans.get(t, plans[3])
        # Emit markdown (valid planner form) — also include JSON for robustness
        md = f"""# Development Plan — iteration {t}

**Objective:** {p['objective']}

**Repair vs growth:** {p['repair_vs_growth']}

## Scope
"""
        for s in p["scope"]:
            md += f"- {s}\n"
        md += "\n## Preserve\n"
        for s in p["preserve"]:
            md += f"- {s}\n"
        md += "\n## Acceptance criteria\n"
        for s in p["acceptance_criteria"]:
            md += f"- {s}\n"
        md += f"\n## Rationale\n{p['rationale']}\n"
        md += "\n```json\n" + json.dumps({"iteration": t, **p}, indent=2) + "\n```\n"
        return md

    def _dev(self, prompt: str) -> str:
        self._dev_n += 1
        impls = {1: _IMPL_T1, 2: _IMPL_T2, 3: _IMPL_T3}
        body = impls.get(self._dev_n, _IMPL_T3)
        return f"Implementing increment {self._dev_n}.\n\n```file path=todo.py\n{body}```\n"

    def _qa(self, prompt: str) -> str:
        self._qa_n += 1
        t = self._qa_n
        # Infer runtime ok from prompt blob
        runtime_ok = '"ok": true' in prompt.lower() or '"ok": true' in prompt
        # Also look for compile ok
        if "py_compile" in prompt and '"ok": false' in prompt:
            # still may partially pass criteria for implemented bits
            pass

        # Progressive: more criteria pass as iterations grow
        if t == 1:
            criteria = [
                {"criterion": "add returns Todo with unique id", "status": "pass", "detail": "exercised via Runtime/import"},
                {"criterion": "empty title raises", "status": "pass", "detail": "ValueError path present"},
                {"criterion": "list returns added items", "status": "pass", "detail": "list implemented"},
                {"criterion": "complete/remove", "status": "fail", "detail": "NotImplementedError"},
                {"criterion": "persistence", "status": "fail", "detail": "not yet"},
            ]
            gaps = ["complete/remove not implemented", "persistence missing"]
            passed = False
        elif t == 2:
            criteria = [
                {"criterion": "complete marks done", "status": "pass", "detail": "ok"},
                {"criterion": "remove deletes item", "status": "pass", "detail": "ok"},
                {"criterion": "missing id raises", "status": "pass", "detail": "KeyError"},
                {"criterion": "add/list preserved", "status": "pass", "detail": "regression check"},
                {"criterion": "persistence", "status": "fail", "detail": "still NotImplemented"},
            ]
            gaps = ["JSON save/load missing"]
            passed = False
        else:
            criteria = [
                {"criterion": "save writes todos.json", "status": "pass", "detail": "ok"},
                {"criterion": "load restores items", "status": "pass", "detail": "ok"},
                {"criterion": "CLI list returns 0", "status": "pass", "detail": "ok"},
                {"criterion": "full SPEC coverage", "status": "pass", "detail": "add/list/complete/remove/persist"},
            ]
            gaps = []
            passed = True

        report = {
            "iteration": t,
            "passed": passed and runtime_ok if t >= 3 else passed,
            "criteria_results": criteria,
            "gaps": gaps,
            "regressions": [],
            "runtime_check": {"ok": runtime_ok, "note": "echoed from Runtime.check"},
            "notes": f"Mock QA for iteration {t}; independent of Developer claims.",
        }
        # Force overall passed true at t3 when tests likely green
        if t >= 3:
            report["passed"] = True
        return "```json\n" + json.dumps(report, indent=2) + "\n```\n"


def main() -> int:
    print("=" * 68)
    print("Harness-of-Harness — CPU loop demo (MockLLM)")
    print("Paper: Multi-Day Autonomous Software Development (arXiv:2609.01481)")
    print("=" * 68)

    # Single-writer sanity
    with tempfile.TemporaryDirectory(prefix="hoh_sw_") as td:
        probe = ArtifactStore(Path(td))
        assert_read_only_roles_cannot_write(probe)
        print("\n[ok] single-writer: planner/qa cannot mutate artifact")

    out_root = _ROOT / "artifacts" / "cpu_demo"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(out_root)
    store.seed_from(TASK / "starter")

    # Copy tests alongside workspace so Runtime.check can discover them
    tests_src = TASK / "tests"
    tests_dst = store.workspace / "tests"
    if tests_dst.exists():
        shutil.rmtree(tests_dst)
    shutil.copytree(tests_src, tests_dst)

    llm = MockLLM()
    cfg = LoopConfig(
        T=3,
        max_schema_retries=2,
        primary_module="todo.py",
        tests_dir=store.workspace / "tests",
        snapshot_each=True,
    )
    hoh = HarnessOfHarness(spec=SPEC, store=store, llm=llm, config=cfg)
    result = hoh.run()

    print(f"\nIterations completed: {len(result.metrics.iterations)}")
    print(f"LLM calls: {llm.calls}")
    for m in result.metrics.iterations:
        print(
            f"  t={m.iteration}: plan={m.plan_ok} dev={m.developer_ok} "
            f"qa_passed={m.qa_passed} runtime_ok={m.runtime_ok} "
            f"criteria={m.n_criteria_pass}/{m.n_criteria} "
            f"gaps={m.n_gaps} schema_retries={m.schema_retries}"
        )

    summary = result.metrics.summary()
    print("\n" + "-" * 68)
    print(f"Final criteria pass rate: {summary['final_pass_rate']:.1f}%")
    print(f"Artifacts: {out_root}")
    print(f"Final workspace: {result.final_workspace}")
    print("-" * 68)

    # Show paper headlines for context
    paper = paper_headline_results()
    print(
        f"\nPaper headline (real): avg relative gain {paper['avg_relative_gain_pct']}% "
        f"(max {paper['max_relative_gain_pct']}%) after 3 iterations vs standalone harnesses."
    )
    print(
        "GameCraft Overall: Codex 49.58→71.52; OpenCode 26.90→48.98; Pi 42.16→58.78"
    )
    print(
        "\nTakeaway: Planner scopes a small increment; Developer is sole writer; "
        "QA evaluates a frozen read-only A_t with Runtime.check — Algorithm 1."
    )

    # Success gate: 3 iterations, final QA passed, runtime ok on last
    last = result.metrics.iterations[-1] if result.metrics.iterations else None
    if last and last.plan_ok and last.developer_ok and last.qa_passed and last.runtime_ok:
        print("\nCPU demo PASSED")
        return 0
    print("\nCPU demo FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
