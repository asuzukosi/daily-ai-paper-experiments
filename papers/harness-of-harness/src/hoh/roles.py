"""Planner / Developer / QA role invocations with schema retries.

Contracts (paper):
  - Planner: read_only(A); emits D_t
  - Developer: sole writer of A; emits patches → A_t
  - QA: read_only(A); emits E_t given Runtime.check
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hoh.artifacts import ArtifactStore
from hoh.prompts import (
    build_developer_prompt,
    build_planner_prompt,
    build_qa_prompt,
    progressive_index,
)
from hoh.schemas import (
    DevelopmentPlan,
    EvidenceReport,
    extract_json_object,
    parse_plan_markdown,
    plan_to_markdown,
    validate_evidence,
    validate_plan,
)


LLMFn = Callable[[str, str], str]  # (system_or_role, user_prompt) -> completion text


@dataclass
class RoleResult:
    ok: bool
    payload: Any
    raw: str
    retries: int
    error: str = ""


class BaseRole(ABC):
    name: str

    def __init__(self, llm: LLMFn, *, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    @abstractmethod
    def run(self, **kwargs: Any) -> RoleResult:
        ...


class PlannerRole(BaseRole):
    name = "planner"

    def run(
        self,
        *,
        iteration: int,
        spec: str,
        evidence: Optional[Dict[str, Any]],
        store: ArtifactStore,
        readonly_root: Path,
    ) -> RoleResult:
        index = progressive_index(store.list_files())
        # Also list the frozen copy path for honesty
        prompt = build_planner_prompt(
            loop_index=iteration,
            spec=spec,
            evidence=evidence,
            artifact_index=index + f"\n(read-only copy at {readonly_root})",
        )
        raw = ""
        last_err = ""
        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt > 0:
                retry_note = (
                    f"\n\n## Schema retry ({attempt}/{self.max_retries})\n"
                    f"Previous output failed validation: {last_err}\n"
                    "Return a VALID DevelopmentPlan JSON or markdown with required fields."
                )
            raw = self.llm("planner", prompt + retry_note)
            data = extract_json_object(raw) or parse_plan_markdown(raw, iteration)
            if data is None:
                last_err = "could not parse JSON or plan markdown"
                continue
            data.setdefault("iteration", iteration)
            ok, msg, plan = validate_plan(data)
            if ok and plan is not None:
                return RoleResult(True, plan, raw, attempt)
            last_err = msg
        return RoleResult(False, None, raw, self.max_retries, last_err)


_PATCH_BLOCK = re.compile(
    r"```(?:patch|diff)?\s*path\s*=\s*([^\s`]+)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FILE_BLOCK = re.compile(
    r"```(?:file|python|py)?\s*path\s*=\s*([^\s`]+)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_file_patches(text: str) -> Dict[str, str]:
    """Extract path=… fenced full-file contents from developer output."""
    files: Dict[str, str] = {}
    for rx in (_PATCH_BLOCK, _FILE_BLOCK):
        for m in rx.finditer(text):
            rel = m.group(1).strip().strip('"').strip("'")
            body = m.group(2)
            # If it looks like a unified diff with ---/+++, skip naive apply;
            # our mock/GPU paths emit full-file bodies.
            if body.lstrip().startswith("--- ") and "\n+++" in body:
                continue
            files[rel] = body if body.endswith("\n") else body + "\n"
    return files


class DeveloperRole(BaseRole):
    name = "developer"

    def run(
        self,
        *,
        iteration: int,
        spec: str,
        plan: DevelopmentPlan,
        store: ArtifactStore,
    ) -> RoleResult:
        listing = progressive_index(store.list_files())
        # Include brief file previews for small trees
        previews = []
        for rel in store.list_files()[:12]:
            try:
                content = store.read_text(rel)
                previews.append(f"### {rel}\n```\n{content[:1500]}\n```")
            except OSError:
                pass
        prompt = build_developer_prompt(
            loop_index=iteration,
            spec=spec,
            plan=plan.to_dict(),
            artifact_listing=listing + "\n\n" + "\n".join(previews),
        )
        raw = ""
        last_err = ""
        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt > 0:
                retry_note = (
                    f"\n\n## Retry ({attempt}/{self.max_retries})\n"
                    f"Previous patches failed: {last_err}\n"
                    "Emit ```file path=REL``` or ```patch path=REL``` full-file blocks."
                )
            raw = self.llm("developer", prompt + retry_note)
            files = parse_file_patches(raw)
            if not files:
                last_err = "no path= file/patch blocks found"
                continue
            try:
                written = store.apply_full_files(files, role="developer")
            except PermissionError as e:
                return RoleResult(False, None, raw, attempt, str(e))
            except OSError as e:
                last_err = str(e)
                continue
            return RoleResult(True, {"written": written}, raw, attempt)
        return RoleResult(False, None, raw, self.max_retries, last_err)


class QATesterRole(BaseRole):
    name = "qa"

    def run(
        self,
        *,
        iteration: int,
        spec: str,
        plan: DevelopmentPlan,
        store: ArtifactStore,
        readonly_root: Path,
        runtime: Dict[str, Any],
    ) -> RoleResult:
        if readonly_root.exists():
            skip_parts = {"__pycache__", ".git", ".venv", "venv"}
            skip_suffixes = {".pyc", ".pyo", ".so", ".dylib", ".dll"}
            ro_files = []
            for p in sorted(readonly_root.rglob("*")):
                if not p.is_file():
                    continue
                if any(part in skip_parts for part in p.parts):
                    continue
                if p.suffix.lower() in skip_suffixes:
                    continue
                ro_files.append(str(p.relative_to(readonly_root)))
            listing = progressive_index(ro_files)
        else:
            listing = progressive_index(store.list_files())
        prompt = build_qa_prompt(
            loop_index=iteration,
            spec=spec,
            plan=plan.to_dict(),
            artifact_listing=listing + f"\n(frozen copy: {readonly_root})",
            runtime_check=runtime,
        )
        raw = ""
        last_err = ""
        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt > 0:
                retry_note = (
                    f"\n\n## Schema retry ({attempt}/{self.max_retries})\n"
                    f"Previous output failed validation: {last_err}\n"
                    "Return VALID EvidenceReport JSON."
                )
            raw = self.llm("qa", prompt + retry_note)
            data = extract_json_object(raw)
            if data is None:
                last_err = "could not parse JSON EvidenceReport"
                continue
            data.setdefault("iteration", iteration)
            data.setdefault("runtime_check", runtime)
            ok, msg, report = validate_evidence(data)
            if ok and report is not None:
                return RoleResult(True, report, raw, attempt)
            last_err = msg
        return RoleResult(False, None, raw, self.max_retries, last_err)


def assert_read_only_roles_cannot_write(store: ArtifactStore) -> None:
    """Sanity check used in demos / tests."""
    for role in ("planner", "qa", "runtime"):
        try:
            store.write_file("_forbidden.txt", "nope", role=role)
            raise AssertionError(f"role {role} was allowed to write")
        except PermissionError:
            pass
