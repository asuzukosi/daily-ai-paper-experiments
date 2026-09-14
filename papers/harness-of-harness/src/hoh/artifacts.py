"""Versioned artifact store with single-writer discipline.

Only Developer may call write APIs. Planner/QA get read-only snapshots
(copies) of the workspace at the appropriate loop points.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ArtifactStore:
    """Filesystem-backed A_t with iteration snapshots under versions/."""

    root: Path
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.workspace = self.root / "workspace"
        self.versions = self.root / "versions"
        self.evidence_dir = self.root / "evidence"
        self.plans_dir = self.root / "plans"
        for d in (self.workspace, self.versions, self.evidence_dir, self.plans_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- read APIs (Planner / QA) ----

    def list_files(self, *, relative_to_workspace: bool = True) -> List[str]:
        files: List[str] = []
        if not self.workspace.exists():
            return files
        skip_parts = {"__pycache__", ".git", ".venv", "venv"}
        skip_suffixes = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".bin", ".pt", ".safetensors"}
        for p in sorted(self.workspace.rglob("*")):
            if not p.is_file():
                continue
            if any(part in skip_parts for part in p.parts):
                continue
            if p.suffix.lower() in skip_suffixes:
                continue
            files.append(
                str(p.relative_to(self.workspace)) if relative_to_workspace else str(p)
            )
        return files

    def read_text(self, rel_path: str) -> str:
        path = self.workspace / rel_path
        return path.read_text(encoding="utf-8", errors="replace")

    def read_only_copy(self, dest: Path) -> Path:
        """Materialize a frozen read-only copy of the workspace for Planner/QA."""
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        if self.workspace.exists() and any(self.workspace.iterdir()):
            shutil.copytree(self.workspace, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        # Best-effort read-only bits (still a copy; process can chmod back)
        for p in dest.rglob("*"):
            try:
                if p.is_file():
                    p.chmod(0o444)
                elif p.is_dir():
                    p.chmod(0o555)
            except OSError:
                pass
        return dest

    # ---- write APIs (Developer only) ----

    def write_file(self, rel_path: str, content: str, *, role: str) -> None:
        self._assert_developer(role)
        path = self.workspace / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def apply_full_files(self, files: Dict[str, str], *, role: str) -> List[str]:
        """Write multiple full-file contents. Returns list of relative paths written."""
        self._assert_developer(role)
        written = []
        for rel, content in files.items():
            self.write_file(rel, content, role=role)
            written.append(rel)
        return written

    def seed_from(self, starter: Path) -> None:
        """Initialize workspace from a starter tree (iteration 0 / A0)."""
        starter = Path(starter)
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        if starter.exists():
            shutil.copytree(starter, self.workspace)
        else:
            self.workspace.mkdir(parents=True, exist_ok=True)

    def snapshot(self, iteration: int, *, label: str = "") -> Path:
        dest = self.versions / f"A_{iteration}"
        if dest.exists():
            shutil.rmtree(dest)
        if self.workspace.exists() and any(self.workspace.iterdir()):
            shutil.copytree(self.workspace, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        meta = {"iteration": iteration, "label": label}
        (dest / "_hoh_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.history.append(meta)
        return dest

    def save_plan(self, iteration: int, plan: Dict) -> Path:
        path = self.plans_dir / f"D_{iteration}.json"
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return path

    def save_evidence(self, iteration: int, evidence: Dict) -> Path:
        path = self.evidence_dir / f"E_{iteration}.json"
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _assert_developer(role: str) -> None:
        if role != "developer":
            raise PermissionError(
                f"single-writer violation: role={role!r} cannot modify artifact "
                "(only 'developer' may write)"
            )
