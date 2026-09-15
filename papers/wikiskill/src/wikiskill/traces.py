"""Raw Layer — immutable execution traces (write-once under raw/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from wikiskill.schemas import TraceRecord


class RawStore:
    """Write-once store for execution traces.

    Layout:
      raw/iter_{k}/{split}/{task_id}.json
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def iter_dir(self, iteration: int, split: str) -> Path:
        d = self.root / f"iter_{iteration}" / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, trace: TraceRecord, *, overwrite: bool = False) -> Path:
        path = self.iter_dir(trace.iteration, trace.split) / f"{trace.task_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Raw Layer is write-once; refuse overwrite of {path}")
        path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
        return path

    def write_many(self, traces: Iterable[TraceRecord], *, overwrite: bool = False) -> List[Path]:
        return [self.write(t, overwrite=overwrite) for t in traces]

    def load_split(self, iteration: int, split: str) -> List[TraceRecord]:
        d = self.root / f"iter_{iteration}" / split
        if not d.exists():
            return []
        out: List[TraceRecord] = []
        for p in sorted(d.glob("*.json")):
            out.append(TraceRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        return out

    def sample(
        self,
        iteration: int,
        split: str = "train",
        *,
        max_fail: int = 6,
        max_pass: int = 4,
    ) -> List[TraceRecord]:
        traces = self.load_split(iteration, split)
        fails = [t for t in traces if not t.correct]
        passes = [t for t in traces if t.correct]
        return fails[:max_fail] + passes[:max_pass]

    def outcome_summary(self, iteration: int, split: str = "train") -> str:
        traces = self.load_split(iteration, split)
        lines = [f"# Training outcomes (iteration {iteration}, n={len(traces)})", ""]
        for t in traces:
            status = "PASS" if t.correct else "FAIL"
            lines.append(
                f"- [{status}] {t.task_id}: pred={t.prediction!r} gt={t.ground_truth!r}"
            )
        return "\n".join(lines) + "\n"
