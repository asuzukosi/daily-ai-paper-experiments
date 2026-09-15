"""Harness-of-Harness (HoH) — research scaffolding for arXiv:2609.01481.

Faithful to Algorithm 1:
  E0 = empty
  for t = 1..T:
    Dt = Planner(S, E; read_only(A))
    At = Developer(A; S, D)
    Et = QATester(read_only(A); S, D, Runtime.check)

Single-writer: only Developer may mutate the artifact.
Planner / QA receive read-only copies.
"""

from hoh.loop import HarnessOfHarness, LoopConfig, LoopResult
from hoh.schemas import DevelopmentPlan, EvidenceReport, validate_plan, validate_evidence

__all__ = [
    "HarnessOfHarness",
    "LoopConfig",
    "LoopResult",
    "DevelopmentPlan",
    "EvidenceReport",
    "validate_plan",
    "validate_evidence",
]

__version__ = "0.1.0"
