"""Metrics helpers + paper headline citations."""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import EpisodeResult


def mean_score(results: List[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def success_rate(results: List[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def paper_headline_results() -> Dict[str, Any]:
    """Citation-only numbers from the paper (not our mini-bench)."""
    return {
        "table1_note": "PG ranks first or joint-first in 21/24 model–benchmark settings",
        "sign_test_p": 4.3e-4,
        "highlights": {
            "BFCL_v3_Gemini35Flash": {"pg": 67.00, "best_baseline": 58.00, "delta": 9.00},
            "GDPval_Gemini31Pro": {"pg": 78.78, "best_baseline": 71.37, "delta": 7.41},
            "tau_bench_Gemini31Pro": {"pg": 80.00, "best_baseline": 73.04, "delta": 6.96},
            "ALFWorld_Gemini31Pro": {"pg": 100.00},
        },
        "construction_modes_hotpotqa_ans_f1": {
            "unguided": 71.21,
            "mode5_scratch_evolution": 78.79,
            "mode3_expert_evolution": 76.34,
        },
        "multichallenge_overall": {
            "unguided": 87.50,
            "mode1_flawed_expert": 58.93,
            "mode3_repaired": 92.86,
        },
        "enterprise_survival_claude": {"baseline": 44.0, "pg": 58.0},
        "ablation_subgraph_generative": {
            "MultiChallenge": 89.31,
            "GDPval": 63.99,
            "ALFWorld": 81.53,
        },
    }
