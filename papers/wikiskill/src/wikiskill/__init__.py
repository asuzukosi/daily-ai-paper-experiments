"""WikiSkill — Compiling Agent Experience into Persistent Knowledge for Skill Evolution.

Paper: arXiv:2608.27454 (Tang et al., Google Research / Virginia Tech)
Three-layer workspace (raw / wiki / skills) + four-step evolutionary loop.
"""

from wikiskill.loop import LoopConfig, LoopResult, WikiSkillLoop, load_jsonl
from wikiskill.metrics import paper_headline_results

__version__ = "0.1.0"
__all__ = [
    "WikiSkillLoop",
    "LoopConfig",
    "LoopResult",
    "load_jsonl",
    "paper_headline_results",
    "__version__",
]
