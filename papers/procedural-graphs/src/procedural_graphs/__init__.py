"""Procedural Graphs — self-evolving execution structures for LLM agents (arXiv:2609.09153)."""

from .graph import ProceduralGraph, flawed_expert_prior, minimal_skeleton
from .loop import EvolutionConfig, EvolutionLoop

__all__ = [
    "ProceduralGraph",
    "minimal_skeleton",
    "flawed_expert_prior",
    "EvolutionLoop",
    "EvolutionConfig",
]
__version__ = "0.1.0"
