# Procedural Graphs — short summary

**arXiv:** [2609.09153](https://arxiv.org/abs/2609.09153) · Lu, Chen, Wu, Arık (Google / Georgia Tech / Peking)  
**Week:** September 7 – September 13, 2026

## One-liner

Represent agent procedures as an editable attributed graph; guide the solver online from a localized subgraph; evolve topology/attributes offline with validation gating and rejection memory.

## Method

- **PG:** `(procedure, relation, procedure)` + `condition/guidance/pitfalls`
- **Online:** Match `ut` → `h=2` neighborhood → guidance LLM Ψ → soft bias for ReAct solver Φ
- **Offline:** contrast fail vs success → ΔG → commit iff `Sval` ≥ previous → keep rejects in `H_rejected`

## Headline results

- First/joint-first in 21/24 settings; BFCL +9.0, GDPval +7.41, τ-bench +6.96 (selected margins)
- Scratch+evolution (Mode 5) beats unguided on HotpotQA; evolution repairs flawed expert priors
- Localized generative guidance beats full-graph raw/generative ablations (Table 3)

## Our package

CPU MockLLM demo evolves a shopping tool-use graph on `tasks/mini_shop` and compares no-graph / static / evolved. GPU default: **Qwen2.5-7B-Instruct**.
