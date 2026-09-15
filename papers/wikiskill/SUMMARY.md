# WikiSkill — short summary

**arXiv:** [2608.27454](https://arxiv.org/abs/2608.27454) · Tang et al. (Google Research / Virginia Tech)  
**Week:** August 31 – September 6, 2026

## One-liner

Co-evolve a **persistent wiki** (never reset) with a **gated skill set** so agent experience compounds into reliable procedural skills.

## Method

Three layers: immutable `raw/` traces, compounding `wiki/` patterns+logs+skill-impact, reversible `skills/`.  
Four steps per iteration: Inference (skills only) → Wiki Maintainer → Skill Proposer (ReAct) → Gating/Rollback (wiki always kept).

## Headline results (Table 1 Avg)

- Qwen-3.5-4B: 26.2 → **38.5**
- Qwen-3.5-9B: 29.9 → **47.4** (beats 27B no-skill 39.4)
- Qwen-3.6-27B: 39.4 → **63.3**
- Gemma-4-31B: 41.3 → **54.9**
- Gemini-3.5-Flash: 49.5 → **68.1** (LiveMath 33.0→72.6; SpreadSheet 50.5→76.6)
- Transfer: 9B + 27B skills 70.2% vs self-evolved 63.4% on ALFWorld
- Ablation: persistent wiki is critical

## Our package

CPU MockLLM demo evolves skills on `tasks/mini_livemath` (val ~0.25 → ~1.0 by K=3). GPU default: **Qwen2.5-7B-Instruct**.
