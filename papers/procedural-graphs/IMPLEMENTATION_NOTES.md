# Implementation notes

- Faithful scaffold of arXiv:2609.09153 for DAIR daily paper experiments (2026-09-15).
- MockLLM is intentionally scripted so unguided agents skip Verify and guided/evolved agents follow CheckInventory→Verify→Answer — gating sees real score deltas.
- GPU path shares the same loop; quality depends on Qwen instruction-following for tool-call formatting.
- Never launch paid RunPod pods from automation; scripts under `runpod/` are manual.
- Branch target for shipping: `paper/procedural-graphs-2026-09-15` → PR into `main`.
