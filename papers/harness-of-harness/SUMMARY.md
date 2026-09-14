# Summary — Harness-of-Harness (arXiv:2609.01481)

**Title:** Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement  
**Authors:** Haoyang Yan†, Min-Le Su†, Hangfan Zhang†, Zhanhao Li†, Chen Zhang, Shao Zhang, Yang Chen, Lei Bai, Shuyue Hu (Shanghai AI Lab)  
**Links:** [arXiv](https://arxiv.org/abs/2609.01481) · [PDF](https://arxiv.org/pdf/2609.01481) · [Code](https://github.com/Flesymeb/HarnessOfHarness)  
**Week:** August 31 – September 6, 2026 (DAIR)

## Problem

Autonomous software development asks coding agents to turn a fixed high-level specification **S** into complete, usable software **without** continuous human tasking. Long trajectories drift: local repairs fight global requirements, evidence invalidates earlier assumptions, and agents loop on inspection/repair or declare victory too early.

## Method (Algorithm 1)

HoH wraps an existing coding **harness** and runs iterative **Planner → Developer → QA Tester** loops with two cross-loop states: artifact **A_t** and evidence **E_t** (spec **S** fixed).

```
E0 ← ∅
for t = 1..T:
  Dt ← ProjectPlanner(S, E_{t-1}; read_only(A_{t-1}))
  At ← Developer(A_{t-1}; S, Dt)
  Et ← QATester(read_only(At); S, Dt, Runtime.check(At))
return A_T
```

Key decisions: balance repair vs capability growth; small verifiable increments; separate implementation-time testing from independent QA; constrain verifiable **output schemas** (retry on fail), not workflows; progressive disclosure of artifacts/tools/skills; versioned histories; **single-writer** (only Developer mutates A; Planner/QA read-only).

## Results (paper, exact)

- Avg relative gain **52.25%**, max **82.86%** after 3 iterations vs standalone harnesses.
- **GameCraft Overall:** Codex 49.58→71.52 (+21.93); OpenCode 26.90→48.98 (+22.08); Pi 42.16→58.78 (+16.62).
- **FrontierSWE Dominance:** Codex 44%→71% (+27); OpenCode 25%→44% (+19); Pi 35%→64% (+29).
- **ProgramBench pass rate:** Codex 60.41→66.50; OpenCode 45.27→57.56; Pi 35.83→52.68.
- Multi-day: **70+** iterations FPS game Fusepoint.
- Harness–model pairs: Codex+GPT-5.5, OpenCode+DeepSeek-V4-Pro, Pi+MiniMax-M3.

## Our experiment mapping

| Paper piece | This package |
|-------------|--------------|
| Algorithm 1 loop | `src/hoh/loop.py` |
| Role contracts + schema retries | `src/hoh/roles.py`, `schemas.py`, `prompts.py` |
| Single-writer artifact + snapshots | `src/hoh/artifacts.py` |
| Runtime.check | `src/hoh/runtime.py` |
| Metrics / paper citation helpers | `src/hoh/metrics.py` |
| CPU MockLLM T=3 demo | `demos/cpu_loop_demo.py` |
| Mini ProgramBench-style task | `tasks/mini_cli_todo/` |
| GPU driver (Qwen2.5-Coder-7B) | `experiments/run_gpu_experiment.py` |
| RunPod wrappers | `runpod/` |

**Base model:** `Qwen/Qwen2.5-Coder-7B-Instruct` — coding-specialized, ~24GB VRAM, meaningful for mini todo loops. Optional: `Qwen2.5-Coder-14B-Instruct` on A100 80GB.

## Honest scope

CPU demo fully exercises schemas, single-writer, Runtime.check, and T=3 progressive growth with MockLLM. GPU path loads a real coder LM via transformers and runs the same loop; it is a **research scaffold**, not a reproduction of Codex/OpenCode/Pi + closed frontier models from the paper.
