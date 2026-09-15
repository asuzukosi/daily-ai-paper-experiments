# Implementation notes — WikiSkill (2026-09-13)

## What ran

- `PYTHONPATH=src python3 demos/cpu_wikiskill_demo.py` — **passed**
- Baseline val (empty skills) ≈ **0.25**
- After K=3 with MockLLM skill evolution: val/test ≈ **1.00**
- Accepted skills (example run): `final-answer-format`, `multi-step-arithmetic`, `unit-normalization`
- Wiki patterns accumulated (example): `missing-final-answer-format`, `off-by-one-arithmetic`, `unit-contamination`
- Artifacts under `artifacts/cpu_demo/{raw,wiki,skills,run_summary.json}`

## Caveats

- MockLLM is **scripted** to exhibit format / arithmetic / unit failures that clear when the matching skill is injected. This validates the loop mechanics end-to-end on CPU; it is not a claim about Qwen2.5-7B accuracy on LiveMath.
- GPU path hard-gates without CUDA (`experiments/run_gpu_experiment.py` exit code 2). Do not run GPU jobs from this box; use RunPod scripts manually.
- Mini LiveMath split is a **research scaffold** (~32/12/18), not the paper’s full LiveMathematicianBench.
- Paper numbers cited are Table 1 / text only — no invented metrics.
- GitHub push deferred (see `PENDING_PUSH.md`).

## Base model

Primary: `Qwen/Qwen2.5-7B-Instruct`  
Optional alt: `Qwen/Qwen2.5-14B-Instruct` (A100 80GB)
