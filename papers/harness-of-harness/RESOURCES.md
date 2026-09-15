# Resources — Harness-of-Harness experiments

Paper: [arXiv:2609.01481](https://arxiv.org/abs/2609.01481) · [PDF](https://arxiv.org/pdf/2609.01481)

Default base model: **`Qwen/Qwen2.5-Coder-7B-Instruct`** (coding-specialized; fits ~24GB VRAM).

## Experiment table

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| cpu_demo | n/a (MockLLM) | none | — | <1GB | <10 s | stdlib + package; T=3 mini todo |
| smoke | Qwen2.5-Coder-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥50GB | 20–45 min first run / ~10–20 min after | T=2 |
| mini_programbench | Qwen2.5-Coder-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥60GB | 45–120 min | T=3 |

## Heavier alternative

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| smoke / mini (alt) | Qwen2.5-Coder-14B-Instruct | 1× A100 80GB | ≥80GB | ≥100GB | 1–3 h | stronger coding; edit `model_id` in YAML |

## Host RAM

| Experiment | System RAM |
|------------|------------|
| smoke | ≥32 GB |
| mini_programbench | ≥32 GB |
| 14B alt | ≥96 GB recommended |

## Disk breakdown (smoke)

- Model weights (Qwen2.5-Coder-7B-Instruct, bf16/safetensors): ~15 GB
- HF cache overhead + tokenizer: ~2–5 GB
- Artifacts / versioned workspaces: <2 GB
- Container / env: budget to **≥50 GB** total disk

## Notes

- First run is dominated by Hugging Face download; subsequent runs reuse the cache.
- GPU driver **hard-gates** without CUDA (exit code 2) — use `experiments/run_cpu_demo.sh` on CPU.
- Do **not** auto-create paid pods; use `runpod/00_create_pod.example.sh` only after editing.
- Paper harness–model pairs (Codex+GPT-5.5, OpenCode+DeepSeek-V4-Pro, Pi+MiniMax-M3) are cited for comparison; our default is open Qwen2.5-Coder-7B for reproducible RunPod cost.

Regenerate a live table from configs (no GPU):

```bash
python scripts/estimate_resources.py
```
