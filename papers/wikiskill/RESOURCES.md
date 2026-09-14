# Resources — WikiSkill experiments

Paper: [arXiv:2608.27454](https://arxiv.org/abs/2608.27454) · [PDF](https://arxiv.org/pdf/2608.27454)

Default base model: **`Qwen/Qwen2.5-7B-Instruct`** (fits ~24GB VRAM in bf16).

## Experiment table

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| cpu_demo | n/a (MockLLM) | none | — | <1GB | <10 s | stdlib + package; K=3 mini LiveMath |
| smoke | Qwen2.5-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥50GB | 20–45 min first run / ~10–20 min after | tokenizer + optional tiny generate |
| mini_evolve | Qwen2.5-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥60GB | 45–150 min | K=3 full loop |

## Heavier alternative

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| smoke / mini (alt) | Qwen2.5-14B-Instruct | 1× A100 80GB | ≥80GB | ≥100GB | 1–3 h | stronger; edit `model_id` in YAML |

## Host RAM

| Experiment | System RAM |
|------------|------------|
| smoke | ≥32 GB |
| mini_evolve | ≥32 GB |
| 14B alt | ≥96 GB recommended |

## Disk breakdown (smoke)

- Model weights (Qwen2.5-7B-Instruct, bf16/safetensors): ~15 GB
- HF cache overhead + tokenizer: ~2–5 GB
- Artifacts (raw/wiki/skills): <2 GB
- Container / env: budget to **≥50 GB** total disk

## Cost drivers

- First-run HF download dominates wall clock.
- Per-iteration cost ∝ (|D_train| + |D_val|) × (Inference + Maintainer + Proposer + Val) LLM calls.
- Wiki growth is disk-cheap; skill rollbacks are CPU-cheap.
- Do **not** auto-create paid pods; use `runpod/00_create_pod.example.sh` only after editing.

Regenerate a live table from configs (no GPU):

```bash
python scripts/estimate_resources.py
```
