# Resources — Procedural Graphs experiments

Paper: [arXiv:2609.09153](https://arxiv.org/abs/2609.09153) · [PDF](https://arxiv.org/pdf/2609.09153)

Default base model: **`Qwen/Qwen2.5-7B-Instruct`** (fits ~24GB VRAM in bf16).

## Experiment table

| Experiment | Model | GPU | VRAM | RAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|-----|------|----------------|-------|
| cpu_demo | n/a (MockLLM) | none | — | — | <1GB | <15 s | stdlib + package; K=3 mini_shop |
| smoke | Qwen2.5-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥32GB | ≥50GB | 20–45 min first / ~10–20 min after | tokenizer + optional tiny generate |
| mini_evolve | Qwen2.5-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥32GB | ≥60GB | 60–180 min | K=3 full loop (solver+guidance+refiner) |

## Heavier alternative

| Experiment | Model | GPU | VRAM | RAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|-----|------|----------------|-------|
| smoke / mini (alt) | Qwen2.5-14B-Instruct | 1× A100 80GB | ≥80GB | ≥96GB | ≥100GB | 2–4 h | stronger; edit `model_id` in YAML |

## Disk breakdown (smoke)

- Model weights (Qwen2.5-7B-Instruct, bf16/safetensors): ~15 GB
- HF cache overhead + tokenizer: ~2–5 GB
- Artifacts (graphs/traces): <2 GB
- Container / env: budget to **≥50 GB** total disk

## Cost drivers

- First-run HF download dominates wall clock.
- Per-iteration cost ∝ (|D_train| + |D_val|) × (solver steps × (guidance + solver) + refiner + val) LLM calls.
- Graph JSON growth is disk-cheap; rejection memory is CPU-cheap.
- Do **not** auto-create paid pods; use `runpod/00_create_pod.example.sh` only after editing.

```bash
python scripts/estimate_resources.py
```
