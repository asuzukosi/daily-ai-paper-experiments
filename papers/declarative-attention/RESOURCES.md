# Resources — Declarative Attention experiments

Paper: [arXiv:2609.02737](https://arxiv.org/abs/2609.02737) · [PDF](https://arxiv.org/pdf/2609.02737)

Default base model: **`Qwen/Qwen2.5-7B-Instruct`**.

## Experiment table

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| cpu_demo | n/a (simulator) | none | — | <1GB | <5 s | stdlib protocol demo |
| smoke | Qwen2.5-7B-Instruct | 1× RTX 4090 / A6000 / L40 | ≥24GB | ≥40GB | 15–40 min first run (download) / ~5–10 min after | short context smoke |
| longctx | Qwen2.5-7B-Instruct | 1× A100 80GB (preferred) or A6000 48GB | ≥48GB | ≥80GB | 1–3 h | 32k–64k tokens |

## Heavier alternative

| Experiment | Model | GPU | VRAM | Disk | Est. wall time | Notes |
|------------|-------|-----|------|------|----------------|-------|
| smoke / longctx (alt) | Qwen2.5-14B-Instruct | 1× A100 80GB | ≥80GB | ≥100GB | 30 min–4 h | stronger protocol adherence; edit `model_id` in YAML |

## Host RAM

| Experiment | System RAM |
|------------|------------|
| smoke | ≥32 GB |
| longctx | ≥64 GB |
| 14B alt | ≥96 GB recommended |

## Disk breakdown (smoke)

- Model weights (Qwen2.5-7B-Instruct, bf16/safetensors): ~15 GB
- HF cache overhead + tokenizer: ~2–5 GB
- Artifacts / logs: <1 GB
- Container / env: budget to **≥40 GB** total disk

## Notes

- First run is dominated by Hugging Face download; subsequent runs reuse the cache.
- Smoke uses `max_model_len: 8192`; longctx uses `32768` (raise carefully with VRAM).
- Attended-token metrics in this repo are **projected from the DA controller** unless you wire full vLLM attention-metadata masking (paper-style). Protocol adherence is still measurable.
- Do **not** auto-create paid pods; use `runpod/00_create_pod.example.sh` only after editing.

Regenerate a live table from configs (no GPU):

```bash
python scripts/estimate_resources.py
```
