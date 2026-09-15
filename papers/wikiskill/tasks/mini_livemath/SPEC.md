# Mini LiveMath — SPEC

LiveMath-inspired mini bench for WikiSkill skill-evolution loops.

## Splits
| Split | File | n | Purpose |
|-------|------|---|---------|
| train | train.jsonl | 32 | Inference rollouts + wiki/skill proposals |
| val | val.jsonl | 12 | Gating / rollback threshold R_best |
| test | test.jsonl | 18 | Held-out report metric |

## Tags / failure modes (for skill learning signal)
- `easy` — baseline solvable without skills
- `format` — requires `Final answer: <value>` line (exact-match scorer)
- `arith` — multi-step arithmetic (off-by-one / dropped addend without skill)
- `unit` — answers must be unitless numbers
- `money` / `mul` / `add` / `sub` — topical tags

## Scoring
Exact match after light normalization (strip `$`/`"` wrappers, parse ints).
Ground-truth `answer` field is authoritative.

## Mapping to paper
Analogous to **LiveMathematicianBench (LiveMath)** used in WikiSkill Table 1,
scaled down for a reproducible research scaffold (not the full benchmark).
