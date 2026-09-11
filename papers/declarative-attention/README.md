# Declarative Attention — educational demo

Didactic simulator of the core idea from:

> **Language Models Can Control Their Own Attention** (Ho et al., arXiv:2609.02737)

## What this shows

Language models usually read the **entire KV cache** on every generated token. Declarative Attention (DA) lets the model announce, inside its chain-of-thought, which part of the context it needs:

- `<global>` — attend to all context segments (navigation)
- `<focus magic_chunks="K">` — attend only to named segment(s)
- `<local>` — attend to recent output only (no long-context chunks)

An **inference-style controller** parses those tags like tool calls and restricts which prompt tokens are counted as “read” at each decode step. This demo is a **token-counting simulator**, not a real transformer and not training.

## How to run

```bash
python3 main.py
```

Requirements: Python 3.8+ standard library only (no numpy/torch).

## Expected behavior

Prints:

1. Mock prompt size (scaffold + magic chunks)
2. Mode spans as the controller follows a scripted Acme Corp CoT (mirrors the paper’s example)
3. Per-mode approximate attended vs vanilla tokens
4. Total token-read savings vs full attention

## Files

- `main.py` — entrypoint + DA state machine + demo
- `README.md` — this file
