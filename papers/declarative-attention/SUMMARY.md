# Declarative Attention

**Full title:** Language Models Can Control Their Own Attention  
**Authors:** Namgyu Ho\*, Huzama Ahmad\*, Woosung Koh\* (KAIST AI); Se-Young Yun† (KAIST AI); Tal Schuster†, Cicero Nogueira dos Santos† (Google DeepMind)  
\* Equal contribution · † Corresponding authors  
**Week:** August 31 – September 6, 2026  
**arXiv:** [2609.02737](https://arxiv.org/abs/2609.02737) · [PDF](https://arxiv.org/pdf/2609.02737)  
**DAIR page:** [academy.dair.ai/…](https://academy.dair.ai/papers/language-models-can-control-their-own-attention-2609.02737)  
**Published:** September 2, 2026 (arXiv v1)

> Note: Google co-authors performed only an advisory role.

---

## Motivation

Transformers re-read the entire KV cache at every decode step, even though attention mass concentrates on a small slice of context. In long-context regimes that memory traffic dominates latency (e.g., ~15 GB of KV per step at 1M tokens for a large MoE). Extrinsic sparse-attention methods use proxy scores to guess relevant tokens, but still pay **O(N)** per step to scan the cache. The authors ask: *wouldn't the model already know which parts of the context are relevant?* Declarative Attention (DA) turns that intuition into a protocol: the model **declares** where it needs to look inside its chain-of-thought, and the inference engine skips the rest of the KV read.

## Method

DA is a **zero-shot protocol** (no weight updates) with three attention modes emitted as parseable tags in the CoT:

| Mode | Tag | What is attended |
|------|-----|------------------|
| **Global** | `<global>…</global>` | Full context (navigation / locating relevant segments) |
| **Focus** | `<focus magic_chunks="K">…</focus>` | Named context segment(s) only |
| **Local** | `<local>…</local>` | Recent model output only (no long-context segments) |

**Prompt structure:** system instruction + long context split into ~2K-token addressable **“magic chunks”** + question + DA instruction. The scaffold (system, question, instruction) and the response so far stay visible in every mode; only the long-context segments are gated.

**Decode-time state machine:** starts in global; transitions on opening tags (`<focus…>` / `<local>`); reverts to global on matching close tags. Masking is **block-aligned** (vLLM KV blocks, typically 16–32 tokens) so FlashAttention-style kernels can skip whole blocks. Integrated via vLLM attention-metadata hooks (no custom kernels).

Unlike proxy-scorer sparse attention, DA incurs full-context reads only during declared **global** spans—not as a fixed per-step overhead.

## Results (from the paper)

Zero-shot evaluation across **15 long-context tasks** on off-the-shelf models:

| Model | Attended-token reduction | Accuracy drop |
|-------|--------------------------|---------------|
| **Gemma-4-31B** | **52.0%** (13.43M → 6.45M avg attended tokens/response) | **1.27pp** (87.01% → 85.74%) |
| **Qwen-3.6-27B** | **31.1%** (22.54M → 15.52M) | **2.75pp** (85.31% → 82.56%) |

Additional findings reported in the paper:

- **Scaling:** Relative accuracy under DA rises with size (e.g., ~29% of vanilla at Gemma-4-E4B → ~99% at Gemma-4-31B; ~64% at Qwen-3.5-4B → ~97% at Qwen-3.6-27B). Protocol adherence (valid focus chunk refs) reaches ~99% at the largest models.
- **Mask ablation (DA-nm):** Prompting alone lengthens generations and can *increase* attended tokens; the dynamic mask is what drives savings (up to **71.1%** fewer attended tokens vs maskless DA on Gemma).
- **Mode mix (Gemma-4-31B):** ~27% of generated tokens in `<global>`; `<focus>` / `<local>` (~73%) attend ~12% / ~6% of vanilla tokens per step on average.
- **Wall-clock projection (roofline):** decode cost → **0.71×** (Gemma-4-31B) and **0.77×** (Qwen-3.6-27B) of vanilla on a well-optimized stack; absolute savings grow with context length (up to ~21M tokens saved per response in long buckets).

## Why it matters

Long-context serving (agents over repos, long chats) is structurally expensive because every decode step reloads the full KV cache. DA moves routing into text the model already produces—no auxiliary scorer, no fine-tune required for a usable lower bound—and opens a new axis of sparse attention that can be sharpened with post-training.

## Practical takeaway

For serving stacks: partition long inputs into named chunks, prompt the model to emit `<global>` / `<focus magic_chunks="…">` / `<local>` spans, and have the engine parse those like tool calls to rewrite the KV block table each step. Expect large decode-attention savings with ~1–3pp average accuracy cost on strong models; smaller models may fail the protocol. Training for DA is left as future headroom beyond this zero-shot baseline.

## Sources used

- Full PDF: https://arxiv.org/pdf/2609.02737 (extracted with `pdftotext`)
- Abstract page: https://arxiv.org/abs/2609.02737
- DAIR Academy curator summary: https://academy.dair.ai/papers/language-models-can-control-their-own-attention-2609.02737
