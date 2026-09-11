# Daily AI Paper Experiments

Daily AI paper experiments from [dair-ai/AI-Papers-of-the-Week](https://github.com/dair-ai/AI-Papers-of-the-Week): one unread paper from the latest week, a written summary, and a basic educational implementation.

## Layout

- `papers/<slug>/SUMMARY.md` — paper summary
- `papers/<slug>/` — runnable demo / implementation
- GitHub Issues — implementation broken down per paper

## Conventions

- Never repeat a paper once it has been covered.
- Prefer a small CPU demo that teaches the core idea.
- If a paper needs GPU / large-model experiments: check in experiment code + RunPod commands, but do **not** launch GPU jobs from automation.

## Covered so far

| Week | Paper | Folder |
|------|-------|--------|
| Aug 31 – Sep 6, 2026 | Declarative Attention ([arXiv:2609.02737](https://arxiv.org/abs/2609.02737)) | [`papers/declarative-attention`](papers/declarative-attention) |


## Research package layout (per paper)

Each `papers/<slug>/` should include:

- **README.md** — paper link + full write-up (problem, architecture/diagrams, decisions, results, how our experiment maps)
- **RESOURCES.md** — GPU/VRAM/disk/time per experiment
- **src/** — multi-file implementation
- **experiments/** — configs + bash runners
- **runpod/** — bash scripts to set up and run on a pod (do not auto-create paid pods from automation)
- A practical base model chosen for meaningful results on realistic GPUs
