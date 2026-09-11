# RunPod scaffold — Declarative Attention (optional)

The CPU demo in `../main.py` teaches the protocol. A **full** DA evaluation (long-context models + vLLM block masking) needs GPUs. This folder is scaffolding only.

**Do not auto-start pods from the daily agent.** Run these commands yourself when you want a real experiment.

## Prerequisites

- RunPod account + API key
- [RunPod CLI](https://docs.runpod.io/references/runpodctl) (`runpodctl`) or REST API
- A Hugging Face token if the model weights are gated

```bash
export RUNPOD_API_KEY=...
export HF_TOKEN=...   # if needed
```

## Suggested pod

- GPU: 1× A100 80GB (or 2× if the model + 128k–1M context KV does not fit)
- Template: PyTorch + CUDA image you already trust, or a vLLM image
- Disk: ≥100 GB for weights + logs

### Example: create a pod (CLI shapes vary by CLI version — confirm against current RunPod docs)

```bash
# List available GPU types
runpodctl get gpu

# Create (example flags — adjust to your CLI version / template IDs)
runpodctl create pod \
  --name dair-da-exp \
  --gpuType "NVIDIA A100 80GB" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --volumeSize 100 \
  --containerDiskSize 50 \
  --ports "8888/http,22/tcp"
```

### Sync this experiment folder up

```bash
# From your laptop, after SSH/rsync details are shown in the RunPod UI:
rsync -avz papers/declarative-attention/runpod/ root@<POD_HOST>:/workspace/da-runpod/
```

### On the pod

```bash
cd /workspace/da-runpod
pip install -r requirements.txt
python experiment.py --help
# Start a short smoke run ONLY when you intentionally want GPU spend:
# python experiment.py --model <hf-id> --max-new-tokens 64 --dry-config-check
```

### Stop / remove when done

```bash
runpodctl stop pod <POD_ID>
runpodctl remove pod <POD_ID>
```

## What `experiment.py` is for

A stub entrypoint that documents the intended DA eval loop (chunk the prompt, parse mode tags, apply attention masks in vLLM). It refuses to load a model unless `--i-understand-this-costs-gpu` is passed, so accidental runs are harder.
