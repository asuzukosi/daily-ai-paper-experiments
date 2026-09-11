#!/usr/bin/env bash
# Install Python deps and sanity-check HF / CUDA on the pod.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${DA_ROOT}"
echo "[da] Installing requirements from ${DA_ROOT}/requirements.txt"
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

echo "[da] Checking CUDA…"
python3 - <<'PY'
import sys
try:
    import torch
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device0={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}")
    else:
        print("WARNING: CUDA not visible — smoke/longctx will hard-gate.", file=sys.stderr)
except ImportError:
    print("WARNING: torch not installed yet / failed import", file=sys.stderr)
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[da] HF_TOKEN is set — logging in (huggingface-cli)…"
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || \
    python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
else
  echo "[da] HF_TOKEN unset — public models (Qwen2.5-7B-Instruct) usually still download."
fi

echo "[da] Setup complete."
