#!/usr/bin/env bash
# Install package + GPU extras on a RunPod host (run AFTER pod exists).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
echo "[wikiskill] Setting up env in ${ROOT}"
python3 -m pip install -U pip
python3 -m pip install -e ".[gpu]"
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
echo "[wikiskill] setup done"
