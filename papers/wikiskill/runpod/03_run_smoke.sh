#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
# Optional: skip full weights — tokenizer-only smoke
# export WIKISKILL_SMOKE_SKIP_LOAD=1
bash experiments/run_smoke.sh experiments/configs/smoke_qwen25_7b.yaml
