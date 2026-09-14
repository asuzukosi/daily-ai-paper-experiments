#!/usr/bin/env bash
# GPU smoke: import/config check; optional tiny generate if CUDA present.
# HARD GATE without CUDA (exit 2) — use run_cpu_demo.sh on CPU hosts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
ws_require_python
ws_print_env
CFG="${1:-${WIKISKILL_ROOT}/experiments/configs/smoke_qwen25_7b.yaml}"
ws_info "Smoke config: ${CFG}"
python3 "${WIKISKILL_ROOT}/experiments/run_gpu_experiment.py" --config "${CFG}" --smoke
