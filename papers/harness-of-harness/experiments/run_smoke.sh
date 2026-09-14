#!/usr/bin/env bash
# GPU smoke: T=2 mini ProgramBench-style loop with Qwen2.5-Coder-7B-Instruct.
# HARD GATE inside run_gpu_experiment.py if CUDA missing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
hoh_require_python
hoh_print_env
CFG="${1:-${HOH_ROOT}/experiments/configs/smoke_qwen25_coder_7b.yaml}"
hoh_info "Config: ${CFG}"
exec python3 "${HOH_ROOT}/experiments/run_gpu_experiment.py" --config "${CFG}"
