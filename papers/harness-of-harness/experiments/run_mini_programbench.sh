#!/usr/bin/env bash
# Mini ProgramBench-style GPU experiment (T=3).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
hoh_require_python
hoh_print_env
CFG="${1:-${HOH_ROOT}/experiments/configs/mini_programbench_qwen25_coder_7b.yaml}"
hoh_info "Config: ${CFG}"
exec python3 "${HOH_ROOT}/experiments/run_gpu_experiment.py" --config "${CFG}"
