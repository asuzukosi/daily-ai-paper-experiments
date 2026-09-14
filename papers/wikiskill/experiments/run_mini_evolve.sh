#!/usr/bin/env bash
# Full mini WikiSkill evolution on GPU (Qwen2.5-7B-Instruct).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
ws_require_python
ws_print_env
CFG="${1:-${WIKISKILL_ROOT}/experiments/configs/mini_evolve_qwen25_7b.yaml}"
ws_info "Mini evolve config: ${CFG}"
python3 "${WIKISKILL_ROOT}/experiments/run_gpu_experiment.py" --config "${CFG}"
