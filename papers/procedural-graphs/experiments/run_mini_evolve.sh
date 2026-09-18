#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_common.sh"
pg_require_python
pg_print_env
pg_info "GPU mini evolve — Qwen2.5-7B-Instruct K=3"
python3 "${PG_ROOT}/experiments/run_gpu_experiment.py" \
  --config "${PG_ROOT}/experiments/configs/mini_evolve_qwen25_7b.yaml"
