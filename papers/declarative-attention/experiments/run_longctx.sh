#!/usr/bin/env bash
# Long-context experiment — Qwen2.5-7B-Instruct (requires CUDA / RunPod).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

da_require_python
da_print_env

CONFIG="${1:-${SCRIPT_DIR}/configs/longctx_qwen25_7b.yaml}"
da_info "Config: ${CONFIG}"

if [[ "${DRY_CONFIG_CHECK:-0}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/run_gpu_experiment.py" --config "${CONFIG}" --dry-config-check
  exit 0
fi

python3 "${SCRIPT_DIR}/run_gpu_experiment.py" \
  --config "${CONFIG}" \
  --artifacts "${ARTIFACTS_DIR}"
