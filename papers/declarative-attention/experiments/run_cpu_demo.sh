#!/usr/bin/env bash
# Always-safe CPU protocol simulator (no GPU, stdlib + da package).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

da_require_python
da_print_env
da_info "Running CPU Declarative Attention protocol demo…"
cd "${DA_ROOT}"
python3 demos/cpu_protocol_demo.py
