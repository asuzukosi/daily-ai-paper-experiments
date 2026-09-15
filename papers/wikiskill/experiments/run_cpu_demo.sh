#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
ws_require_python
ws_print_env
ws_info "Running CPU MockLLM demo (K=3 mini_livemath)"
python3 "${WIKISKILL_ROOT}/demos/cpu_wikiskill_demo.py"
