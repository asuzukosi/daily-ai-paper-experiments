#!/usr/bin/env bash
# Run the MockLLM CPU loop demo (no GPU / no torch).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
hoh_require_python
hoh_print_env
hoh_info "Running demos/cpu_loop_demo.py"
exec python3 "${HOH_ROOT}/demos/cpu_loop_demo.py"
