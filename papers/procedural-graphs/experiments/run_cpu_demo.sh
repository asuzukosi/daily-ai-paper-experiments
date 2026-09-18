#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"
pg_require_python
pg_print_env
pg_info "Running CPU MockLLM demo (K=3 mini_shop)"
python3 "${PG_ROOT}/demos/cpu_pg_demo.py"
