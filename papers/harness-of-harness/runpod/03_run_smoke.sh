#!/usr/bin/env bash
# Run GPU smoke on the pod (T=2).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec bash "${HOH_ROOT}/experiments/run_smoke.sh"
