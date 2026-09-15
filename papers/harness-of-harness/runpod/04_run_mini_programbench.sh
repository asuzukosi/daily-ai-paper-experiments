#!/usr/bin/env bash
# Run mini ProgramBench-style GPU experiment (T=3).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec bash "${HOH_ROOT}/experiments/run_mini_programbench.sh"
