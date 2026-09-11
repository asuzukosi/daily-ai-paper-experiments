#!/usr/bin/env bash
# On the pod: run the long-context experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DA_ROOT}"
bash experiments/run_longctx.sh "$@"
