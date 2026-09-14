#!/usr/bin/env bash
set -euo pipefail
WIKISKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WIKISKILL_ROOT
export PYTHONPATH="${WIKISKILL_ROOT}/src:${PYTHONPATH:-}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-${WIKISKILL_ROOT}/artifacts}"
mkdir -p "${ARTIFACTS_DIR}"
ws_info() { echo "[wikiskill] $*"; }
ws_die()  { echo "[wikiskill] ERROR: $*" >&2; exit 1; }
ws_require_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    ws_die "python3 not found"
  fi
}
ws_print_env() {
  ws_info "WIKISKILL_ROOT=${WIKISKILL_ROOT}"
  ws_info "PYTHONPATH=${PYTHONPATH}"
  ws_info "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
  ws_info "python=$(command -v python3) ($(python3 --version 2>&1))"
}
