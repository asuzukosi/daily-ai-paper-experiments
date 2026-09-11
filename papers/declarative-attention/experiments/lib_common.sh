#!/usr/bin/env bash
# Shared env helpers for Declarative Attention experiments.
set -euo pipefail

DA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DA_ROOT
export PYTHONPATH="${DA_ROOT}/src:${PYTHONPATH:-}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-${DA_ROOT}/artifacts}"
mkdir -p "${ARTIFACTS_DIR}"

da_info() { echo "[da] $*"; }
da_die()  { echo "[da] ERROR: $*" >&2; exit 1; }

da_require_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    da_die "python3 not found"
  fi
}

da_print_env() {
  da_info "DA_ROOT=${DA_ROOT}"
  da_info "PYTHONPATH=${PYTHONPATH}"
  da_info "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
  da_info "python=$(command -v python3) ($(python3 --version 2>&1))"
}
