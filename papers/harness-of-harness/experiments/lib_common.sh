#!/usr/bin/env bash
# Shared env helpers for Harness-of-Harness experiments.
set -euo pipefail

HOH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOH_ROOT
export PYTHONPATH="${HOH_ROOT}/src:${PYTHONPATH:-}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-${HOH_ROOT}/artifacts}"
mkdir -p "${ARTIFACTS_DIR}"

hoh_info() { echo "[hoh] $*"; }
hoh_die()  { echo "[hoh] ERROR: $*" >&2; exit 1; }

hoh_require_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    hoh_die "python3 not found"
  fi
}

hoh_print_env() {
  hoh_info "HOH_ROOT=${HOH_ROOT}"
  hoh_info "PYTHONPATH=${PYTHONPATH}"
  hoh_info "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
  hoh_info "python=$(command -v python3) ($(python3 --version 2>&1))"
}
