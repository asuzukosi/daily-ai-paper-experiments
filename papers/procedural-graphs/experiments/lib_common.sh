#!/usr/bin/env bash
set -euo pipefail
PG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PG_ROOT
export PYTHONPATH="${PG_ROOT}/src:${PYTHONPATH:-}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PG_ROOT}/artifacts}"
mkdir -p "${ARTIFACTS_DIR}"
pg_info() { echo "[procedural-graphs] $*"; }
pg_die()  { echo "[procedural-graphs] ERROR: $*" >&2; exit 1; }
pg_require_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    pg_die "python3 not found"
  fi
}
pg_print_env() {
  pg_info "PG_ROOT=${PG_ROOT}"
  pg_info "PYTHONPATH=${PYTHONPATH}"
  pg_info "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
  pg_info "python=$(command -v python3) ($(python3 --version 2>&1))"
}
