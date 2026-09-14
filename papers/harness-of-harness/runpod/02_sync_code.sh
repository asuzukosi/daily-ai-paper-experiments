#!/usr/bin/env bash
# Sync this package onto a pod (example rsync). Edit HOST / REMOTE_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOH_POD_HOST:-}"
REMOTE_DIR="${HOH_POD_DIR:-/workspace/harness-of-harness}"

if [[ -z "${HOST}" ]]; then
  echo "Set HOH_POD_HOST (and optional HOH_POD_DIR) then re-run."
  echo "Example: export HOH_POD_HOST=root@<ip> HOH_POD_DIR=/workspace/harness-of-harness"
  exit 1
fi

rsync -avz --exclude .venv --exclude artifacts --exclude '__pycache__' \
  "${HOH_ROOT}/" "${HOST}:${REMOTE_DIR}/"
echo "[hoh] synced to ${HOST}:${REMOTE_DIR}"
