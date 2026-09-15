#!/usr/bin/env bash
# Sync local package to a remote RunPod host via rsync/scp.
# Usage: REMOTE=root@IP:~/wikiskill bash runpod/02_sync_code.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-}"
if [[ -z "${REMOTE}" ]]; then
  echo "Set REMOTE=user@host:/path/to/wikiskill" >&2
  exit 1
fi
rsync -avz --exclude '.venv' --exclude 'artifacts/*' --exclude '__pycache__' \
  --exclude '*.egg-info' "${ROOT}/" "${REMOTE}/"
echo "[wikiskill] synced to ${REMOTE}"
