#!/usr/bin/env bash
# Sync / refresh checkout on an existing pod (manual).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "[procedural-graphs] Code root: ${ROOT}"
echo "If using git: git pull origin paper/procedural-graphs-2026-09-15"
echo "Or rsync from your workstation into this directory."
ls -la "${ROOT}" | head
