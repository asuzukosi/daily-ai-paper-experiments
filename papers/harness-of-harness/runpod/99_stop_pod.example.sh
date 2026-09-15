#!/usr/bin/env bash
# EXAMPLE ONLY — does NOT stop/destroy a paid pod unless you uncomment.
set -euo pipefail

echo "=== EXAMPLE stop-pod script (no-op unless you uncomment) ==="
cat <<'HINT'
# List pods
# runpodctl get pod

# Stop (keep storage) — UNCOMMENT AND EDIT:
# runpodctl stop pod <POD_ID>

# Remove entirely — DANGEROUS:
# runpodctl remove pod <POD_ID>
HINT
echo "Exiting without stopping anything."
exit 0
