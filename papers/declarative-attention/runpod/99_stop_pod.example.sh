#!/usr/bin/env bash
# EXAMPLE — stop / remove a pod. Replace <POD_ID>. Does nothing by default.
set -euo pipefail

POD_ID="${1:-}"
if [[ -z "${POD_ID}" ]]; then
  echo "Usage: $0 <POD_ID>"
  echo "Example (uncomment to execute):"
  echo "  # runpodctl stop pod <POD_ID>"
  echo "  # runpodctl remove pod <POD_ID>"
  exit 1
fi

echo "Would stop pod ${POD_ID}. Uncomment below to actually spend/stop."
# runpodctl stop pod "${POD_ID}"
# runpodctl remove pod "${POD_ID}"
