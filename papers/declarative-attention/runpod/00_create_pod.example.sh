#!/usr/bin/env bash
# EXAMPLE ONLY — does NOT auto-create a paid pod.
# Edit GPU type / template / sizes, then run manually when you intend to spend.
#
# Prerequisites:
#   export RUNPOD_API_KEY=...
#   runpodctl installed (https://docs.runpod.io/references/runpodctl)
set -euo pipefail

echo "=== EXAMPLE create-pod script (will not run create unless you uncomment) ==="
echo "Review RunPod CLI docs for your CLI version before spending money."
echo

cat <<'HINT'
# List GPUs
runpodctl get gpu

# Example create (UNCOMMENT AND EDIT before use):
# runpodctl create pod \
#   --name dair-da-qwen25-7b \
#   --gpuType "NVIDIA GeForce RTX 4090" \
#   --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
#   --volumeSize 80 \
#   --containerDiskSize 40 \
#   --ports "8888/http,22/tcp"

# Prefer A100 80GB for longctx:
#   --gpuType "NVIDIA A100 80GB"
HINT

echo
echo "This script exits without creating anything."
exit 0
