#!/usr/bin/env bash
# EXAMPLE ONLY — do NOT auto-run; creates a paid RunPod pod.
# Edit GPU type / image / volume, then run manually if you accept the cost.
set -euo pipefail
echo "=== EXAMPLE: create RunPod pod for WikiSkill (Qwen2.5-7B-Instruct) ==="
echo "Suggested: 1x RTX 4090 / A6000 / L40S, ≥50GB disk, PyTorch + CUDA image"
echo "After create: sync code, then bash runpod/01_setup_env.sh"
echo "This script intentionally does not call the RunPod API."
exit 0
