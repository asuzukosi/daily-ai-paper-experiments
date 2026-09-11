#!/usr/bin/env bash
# Hints for syncing this experiment folder onto a RunPod instance.
set -euo pipefail

cat <<'EOFHINT'
# From your Mac / laptop (after noting SSH host from the RunPod UI):
#
#   rsync -avz --exclude artifacts --exclude '.git' \
#     papers/declarative-attention/ \
#     root@<POD_HOST>:/workspace/daily-ai-paper-experiments/papers/declarative-attention/
#
# Or scp a tarball:
#   tar czf da.tar.gz papers/declarative-attention
#   scp da.tar.gz root@<POD_HOST>:/workspace/
#   ssh root@<POD_HOST> 'mkdir -p /workspace/daily-ai-paper-experiments/papers && tar xzf /workspace/da.tar.gz -C /workspace/daily-ai-paper-experiments/papers --strip-components=1'
#
# Then on the pod:
#   cd /workspace/daily-ai-paper-experiments/papers/declarative-attention
#   bash runpod/01_setup_env.sh
#   bash runpod/03_run_smoke.sh
EOFHINT
