#!/usr/bin/env bash
# Submit one camera-ready job onto any idle modern GPU (both partitions; 1080Ti / TitanRTX
# nodes excluded, since recent CUDA wheels may not support them). Usage:
#   bash scripts/camera_ready/submit.sh <job.slurm> KEY=VALUE ...
#   bash scripts/camera_ready/submit.sh pack.slurm EXPS="global_seed42 global_seed43_rep2"
# Every job logs its GPU in preflight; collation reads it from the log.
set -euo pipefail
job="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
exports="ALL"
for kv in "$@"; do exports+=",${kv}"; done
sbatch --partition=Contributors,general --gres=gpu:1 \
  --exclude=GPU3,GPU4,GPU6,GPU8,GPU9,GPU22,GPU42 \
  --export="${exports}" "${here}/${job}"
