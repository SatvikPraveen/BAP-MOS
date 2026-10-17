#!/usr/bin/env bash
# Submit one camera-ready job with this cluster's site settings. Usage:
#   bash scripts/camera_ready/submit.sh <job.slurm> KEY=VALUE ...
#   bash scripts/camera_ready/submit.sh pack.slurm EXPS="global_seed42 global_seed43_rep2"
#   SBATCH_ARGS="--cpus-per-task=24 --mem=120G" bash scripts/camera_ready/submit.sh pack.slurm ...
# Job scripts hold only portable resources (CPUs, memory, time); partition, account, QOS,
# GPU request and log path are set here. Every job logs its GPU in preflight.
#   GAIVI: any modern GPU on Contributors/general (1080Ti, TitanRTX and the A40 nodes
#          GPU43/GPU44 excluded; the A40 nodes fail CUDA init with the cu129 build)
#   Anvil: one A100 on the ai partition, account cis260833-ai
set -euo pipefail
job="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
cr_root="$(cd "${here}/../../.." && pwd)"
exports="ALL,CR_ROOT=${cr_root}"
for kv in "$@"; do exports+=",${kv}"; done
mkdir -p "${cr_root}/logs"
case "$(hostname -f)" in
  *gaivi*|GPU*)
    site=(--partition=Contributors,general --qos=maxjobs50 --gres=gpu:1
          --exclude=GPU3,GPU4,GPU6,GPU8,GPU9,GPU22,GPU42,GPU43,GPU44) ;;
  *anvil*)
    site=(--partition=ai --account=cis260833-ai --gres=gpu:1) ;;
  *) echo "submit.sh: unknown cluster $(hostname -f)" >&2; exit 2 ;;
esac
extra=()
read -r -a extra <<< "${SBATCH_ARGS:-}"
sbatch "${site[@]}" "${extra[@]}" --output="${cr_root}/logs/%x-%j.out" --export="${exports}" "${here}/${job}"
