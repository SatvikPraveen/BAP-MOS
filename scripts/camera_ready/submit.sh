#!/usr/bin/env bash
# Submit one camera-ready job with this cluster's site settings. Usage:
#   bash scripts/camera_ready/submit.sh <job.slurm> KEY=VALUE ...
#   bash scripts/camera_ready/submit.sh pack.slurm EXPS="global_seed42 global_seed43_rep2"
#   SBATCH_ARGS="--cpus-per-task=24 --mem=120G" bash scripts/camera_ready/submit.sh pack.slurm ...
# Job scripts hold only portable resources (CPUs, memory, time); partition, account, QOS,
# GPU request and log path are set here. Every job logs its GPU in preflight.
# Time limits are short (2-3x the measured run time) so jobs backfill instead of holding
# the queue; every job resumes from its last checkpoint, so a timed-out job is resubmitted.
# Longer cells (e.g. Table I full model): SBATCH_ARGS="--time=04:00:00".
#   GAIVI: any modern GPU on nopreempt/YES/Contributors/general (higher partition tiers
#          first; 1080Ti, TitanRTX and the A40 nodes
#          GPU43/GPU44 excluded; the A40 nodes fail CUDA init with the cu129 build; GPU49
#          excluded after uncorrectable ECC errors)
#   Anvil: one A100 on the ai partition, account cis260833-ai
#   Delta: one A100 on gpuA100x4, account biak-delta-gpu
# TRUS (clinical) jobs run on GAIVI only; Anvil and Delta take PFUS1 jobs.
set -euo pipefail
job="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
cr_root="$(cd "${here}/../../.." && pwd)"
exports="ALL,CR_ROOT=${cr_root}"
for kv in "$@"; do exports+=",${kv}"; done
mkdir -p "${cr_root}/logs"
if [[ ! "$(hostname -f)" =~ gaivi|^GPU ]] && [[ "${exports}" != *DATASET=pfus1* ]]; then
  echo "submit.sh: only PFUS1 jobs (DATASET=pfus1) may run off GAIVI; TRUS stays on GAIVI" >&2
  exit 2
fi
case "$(hostname -f)" in
  *gaivi*|GPU*)
    site=(--partition=nopreempt,YES,Contributors,general --qos=maxjobs50 --gres=gpu:1
          --exclude=GPU3,GPU4,GPU6,GPU8,GPU9,GPU22,GPU42,GPU43,GPU44,GPU49) ;;
  *anvil*)
    site=(--partition=ai --account=cis260833-ai --gres=gpu:1) ;;
  *delta*|dt-*)
    site=(--partition=gpuA100x4 --account=biak-delta-gpu --gres=gpu:1) ;;
  *) echo "submit.sh: unknown cluster $(hostname -f)" >&2; exit 2 ;;
esac
extra=()
read -r -a extra <<< "${SBATCH_ARGS:-}"
sbatch "${site[@]}" "${extra[@]}" --output="${cr_root}/logs/%x-%j.out" --export="${exports}" "${here}/${job}"
