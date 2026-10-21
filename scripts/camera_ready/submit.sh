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
# GPUS=N (one node) requests N GPUs instead of one; nnunet.slurm then trains with DDP.
#   GAIVI: only nodes that have run camera-ready jobs cleanly: GPU48 (H100), GPU52 (A6000),
#          GPU53 (L40S), GPU54 (6000Ada), GPU56/GPU57 (RTX PRO 6000). Every other node is
#          excluded: GPU2/GPU43/GPU44 drivers fail cu129 kernels, GPU49 had uncorrectable
#          ECC errors, 1080Ti/TitanRTX are too old, and the rest are untested.
#          ALLOW_NODES=GPU1,GPU46 (3 x A100) admits untested nodes for one job; its preflight
#          still refuses a node whose CUDA kernels or data mount fail.
#   Anvil: one A100 on the ai partition, account cis260833-ai
#   Delta: one A100 on gpuA100x4, account biak-delta-gpu
# TRUS (clinical) jobs run on GAIVI only; Anvil and Delta take PFUS1 jobs.
set -euo pipefail
job="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
cr_root="$(cd "${here}/../../.." && pwd)"
exports="ALL,CR_ROOT=${cr_root}"
gpus=1
also_exclude=""   # EXCLUDE_NODES=a,b: keep this job off further nodes (e.g. small-memory GPUs)
allow=""          # ALLOW_NODES=a,b: take these off the site exclusion list for this job
for kv in "$@"; do
  exports+=",${kv}"
  [[ "${kv}" == GPUS=* ]] && gpus="${kv#GPUS=}"
  [[ "${kv}" == EXCLUDE_NODES=* ]] && also_exclude=",${kv#EXCLUDE_NODES=}"
  [[ "${kv}" == ALLOW_NODES=* ]] && allow="${kv#ALLOW_NODES=}"
done
mkdir -p "${cr_root}/logs"
if [[ ! "$(hostname -f)" =~ gaivi|^GPU ]] && [[ "${exports}" != *DATASET=pfus1* ]]; then
  echo "submit.sh: only PFUS1 jobs (DATASET=pfus1) may run off GAIVI; TRUS stays on GAIVI" >&2
  exit 2
fi
case "$(hostname -f)" in
  *gaivi*|GPU*)
    exclude="GPU1,GPU2,GPU3,GPU4,GPU6,GPU8,GPU9,GPU22,GPU42,GPU43,GPU44,GPU45,GPU46,GPU47,GPU49,GPU50,GPU51,GPU55${also_exclude}"
    for n in ${allow//,/ }; do exclude="$(tr ',' '\n' <<< "${exclude}" | grep -vx "${n}" | paste -sd, -)"; done
    site=(--partition=YES,Contributors,general --qos=maxjobs50 --gres=gpu:${gpus} --exclude="${exclude}") ;;
  *anvil*)
    site=(--partition=ai --account=cis260833-ai --gres=gpu:${gpus}) ;;
  *delta*|dt-*)
    site=(--partition=gpuA100x4 --account=biak-delta-gpu --gres=gpu:${gpus}) ;;
  *) echo "submit.sh: unknown cluster $(hostname -f)" >&2; exit 2 ;;
esac
extra=()
read -r -a extra <<< "${SBATCH_ARGS:-}"
sbatch "${site[@]}" "${extra[@]}" --output="${cr_root}/logs/%x-%j.out" --export="${exports}" "${here}/${job}"
