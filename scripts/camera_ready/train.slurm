#!/usr/bin/env bash
#SBATCH --job-name=cr-train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Train one camera-ready experiment (experiments/prostate/camera_ready/version.yaml):
#   bash scripts/camera_ready/submit.sh train.slurm EXP=global_seed42
# Resumes from last_checkpoint.pth when the run folder already has one.
# PFUS1: also set VERSION=bapmos_camera_ready_pfus1_<sam|medsam> DATASET=pfus1
#        RUN_ROOT=runs/bladder/camera_ready/<sam|medsam>.
# Submit through submit.sh (site settings); preflight logs the GPU each run used.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${EXP:?set EXP=<experiment name>}"
cr_preflight
if [[ "${EXP}" == loco_* ]]; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
fi

if [[ -f "${RUN_ROOT:-runs/prostate/camera_ready}/${EXP}/.train_done" ]]; then
  echo "train: ${EXP} already done; nothing to do"
  exit 0
fi
resume=()
if [[ -f "${RUN_ROOT:-runs/prostate/camera_ready}/${EXP}/last_checkpoint.pth" ]]; then
  resume=(--resume auto)
fi
start=$(date +%s)
"${PY}" -m bapmos.method.bap_mos_trainer \
  --version "${VERSION:-bapmos_camera_ready}" --dataset "${DATASET:-pooled}" \
  --experiment "${EXP}" --run-name "${EXP}" \
  --no-wandb --skip-test "${resume[@]}"
touch "${RUN_ROOT:-runs/prostate/camera_ready}/${EXP}/.train_done"
echo "train: ${EXP} finished in $(( $(date +%s) - start )) s"
