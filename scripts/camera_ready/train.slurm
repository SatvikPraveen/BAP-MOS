#!/usr/bin/env bash
#SBATCH --job-name=cr-train
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# Train one camera-ready experiment (experiments/prostate/camera_ready/version.yaml):
#   sbatch --export=ALL,EXP=global_seed42 scripts/camera_ready/train.slurm
# Resumes from last_checkpoint.pth when the run folder already has one.
# PFUS1: also set VERSION=bapmos_camera_ready_pfus1_<sam|medsam> DATASET=pfus1
#        RUN_ROOT=runs/bladder/camera_ready/<sam|medsam>.
# Default A100; submit with --gres=gpu:<TYPE>:1 to use another modern GPU (not 1080Ti/TitanRTX).
# preflight logs the device, so every run records the GPU it used.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${EXP:?set EXP=<experiment name>}"
cr_preflight
if [[ "${EXP}" == loco_* ]]; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
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
echo "train: ${EXP} finished in $(( $(date +%s) - start )) s"
