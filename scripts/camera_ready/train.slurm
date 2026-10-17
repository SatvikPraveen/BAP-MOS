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
# All training runs use one GPU type (A100) so throughput, not results, varies by node.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${EXP:?set EXP=<experiment name>}"
cr_preflight
if [[ "${EXP}" == loco_* ]]; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
fi

resume=()
if [[ -f "runs/prostate/camera_ready/${EXP}/last_checkpoint.pth" ]]; then
  resume=(--resume auto)
fi
start=$(date +%s)
"${PY}" -m bapmos.method.bap_mos_trainer \
  --version bapmos_camera_ready --dataset pooled \
  --experiment "${EXP}" --run-name "${EXP}" \
  --no-wandb --skip-test "${resume[@]}"
echo "train: ${EXP} finished in $(( $(date +%s) - start )) s"
