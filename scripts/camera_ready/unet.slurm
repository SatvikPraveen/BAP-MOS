#!/usr/bin/env bash
#SBATCH --job-name=cr-unet
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# Train and export one U-Net baseline (experiments/prostate/baselines/unet/README.md recipe).
#   sbatch --export=ALL,SEED=42 scripts/camera_ready/unet.slurm                 # pooled split
#   sbatch --export=ALL,SEED=42,SPLITS=splits_loco/heldout_case2 scripts/camera_ready/unet.slurm
#   sbatch --export=ALL,SEED=42,DATASET=pfus1 scripts/camera_ready/unet.slurm   # PFUS1 schedule
# The export keeps every prediction PNG: predictions/multiclass/*_pred_ids.png are the
# prompt source for the fully automatic BAP-MOS evaluation (export.slurm PROMPT_MASK_DIR).
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${SEED:?set SEED=42|43|44}"
cr_preflight

case "${SEED}" in
  42) suffix=seed42 ;;
  43) suffix=seed43_rep2 ;;
  44) suffix=seed44_rep3 ;;
  *) echo "unsupported SEED=${SEED}" >&2; exit 2 ;;
esac
split_args=()
tag=pooled
data_root=data/prostate/pooled
objective=(--checkpoint-objective-metric ptv_kfold_msd)
if [[ "${DATASET:-pooled}" == pfus1 ]]; then
  data_root=data/bladder/pfus1
  objective+=(--checkpoint-objective-organ Bladder)
  tag=pfus1
fi
if [[ -n "${SPLITS:-}" ]]; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
  split_args=(--splits_subdir "${SPLITS}")
  tag="loco_${SPLITS##*heldout_}"
fi
name="unet_${tag}_${suffix}"
run_root=runs/camera_ready_baselines/${tag%%_*}

resume=()
[[ -f "${run_root}/unet/${name}/last_checkpoint.pth" ]] && resume=(--resume auto)
# Keep every prediction on TRUS (automatic-prompt runs); protocol default (10) on PFUS1.
viz_max=1000
[[ "${DATASET:-pooled}" == pfus1 ]] && viz_max=10
"${PY}" -m bapmos.external_baselines.unet.train_multiclass \
  --data_root "${data_root}" --run_root "${run_root}" --run_name "${name}" \
  --seed "${SEED}" "${objective[@]}" \
  --skip-test-after-train "${split_args[@]}" "${resume[@]}"

export_split=()
[[ -n "${SPLITS:-}" ]] && export_split=(--splits-subdir "${SPLITS}")
"${PY}" -m bapmos.external_baselines.unet.run_test_inference \
  --checkpoint "${run_root}/unet/${name}/best_checkpoint.pth" \
  --output_dir "inference_output/camera_ready/unet/${name}" \
  --viz-pdf-max "${viz_max}" --force-inference-output "${export_split[@]}"
