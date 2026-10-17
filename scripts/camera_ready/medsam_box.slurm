#!/usr/bin/env bash
#SBATCH --job-name=cr-medsam
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# MedSAM (box) baseline: MedSAM-initialised SAM ViT-B, decoder fine-tuned with fixed box
# prompts (docs/RUNNING.md recipe). Box jitter is 2 px, the same for every box-prompted method.
#   sbatch --export=ALL,SEED=42 scripts/camera_ready/medsam_box.slurm               # TRUS
#   sbatch --export=ALL,SEED=42,DATASET=pfus1 scripts/camera_ready/medsam_box.slurm # PFUS1
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${SEED:?set SEED=42|43|44}"
cr_preflight
case "${SEED}" in
  42) suffix=seed42 ;; 43) suffix=seed43_rep2 ;; 44) suffix=seed44_rep3 ;;
  *) echo "unsupported SEED=${SEED}" >&2; exit 2 ;;
esac
case "${DATASET:-pooled}" in
  pooled) data_root=data/prostate/pooled; objective=(--checkpoint-objective-metric ptv_kfold_msd) ;;
  pfus1)  data_root=data/bladder/pfus1
          objective=(--checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder) ;;
  *) echo "unsupported DATASET=${DATASET}" >&2; exit 2 ;;
esac
name="medsam_${DATASET:-pooled}_${suffix}"
run_root="runs/camera_ready_baselines/${DATASET:-pooled}"
resume=()
[[ -f "${run_root}/medsam_init/${name}/last_checkpoint.pth" ]] && resume=(--resume auto)

"${PY}" -m bapmos.external_baselines.medsam_init.train_decoder_boxes \
  --data_root "${data_root}" --medsam_checkpoint models/medsam/medsam_vit_b.pth \
  --run_root "${run_root}" --run_name "${name}" --seed "${SEED}" --box_noise_pixels 2 \
  --skip-test-after-train "${objective[@]}" "${resume[@]}"
"${PY}" -m bapmos.external_baselines.medsam_init.run_test_inference \
  --checkpoint "${run_root}/medsam_init/${name}/best_checkpoint.pth" \
  --output_dir "inference_output/camera_ready/medsam/${name}" \
  --viz-pdf-max 1000 --force-inference-output
