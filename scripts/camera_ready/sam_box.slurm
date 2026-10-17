#!/usr/bin/env bash
#SBATCH --job-name=cr-sambox
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# SAM (box) baseline on PFUS1 (docs/RUNNING.md recipe: 100 epochs, patience 20, batch 128)
# with the 2 px box jitter every box-prompted camera-ready method uses. On TRUS the SAM (box)
# row is the Table I multi-organ / box / decoder cell (table1.slurm).
#   bash scripts/camera_ready/submit.sh sam_box.slurm SEED=42
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${SEED:?set SEED=42|43|44}"
cr_preflight
case "${SEED}" in
  42) suffix=seed42 ;; 43) suffix=seed43_rep2 ;; 44) suffix=seed44_rep3 ;;
  *) echo "unsupported SEED=${SEED}" >&2; exit 2 ;;
esac
name="box_pfus1_${suffix}"
run_root=runs/camera_ready_baselines/pfus1/box
resume=()
[[ -f "${run_root}/${name}/last_checkpoint.pth" ]] && resume=(--resume auto)
"${PY}" -m bapmos.multiorgan.train_sam_multiorgan_decoder_box \
  --data_root data/bladder/pfus1 --run_root "${run_root}" --run_name "${name}" --seed "${SEED}" \
  --max_epochs 100 --patience 20 --batch_size 128 --box_jitter_px 2 --skip_test_eval \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder "${resume[@]}"
"${PY}" -m bapmos.multiorgan.run_test_inference \
  --checkpoint "${run_root}/${name}/best_checkpoint.pth" \
  --output_dir "inference_output/camera_ready/box/${name}" \
  --viz-pdf-max 10 --force-inference-output
