#!/usr/bin/env bash
#SBATCH --job-name=cr-export-box
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Test export for a fixed-box SAM or MedSAM decoder with paired test-time box perturbation
# (the same offsets as export.slurm with NOISE_MODE=shift_paired|loose_paired):
#   bash scripts/camera_ready/submit.sh export_box_baseline.slurm KIND=sam \
#     CKPT=runs/.../best_checkpoint.pth OUT=inference_output/camera_ready/<group>/<name> \
#     NOISE=20 NOISE_MODE=loose_paired
# KIND=sam: multi-organ SAM box decoder (Table I cell); KIND=medsam: MedSAM (box) baseline.
# Automatic prompts instead: PROMPT_MASK_DIR=<U-Net export root> (boxes from its *_pred_ids.png),
# with NOISE unset (0).
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${KIND:?set KIND=sam|medsam}" "${CKPT:?set CKPT=<best_checkpoint.pth>}" "${OUT:?set OUT=<output dir>}"
if [[ -z "${PROMPT_MASK_DIR:-}" ]]; then
  : "${NOISE:?set NOISE=<px>}" "${NOISE_MODE:?set NOISE_MODE=shift_paired|loose_paired}"
fi
cr_preflight
case "${KIND}" in
  sam)    module=bapmos.multiorgan.run_test_inference ;;
  medsam) module=bapmos.external_baselines.medsam_init.run_test_inference ;;
  *) echo "unsupported KIND=${KIND}" >&2; exit 2 ;;
esac
extra=()
[[ -n "${PROMPT_MASK_DIR:-}" ]] && extra+=(--prompt-mask-dir "${PROMPT_MASK_DIR}")
"${PY}" -m "${module}" --checkpoint "${CKPT}" --output_dir "${OUT}" --force-inference-output \
  --viz-pdf-max "${VIZ_MAX:-1000}" --prompt-noise-px "${NOISE:-0}" --prompt-noise-mode "${NOISE_MODE:-shift_paired}" \
  "${extra[@]}"
