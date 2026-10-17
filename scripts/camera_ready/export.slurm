#!/usr/bin/env bash
#SBATCH --job-name=cr-export
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Test export for one trained BAP-MOS-family checkpoint, scored on the native grid.
#   sbatch --export=ALL,CKPT=runs/.../best_checkpoint.pth,OUT=inference_output/camera_ready/<name> \
#     scripts/camera_ready/export.slurm
# Optional: NOISE=<px> (prompt perturbation), PROMPT_MASK_DIR=<dir of *_pred_ids.png>,
# SPLITS=<splits_subdir>. Every prediction PNG is kept (VIZ_MAX defaults to 1000) so
# per-slice statistics and prompt-source runs can reuse them.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${CKPT:?set CKPT=<best_checkpoint.pth>}"
: "${OUT:?set OUT=<output dir>}"
cr_preflight

args=(--checkpoint "${CKPT}" --output_dir "${OUT}" --force-inference-output
      --viz-pdf-max "${VIZ_MAX:-1000}" --prompt-noise-px "${NOISE:-0}")
[[ -n "${PROMPT_MASK_DIR:-}" ]] && args+=(--prompt-mask-dir "${PROMPT_MASK_DIR}")
[[ -n "${SPLITS:-}" ]] && args+=(--splits-subdir "${SPLITS}")
"${PY}" -m bapmos.method.run_test_inference "${args[@]}"
