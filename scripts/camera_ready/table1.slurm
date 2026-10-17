#!/usr/bin/env bash
#SBATCH --job-name=cr-table1
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# One Table I cell (decoder-only vs full-model fine-tuning) on the pooled split, then its
# native-grid test export:
#   sbatch --export=ALL,TASK=multi,PROMPT=box,SCOPE=full,SEED=42 scripts/camera_ready/table1.slurm
# TASK=multi|single (single = PTV only), PROMPT=box|points, SCOPE=decoder|full, SEED=42|43|44.
# Fixed-prompt SAM trainers with their own protocol (CE + Dice, fixed 20 px ring for points,
# 2 px train box jitter); the cell differs from its neighbours only in TASK/PROMPT/SCOPE.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${TASK:?}" "${PROMPT:?}" "${SCOPE:?}" "${SEED:?}"
cr_preflight

case "${SEED}" in
  42) suffix=seed42 ;;
  43) suffix=seed43_rep2 ;;
  44) suffix=seed44_rep3 ;;
  *) echo "unsupported SEED=${SEED}" >&2; exit 2 ;;
esac
cell="${TASK}_${PROMPT}_${SCOPE}"
name="${cell}_${suffix}"
run_root="runs/prostate/camera_ready_table1/${cell}"
out="inference_output/camera_ready/table1/${name}"

args=(--data_root data/prostate/pooled --run_root "${run_root}" --run_name "${name}"
      --seed "${SEED}" --finetune-scope "${SCOPE}" --skip_test_eval)
[[ "${PROMPT}" == box ]] && args+=(--box_jitter_px 2)
[[ -f "${run_root}/${name}/last_checkpoint.pth" ]] && args+=(--resume auto)

case "${TASK}" in
  multi)
    module="bapmos.multiorgan.train_sam_multiorgan_decoder_${PROMPT}"
    export_module=bapmos.multiorgan.run_test_inference
    args+=(--checkpoint-objective-metric ptv_kfold_msd)
    ;;
  single)
    module="bapmos.legacy.singleorgan.train_sam_singleorgan_decoder_${PROMPT}"
    export_module=bapmos.legacy.singleorgan.run_test_inference
    args+=(--organ ptv)
    ;;
  *) echo "unsupported TASK=${TASK}" >&2; exit 2 ;;
esac

start=$(date +%s)
"${PY}" -m "${module}" "${args[@]}"
echo "table1: ${name} trained in $(( $(date +%s) - start )) s"
"${PY}" -m "${export_module}" --checkpoint "${run_root}/${name}/best_checkpoint.pth" \
  --output_dir "${out}" --viz-pdf-max 1000 --force-inference-output
