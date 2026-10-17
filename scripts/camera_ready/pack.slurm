#!/usr/bin/env bash
#SBATCH --job-name=cr-pack
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Several camera-ready suite experiments sharing one GPU. With cached embeddings a
# decoder-only run leaves most of a large GPU idle, so packing raises throughput; each run
# is an independent process with its own seed, log and run folder, and the CPU threads are
# split evenly so metric passes do not compete.
#   bash scripts/camera_ready/submit.sh pack.slurm \
#     EXPS="global_seed42 global_seed43_rep2 global_seed44_rep3"
# Exits non-zero if any packed run fails; rerunning resumes each from last_checkpoint.pth.
# PFUS1: VERSION=bapmos_camera_ready_pfus1_<sam|medsam> DATASET=pfus1
#        RUN_ROOT=runs/bladder/camera_ready/<sam|medsam>  EXPS="pfus1_seed42 ..."
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${EXPS:?set EXPS=\"exp1 exp2 ...\"}"
cr_preflight
# Experiments separated by spaces or '+' (keeper plans cannot carry spaces in a value).
read -r -a exps <<< "${EXPS//+/ }"
threads=$(( ${SLURM_CPUS_PER_TASK:-16} / ${#exps[@]} ))
(( threads < 1 )) && threads=1
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
if printf '%s\n' "${exps[@]}" | grep -q '^loco_'; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
fi

pids=()
for exp in "${exps[@]}"; do
  if [[ -f "${RUN_ROOT:-runs/prostate/camera_ready}/${exp}/.train_done" ]]; then
    echo "pack: ${exp} already done; skipping"
    pids+=("")
    continue
  fi
  resume=()
  [[ -f "${RUN_ROOT:-runs/prostate/camera_ready}/${exp}/last_checkpoint.pth" ]] && resume=(--resume auto)
  log="${CR_ROOT}/logs/run-${exp}.log"
  echo "pack: starting ${exp} (${threads} threads) → ${log}"
  "${PY}" -m bapmos.method.bap_mos_trainer \
    --version "${VERSION:-bapmos_camera_ready}" --dataset "${DATASET:-pooled}" \
    --experiment "${exp}" --run-name "${exp}" \
    --no-wandb --skip-test "${resume[@]}" >> "${log}" 2>&1 &
  pids+=($!)
done

status=0
trained=()
for i in "${!pids[@]}"; do
  [[ -z "${pids[$i]}" ]] && continue
  if wait "${pids[$i]}"; then
    trained+=("${exps[$i]}")
    echo "pack: ${exps[$i]} trained"
  else
    echo "pack: ${exps[$i]} FAILED (see ${CR_ROOT}/logs/run-${exps[$i]}.log)"
    status=1
  fi
done

# Export in the same allocation (TRUS suite only), then mark each run done. The keeper's
# export lines require .train_done, so they only fire for exports this step did not finish.
run_export() {  # <exp> <out> [extra args...]
  local exp="$1" out="$2"; shift 2
  [[ -f "${out}/sites.json" ]] && return 0
  "${PY}" -m bapmos.method.run_test_inference \
    --checkpoint "${RUN_ROOT:-runs/prostate/camera_ready}/${exp}/best_checkpoint.pth" \
    --output_dir "${out}" --force-inference-output --viz-pdf-max 1000 "$@" \
    >> "${CR_ROOT}/logs/run-${exp}.log" 2>&1 || echo "pack: export ${out} failed"
}
for exp in "${trained[@]}"; do
  if [[ "${DATASET:-pooled}" == pooled ]]; then
    io=inference_output/camera_ready
    run_export "${exp}" "${io}/suite/${exp}"
    if [[ "${exp}" == reference_* || "${exp}" == box_only_* ]]; then
      for n in 5 10 20; do
        run_export "${exp}" "${io}/noise${n}/${exp}" --prompt-noise-px "${n}"
      done
      unet="${io}/unet/unet_pooled_seed${exp##*_seed}"
      [[ -f "${unet}/sites.json" ]] && run_export "${exp}" "${io}/auto_unet/${exp}" --prompt-mask-dir "${unet}"
    fi
  fi
  touch "${RUN_ROOT:-runs/prostate/camera_ready}/${exp}/.train_done"
  echo "pack: ${exp} done"
done
exit "${status}"
