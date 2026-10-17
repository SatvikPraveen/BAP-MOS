#!/usr/bin/env bash
#SBATCH --job-name=cr-pack
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# Several camera-ready suite experiments sharing one GPU. With cached embeddings a
# decoder-only run leaves most of a large GPU idle, so packing raises throughput; each run
# is an independent process with its own seed, log and run folder, and the CPU threads are
# split evenly so metric passes do not compete.
#   sbatch --export=ALL,EXPS="global_seed42 global_seed43_rep2 global_seed44_rep3" \
#     --gres=gpu:H100:1 scripts/camera_ready/pack.slurm
# Exits non-zero if any packed run fails; rerunning resumes each from last_checkpoint.pth.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
: "${EXPS:?set EXPS=\"exp1 exp2 ...\"}"
cr_preflight
read -r -a exps <<< "${EXPS}"
threads=$(( ${SLURM_CPUS_PER_TASK:-16} / ${#exps[@]} ))
(( threads < 1 )) && threads=1
export OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads} OPENBLAS_NUM_THREADS=${threads}
if printf '%s\n' "${exps[@]}" | grep -q '^loco_'; then
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
fi

pids=()
for exp in "${exps[@]}"; do
  resume=()
  [[ -f "runs/prostate/camera_ready/${exp}/last_checkpoint.pth" ]] && resume=(--resume auto)
  log="${CR_ROOT}/logs/run-${exp}.log"
  echo "pack: starting ${exp} (${threads} threads) → ${log}"
  "${PY}" -m bapmos.method.bap_mos_trainer \
    --version bapmos_camera_ready --dataset pooled \
    --experiment "${exp}" --run-name "${exp}" \
    --no-wandb --skip-test "${resume[@]}" >> "${log}" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "pack: ${exps[$i]} done"
  else
    echo "pack: ${exps[$i]} FAILED (see ${CR_ROOT}/logs/run-${exps[$i]}.log)"
    status=1
  fi
done
exit "${status}"
