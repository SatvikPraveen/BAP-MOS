#!/usr/bin/env bash
# nnU-Net settings shared by nnunet_prep.slurm and nnunet.slurm (sourced after _env.sh).
# env_nnunet is a clone of the paper-era nnU-Net env with the branch's
# nnUNetTrainerBapMosProtocol linked in; its versions are in env_nnunet_freeze.txt.
NNBIN="${CR_ROOT}/env_nnunet/bin"
NNPY="${NNBIN}/python"
NN="${CR_ROOT}/nnunet/${DATASET:?set DATASET=pooled|pfus1}"
export nnUNet_raw="${NN}/raw" nnUNet_preprocessed="${NN}/preprocessed"
export nnUNet_compile=f nnUNet_wandb_enabled=0
case "${DATASET}" in
  pooled)
    NN_DATA_ROOT=data/prostate/pooled; NN_ID=505; NN_NAME=BapMosPooledTrainVal
    NN_SITES=(simulation case1 case2)
    # Paper-era TRUS nnU-Net: batch 1 on one GPU (640x768 patches).
    export NNUNET_MAX_EPOCHS=300 NNUNET_PATIENCE=40 NNUNET_BATCH_SIZE=1
    ;;
  pfus1)
    NN_DATA_ROOT=data/bladder/pfus1; NN_ID=503; NN_NAME=BapMosPfus1TrainVal
    NN_SITES=(test)
    # PFUS1 nnU-Net: global batch 128 under DDP, split over GPUS (4 x 32, or 2 x 64 on 80-96 GB GPUs).
    # Optimizer as every PFUS1 model and the U-Net baseline: Adam, flat lr 1e-4, no weight decay.
    export NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder NNUNET_BATCH_SIZE=128 NNUNET_OPTIMIZER=adam
    # One validation epoch scores 2175 frames; save checkpoint_latest every epoch so each
    # short job resumes where the previous one stopped.
    export NNUNET_SAVE_EVERY=1
    # The 32-CPU job scores validation with 24 workers (order-preserving; same numbers as serial).
    export BAPMOS_METRIC_WORKERS=24
    ;;
  *) echo "unsupported DATASET=${DATASET}" >&2; exit 2 ;;
esac
NN_DS="Dataset${NN_ID}_${NN_NAME}"
export BAPMOS_NNUNET_DATA_ROOT="${NN_DATA_ROOT}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}"

# Run a training command; once fold_dir holds checkpoint_final and a training log reports the
# final validation, allow a 2-minute grace for the DDP processes to exit, then stop them
# (they can hang in teardown and hold the GPUs until the time limit). Any other exit status
# is returned unchanged.
nn_kill_tree() {
  local child
  for child in $(pgrep -P "$1"); do nn_kill_tree "${child}" "$2"; done
  kill "-$2" "$1" 2>/dev/null || true
}
nn_train_reaped() {
  local fold_dir="$1"; shift
  "$@" &
  local pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${NN_REAP_POLL:-60}"
    if [[ -f "${fold_dir}/checkpoint_final.pth" ]] \
        && grep -qs "Mean Validation Dice" "${fold_dir}"/training_log_*.txt; then
      local t=0
      while kill -0 "${pid}" 2>/dev/null && (( t < ${NN_REAP_GRACE:-120} )); do sleep 5; t=$((t + 5)); done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "nnunet: training finished but its processes did not exit; stopping them" >&2
        nn_kill_tree "${pid}" TERM
        sleep 15
        nn_kill_tree "${pid}" KILL
      fi
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
  done
  wait "${pid}"
}
