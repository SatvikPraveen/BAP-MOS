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
    export NNUNET_MAX_EPOCHS=300 NNUNET_PATIENCE=40
    ;;
  pfus1)
    NN_DATA_ROOT=data/bladder/pfus1; NN_ID=503; NN_NAME=BapMosPfus1TrainVal
    NN_SITES=(test)
    export NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder
    ;;
  *) echo "unsupported DATASET=${DATASET}" >&2; exit 2 ;;
esac
NN_DS="Dataset${NN_ID}_${NN_NAME}"
export BAPMOS_NNUNET_DATA_ROOT="${NN_DATA_ROOT}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}"
