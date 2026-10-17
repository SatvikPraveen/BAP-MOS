#!/usr/bin/env bash
#SBATCH --job-name=cr-nnunet
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#
# One nnU-Net replicate (2D, fold 0 = the repository's train/val split, BAP-MOS protocol
# trainer), then per-site prediction and native-grid export. nnU-Net has no seed option;
# REP=1|2|3 are independent runs with their own nnUNet_results, named seed42/43/44 for
# collation like every other row.
#   bash scripts/camera_ready/submit.sh nnunet.slurm DATASET=pooled REP=1
#   bash scripts/camera_ready/submit.sh nnunet.slurm DATASET=pfus1 REP=1 GPUS=4
# Batch sizes follow the paper-era runs (_nnunet_env.sh): TRUS 1 on one GPU; PFUS1 a global
# 128 under nnU-Net's DDP (GPUS=4, 32 per GPU), where batch Dice gathers tp/fp/fn over all
# ranks, so the loss equals that of a single batch 128.
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
source "${REPO}/scripts/camera_ready/_nnunet_env.sh"
: "${REP:?set REP=1|2|3}"
GPUS="${GPUS:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cr_preflight
( cd "${NN}" && sha256sum --quiet -c "${CR_ROOT}/NNUNET_${DATASET}_SHA256.txt" ) \
  || { echo "nnunet: raw inputs differ from NNUNET_${DATASET}_SHA256.txt; refusing" >&2; exit 3; }
case "${REP}" in
  1) seed=42; suffix=seed42 ;; 2) seed=43; suffix=seed43_rep2 ;; 3) seed=44; suffix=seed44_rep3 ;;
  *) echo "unsupported REP=${REP}" >&2; exit 2 ;;
esac
export nnUNet_results="${NN}/results_${suffix}"
mkdir -p "${nnUNet_results}"
name="nnunet_${DATASET}_${suffix}"
out="inference_output/camera_ready/nnunet/${name}"

done_flag="${nnUNet_results}/.train_done"
if [[ ! -f "${done_flag}" ]]; then
  cont=()
  [[ -f "${nnUNet_results}/${NN_DS}/nnUNetTrainerBapMosProtocol__nnUNetPlans__2d/fold_0/checkpoint_latest.pth" ]] && cont=(--c)
  "${NNBIN}/nnUNetv2_train" "${NN_ID}" 2d 0 -tr nnUNetTrainerBapMosProtocol -num_gpus "${GPUS}" "${cont[@]}"
  touch "${done_flag}"
fi

for site in "${NN_SITES[@]}"; do
  test_ds="${NN}/raw_test/${site}/${NN_DS}"
  pred="${NN}/pred_${suffix}/${site}"
  "${NNBIN}/nnUNetv2_predict" -i "${test_ds}/imagesTs" -o "${pred}" \
    -d "${NN_ID}" -c 2d -f 0 -tr nnUNetTrainerBapMosProtocol -chk checkpoint_best.pth
  split_args=()
  [[ "${site}" != test ]] && split_args=(--splits-subdir "site_tests/${site}")
  "${NNPY}" -m bapmos.external_baselines.nnunet2d.run_test_inference \
    --predictions_dir "${pred}" --case_mapping "${test_ds}/case_mapping.json" \
    --data_root "${NN_DATA_ROOT}" --output_dir "${out}" "${split_args[@]}" \
    --run-name "${DATASET}_${suffix}" --experiment-seed "${seed}" --force-inference-output \
    --viz-pdf-max 10
done
