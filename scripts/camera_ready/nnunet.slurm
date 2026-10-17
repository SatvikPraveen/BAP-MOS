#!/usr/bin/env bash
#SBATCH --job-name=cr-nnunet
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# One nnU-Net replicate (2D, fold 0 = the repository's train/val split, BAP-MOS protocol
# trainer), then per-site prediction and native-grid export. nnU-Net has no seed option;
# REP=1|2|3 are independent runs with their own nnUNet_results, named seed42/43/44 for
# collation like every other row.
#   bash scripts/camera_ready/submit.sh nnunet.slurm DATASET=pooled REP=1
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
source "${REPO}/scripts/camera_ready/_nnunet_env.sh"
: "${REP:?set REP=1|2|3}"
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

cont=()
[[ -f "${nnUNet_results}/${NN_DS}/nnUNetTrainerBapMosProtocol__nnUNetPlans__2d/fold_0/checkpoint_latest.pth" ]] && cont=(--c)
"${NNBIN}/nnUNetv2_train" "${NN_ID}" 2d 0 -tr nnUNetTrainerBapMosProtocol "${cont[@]}"

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
