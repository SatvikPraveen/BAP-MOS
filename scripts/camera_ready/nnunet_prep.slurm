#!/usr/bin/env bash
#SBATCH --job-name=cr-nnprep
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#
# Shared nnU-Net prep, once per dataset, before any replicate (nnunet.slurm):
#   train+val export -> nnUNet_raw, plan_and_preprocess -> nnUNet_preprocessed, one private
#   test folder per test site under ${NN}/raw_test/<site>/ (so replicates never rewrite the
#   shared raw dataset), and NNUNET_<dataset>_SHA256.txt pinning the raw inputs.
#   bash scripts/camera_ready/submit.sh nnunet_prep.slurm DATASET=pooled
#   bash scripts/camera_ready/submit.sh nnunet_prep.slurm DATASET=pfus1
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
source "${REPO}/scripts/camera_ready/_nnunet_env.sh"
cr_preflight
manifest="${CR_ROOT}/NNUNET_${DATASET}_SHA256.txt"
if [[ -f "${manifest}" && "${FORCE:-0}" != "1" ]]; then
  echo "nnunet_prep: ${manifest} exists; use FORCE=1 to rebuild" >&2
  exit 1
fi

"${NNPY}" -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root "${NN_DATA_ROOT}" --output_parent "${nnUNet_raw}" \
  --dataset_id "${NN_ID}" --dataset_name "${NN_NAME}"
"${NNBIN}/nnUNetv2_plan_and_preprocess" -d "${NN_ID}" -c 2d -np 8 --verify_dataset_integrity

for site in "${NN_SITES[@]}"; do
  parent="${NN}/raw_test/${site}"
  rm -rf "${parent:?}/${NN_DS}"
  mkdir -p "${parent}"
  cp -r "${nnUNet_raw}/${NN_DS}" "${parent}/"
  split_args=()
  [[ "${site}" != test ]] && split_args=(--splits_subdir "site_tests/${site}")
  "${NNPY}" -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
    --data_root "${NN_DATA_ROOT}" --output_parent "${parent}" \
    --dataset_id "${NN_ID}" --dataset_name "${NN_NAME}" "${split_args[@]}" --test_images_only
done

( cd "${NN}" && find raw raw_test -type f | LC_ALL=C sort | xargs sha256sum ) > "${manifest}"
echo "nnunet_prep: pinned $(wc -l < "${manifest}") files -> ${manifest}"
