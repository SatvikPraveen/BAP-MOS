#!/usr/bin/env bash
#SBATCH --job-name=cr-pfembed
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#
# Shared PFUS1 prep (once, before PFUS1 jobs and before "prep.sh pin pfus1"): frozen SAM and
# MedSAM embeddings for every PFUS1 frame, float32 so cached BAP-MOS runs see exactly the
# encoder outputs the on-the-fly baselines compute.
#   bash scripts/camera_ready/submit.sh pfus1_prep_embeddings.slurm DATASET=pfus1
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
for init in meta medsam; do
  if [[ "${init}" == meta ]]; then
    ckpt=models/sam_base/sam_vit_b_01ec64.pth; out=data/bladder/pfus1/sam_embeddings/sam_vit_b
  else
    ckpt=models/medsam/medsam_vit_b.pth; out=data/bladder/pfus1/sam_embeddings/medsam_vit_b
  fi
  "${PY}" -m bapmos.preprocess.bladder.precompute_sam_embeddings \
    --data_root data/bladder/pfus1 --init "${init}" --sam_checkpoint "${ckpt}" \
    --output_dir "${out}" --splits train,val,test --dtype float32 --device cuda
done
