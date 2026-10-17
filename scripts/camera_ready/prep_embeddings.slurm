#!/usr/bin/env bash
#SBATCH --job-name=cr-embed
#SBATCH --partition=Contributors
#SBATCH --qos=maxjobs50
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/satvikpraveen/camera_ready/logs/%x-%j.out
#
# Shared prep step 2 of 3 (after the LOCO lists, before pinning): frozen SAM embeddings for
# every pooled slice, written once and checked against on-the-fly encoding.
#   bash scripts/camera_ready/prep.sh loco      # login node
#   sbatch scripts/camera_ready/prep_embeddings.slurm
#   bash scripts/camera_ready/prep.sh pin       # login node, after this job succeeds
source "${CR_ROOT:-/data/satvikpraveen/camera_ready}/BAP-MOS/scripts/camera_ready/_env.sh"
"${PY}" -m bapmos.preprocess.prostate.precompute_sam_embeddings \
  --out-dir data/prostate/pooled/sam_vit_b_embeddings --device cuda
