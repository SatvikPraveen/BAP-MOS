#!/usr/bin/env bash
# Shared prep for the camera-ready runs, done once on the login node before any job:
#
#   bash scripts/camera_ready/prep.sh loco          1. LOCO split lists
#   sbatch scripts/camera_ready/prep_embeddings.slurm   2. frozen SAM embeddings (GPU)
#   bash scripts/camera_ready/prep.sh pin           3. pin every data file in DATA_SHA256.txt
#   bash scripts/camera_ready/prep.sh check         verify (what every job does at start)
#
# Jobs only verify these (cr_preflight); they never rebuild them. "pin" refuses to overwrite
# an existing manifest unless FORCE=1, so a running wave never sees its inputs change.
source "$(dirname "$0")/_env.sh"

case "${1:-}" in
  loco)
    "${PY}" -m bapmos.preprocess.prostate.create_loco_splits
    ;;
  pin)
    if [[ -f "${CR_ROOT}/DATA_SHA256.txt" && "${FORCE:-0}" != "1" ]]; then
      echo "prep: ${CR_ROOT}/DATA_SHA256.txt exists; use FORCE=1 to re-pin" >&2
      exit 1
    fi
    "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
    emb=data/prostate/pooled/sam_vit_b_embeddings/manifest.json
    [[ -f "${emb}" ]] || { echo "prep: ${emb} missing; run prep_embeddings.slurm first" >&2; exit 1; }
    find data/prostate -type f | LC_ALL=C sort | xargs sha256sum > "${CR_ROOT}/DATA_SHA256.txt"
    echo "prep: pinned $(wc -l < "${CR_ROOT}/DATA_SHA256.txt") files → ${CR_ROOT}/DATA_SHA256.txt"
    ;;
  check)
    cr_preflight
    "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
    ;;
  *)
    echo "usage: $0 {loco|pin|check}" >&2
    exit 2
    ;;
esac
