#!/usr/bin/env bash
# Shared prep for the camera-ready runs, done once on the login node before any job:
#
#   bash scripts/camera_ready/prep.sh loco          1. LOCO split lists
#   sbatch scripts/camera_ready/prep_embeddings.slurm   2. frozen SAM embeddings (GPU)
#   bash scripts/camera_ready/prep.sh pin prostate  3. pin data/prostate in DATA_SHA256_prostate.txt
#   bash scripts/camera_ready/prep.sh pin pfus1        and data/bladder in DATA_SHA256_pfus1.txt
#   DATASET=pfus1 bash scripts/camera_ready/prep.sh check   verify, as every job does at start
#
# Jobs only verify these (cr_preflight); they never rebuild them. "pin" refuses to overwrite
# an existing manifest unless FORCE=1, so a running wave never sees its inputs change.
source "$(dirname "$0")/_env.sh"

case "${1:-}" in
  loco)
    "${PY}" -m bapmos.preprocess.prostate.create_loco_splits
    ;;
  pin)
    # prep.sh pin prostate | pfus1
    case "${2:-}" in
      prostate)
        "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
        for emb in data/prostate/pooled/{sam,medsam}_vit_b_embeddings/manifest.json; do
          [[ -f "${emb}" ]] || { echo "prep: ${emb} missing; run prep_embeddings.slurm first" >&2; exit 1; }
        done
        dir=data/prostate ;;
      pfus1) dir=data/bladder ;;
      *) echo "usage: $0 pin {prostate|pfus1}" >&2; exit 2 ;;
    esac
    out="${CR_ROOT}/DATA_SHA256_${2}.txt"
    if [[ -f "${out}" && "${FORCE:-0}" != "1" ]]; then
      echo "prep: ${out} exists; use FORCE=1 to re-pin" >&2
      exit 1
    fi
    find "${dir}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "${out}"
    echo "prep: pinned $(wc -l < "${out}") files → ${out}"
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
