#!/usr/bin/env bash
# One-time shared prep for the camera-ready runs (run once on the login node, before any job).
#
#   1. writes the LOCO split lists (bapmos.preprocess.prostate.create_loco_splits);
#   2. pins every file under data/prostate in ${CR_ROOT}/DATA_SHA256.txt.
#
# Jobs only verify these (cr_preflight); they never rebuild them. Refuses to overwrite an
# existing manifest unless FORCE=1, so a running wave never sees its inputs change.
source "$(dirname "$0")/_env.sh"

if [[ -f "${CR_ROOT}/DATA_SHA256.txt" && "${FORCE:-0}" != "1" ]]; then
  echo "prep: ${CR_ROOT}/DATA_SHA256.txt exists; verifying instead of rewriting"
  cr_preflight
  "${PY}" -m bapmos.preprocess.prostate.create_loco_splits --check
  exit 0
fi

"${PY}" -m bapmos.preprocess.prostate.create_loco_splits
find data/prostate -type f | LC_ALL=C sort | xargs sha256sum > "${CR_ROOT}/DATA_SHA256.txt"
echo "prep: pinned $(wc -l < "${CR_ROOT}/DATA_SHA256.txt") files → ${CR_ROOT}/DATA_SHA256.txt"
