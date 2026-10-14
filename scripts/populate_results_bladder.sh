#!/usr/bin/env bash
# Ingest PFUS1 bladder inference exports into results/bladder/pfus1/{by_seed,combined}/.
#
# Run from anywhere:
#   bash scripts/populate_results_bladder.sh
#   BUILD_ONLY=1 bash scripts/populate_results_bladder.sh   # rebuild combined/ only
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_SCRIPT_DIR}/_populate_results_lib.sh"

if [[ "${BUILD_ONLY:-0}" != "1" ]]; then
  populate_bladder_ingest
else
  echo "[populate] BUILD_ONLY=1, skipping ingest"
fi

populate_results_build bladder
echo "Done: results/bladder/pfus1/combined/"
