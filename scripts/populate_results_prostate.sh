#!/usr/bin/env bash
# Ingest prostate pooled inference exports → results/by_seed + combined/ paper tables.
#
# Scheduler-agnostic (no Slurm). Run from anywhere:
#   bash scripts/populate_results_prostate.sh
#   LADDER_ONLY=1 bash scripts/populate_results_prostate.sh   # legacy ladder only
#   MAIN_ONLY=1 bash scripts/populate_results_prostate.sh       # bapmos + baselines only
#   BUILD_ONLY=1 bash scripts/populate_results_prostate.sh      # rebuild combined/ only
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_SCRIPT_DIR}/_populate_results_lib.sh"

if [[ "${BUILD_ONLY:-0}" != "1" ]]; then
  if [[ "${MAIN_ONLY:-0}" != "1" ]]; then
    populate_prostate_legacy_ladder_ingest
  fi
  if [[ "${LADDER_ONLY:-0}" != "1" ]]; then
    populate_prostate_main_ingest
  fi
else
  echo "[populate] BUILD_ONLY=1 — skipping ingest"
fi

populate_results_build prostate
echo "Done → results/prostate/pooled/combined/"
