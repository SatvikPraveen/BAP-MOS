#!/usr/bin/env bash
# Thin helpers for BAP-MOS; run from anywhere (the script changes to the repository root).
# Sets PYTHONPATH=src — no editable / pyproject install required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

cmd="${1:-}"
shift || true
case "$cmd" in
  preprocess-prostate)
    python -m bapmos.preprocess.prostate "$@"
    ;;
  preprocess-bladder)
    python -m bapmos.preprocess.bladder "$@"
    ;;
  preprocess-delineation)
    python -m bapmos.preprocess.delineation "$@"
    ;;
  test)
    python -m pytest tests -q "$@"
    ;;
  populate-prostate|populate-results-prostate)
    bash "${ROOT}/scripts/populate_results_prostate.sh" "$@"
    ;;
  populate-bladder|populate-results-bladder)
    bash "${ROOT}/scripts/populate_results_bladder.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {preprocess-prostate|preprocess-bladder|preprocess-delineation|test|populate-prostate|populate-bladder} [args...]"
    exit 1
    ;;
esac
