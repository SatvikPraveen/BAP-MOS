#!/usr/bin/env bash
# Results collation helpers (inference_output → results/by_seed + combined/).
# Scheduler-agnostic — source from BAPMOS/scripts/populate_results_*.sh only.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file; do not execute directly." >&2
  exit 1
fi

bapmos_results_env() {
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [[ ! -d "${root}/src/bapmos" ]]; then
    echo "ERROR: expected BAPMOS package at ${root}" >&2
    return 1
  fi
  cd "${root}"
  export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}"
}

populate_results_one() {
  local corpus="$1"
  local method="$2"
  local run_name="$3"
  local run_dir="$4"
  bapmos_results_env
  run_dir="${run_dir#./}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "ERROR: inference run dir not found: ${run_dir}" >&2
    return 1
  fi
  echo "[populate] ingest-run corpus=${corpus} method=${method} run_name=${run_name}"
  python -m bapmos.results.collate_seeds ingest-run \
    --corpus "${corpus}" \
    --method "${method}" \
    --run-name "${run_name}" \
    --run-dir "${run_dir}"
}

populate_results_build() {
  local corpus="$1"
  bapmos_results_env
  echo "[populate] build --corpus ${corpus}"
  python -m bapmos.results.collate_seeds build --corpus "${corpus}"
}

populate_prostate_legacy_ladder_ingest() {
  local io="inference_output/prostate/pooled"
  shopt -s nullglob

  local seed
  for seed in pooled_seed42 pooled_seed43_rep2 pooled_seed44_rep3; do
    populate_results_one prostate box "${seed}" "${io}/box/box_${seed}"
    populate_results_one prostate point "${seed}" "${io}/point/point_${seed}"
  done

  local run_name
  for run_name in "${io}/box_point"/*; do
    [[ -d "${run_name}" ]] || continue
    run_name="$(basename "${run_name}")"
    populate_results_one prostate box_point "${run_name}" "${io}/box_point/${run_name}"
  done

  for run_name in "${io}/boxpoint_box_point"/*; do
    [[ -d "${run_name}" ]] || continue
    run_name="$(basename "${run_name}")"
    populate_results_one prostate boxpoint_box_point "${run_name}" "${io}/boxpoint_box_point/${run_name}"
  done

  local policy
  for policy in ucb1_global ucb1_per_organ epsilon_greedy_per_organ epsilon_decay_per_organ; do
    for run_name in "${io}/${policy}"/*; do
      [[ -d "${run_name}" ]] || continue
      run_name="$(basename "${run_name}")"
      populate_results_one prostate "${policy}" "${run_name}" "${io}/${policy}/${run_name}"
    done
  done
}

populate_prostate_main_ingest() {
  local io="inference_output/prostate/pooled"
  shopt -s nullglob

  # BAP-MOS method (seed-42 primary export at method root; replicates may use subfolders)
  if [[ -d "${io}/bapmos/simulation" || -d "${io}/bapmos/case1" ]]; then
    populate_results_one prostate bapmos pooled_seed42 "${io}/bapmos"
  fi
  for run_name in "${io}/bapmos"/pooled_seed*; do
    [[ -d "${run_name}" ]] || continue
    run_name="$(basename "${run_name}")"
    populate_results_one prostate bapmos "${run_name}" "${io}/bapmos/${run_name}"
  done
  if [[ -d "${io}/bapmos_medsam/simulation" || -d "${io}/bapmos_medsam/case1" ]]; then
    populate_results_one prostate bapmos_medsam pooled_seed42 "${io}/bapmos_medsam"
  fi
  for run_name in "${io}/bapmos_medsam"/pooled_seed*; do
    [[ -d "${run_name}" ]] || continue
    run_name="$(basename "${run_name}")"
    populate_results_one prostate bapmos_medsam "${run_name}" "${io}/bapmos_medsam/${run_name}"
  done
  for method in bapmos_random bapmos_heuristic bapmos_greedy; do
    for run_name in "${io}/${method}"/pooled_seed*; do
      [[ -d "${run_name}" ]] || continue
      run_name="$(basename "${run_name}")"
      populate_results_one prostate "${method}" "${run_name}" "${io}/${method}/${run_name}"
    done
  done

  # Seed-42 may live at method root (sites directly under unet/medsam/nnunet/).
  for method in unet medsam nnunet; do
    if [[ -d "${io}/${method}/simulation" || -d "${io}/${method}/case1" ]]; then
      populate_results_one prostate "${method}" pooled_seed42 "${io}/${method}"
    fi
  done

  # Per-seed subdirs (preferred): unet_pooled_seed42 / pooled_seed42 / …
  for method in unet medsam nnunet; do
    for run_dir in "${io}/${method}"/*; do
      [[ -d "${run_dir}" ]] || continue
      run_name="$(basename "${run_dir}")"
      case "${run_name}" in
        simulation|case1|case2|metrics|predictions|visualizations) continue ;;
      esac
      collate_name="${run_name}"
      # Training folder prefixes → canonical collate run names (docs/SEEDS.md).
      case "${run_name}" in
        *_seed42) collate_name="pooled_seed42" ;;
        *_seed43_rep2) collate_name="pooled_seed43_rep2" ;;
        *_seed44_rep3) collate_name="pooled_seed44_rep3" ;;
        pooled_seed42|pooled_seed43_rep2|pooled_seed44_rep3) collate_name="${run_name}" ;;
      esac
      populate_results_one prostate "${method}" "${collate_name}" "${io}/${method}/${run_name}"
    done
  done
}
