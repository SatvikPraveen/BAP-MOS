# Results tables (reproduction CSVs)

Per-seed results live under `by_seed/`; mean ± std aggregates live under `combined/`.

This repository does not ship result tables. Running the pipeline writes them under
`results/prostate/` and `results/bladder/`, and those files stay local (they are git-ignored).
The paper's numbers are reported in the [main README](../README.md).

## Example files

[`example/`](example/) shows every table a full run produces, with the same folders, file
names, columns and rows. Only the measured values are placeholders:

| Placeholder | Filled with |
| --- | --- |
| `<dice>`, `<msd_mm>`, `<hd95_mm>` | the metric for that run, site and organ |
| `<n_slices>` | number of evaluated test slices |
| `<dice_mean>`, `<dice_std>`, … | mean and standard deviation over seeds 42, 43 and 44 |
| `<dice_mean>±<dice_std>`, … | the same pair formatted for tables |

For example, a row of `example/prostate/pooled/by_seed/bapmos/pooled_seed42.csv`:

```csv
method,run_name,seed,site,organ,dice,msd_mm,hd95_mm,n_slices,source
bapmos,pooled_seed42,42,simulation,PTV,<dice>,<msd_mm>,<hd95_mm>,<n_slices>,inference_output/prostate/pooled/bapmos/pooled_seed42/simulation/metrics/summary_metrics.csv
```

**Prostate:** train pooled; infer on `simulation` / `case1` / `case2`; aggregate = slice-weighted `site=pooled`.
Distance metrics (MSD / HD95) are in **millimeters** (native physical spacing).

**Bladder PFUS1:** paper distance metrics use **pixel-equivalent units** (CSV columns may still be named `msd_mm` / `hd95_mm` for schema consistency).
See [`docs/RESULTS.md`](../docs/RESULTS.md).

## Exact method IDS (`by_seed/<method>/`)

Use these strings verbatim (`collate_seeds --method`). Source of truth: `src/bapmos/results/layout.py`.

**Prostate pooled:** `box`, `point`, `box_point`, `boxpoint_box_point`, `ucb1_global`, `ucb1_per_organ`, `epsilon_greedy_per_organ`, `epsilon_decay_per_organ`, `bapmos`, `bapmos_medsam`, `bapmos_random`, `bapmos_heuristic`, `bapmos_greedy`, `unet`, `nnunet`, `medsam`

**Bladder PFUS1:** `box`, `point`, `bapmos_sam`, `bapmos_medsam`, `unet`, `nnunet`, `medsam` (no legacy ratio/policy methods on bladder)

## Result layout

```text
results/
  prostate/pooled/
    by_seed/<method>/<run_name>.csv
    combined/
      prostate_pooled_mean_std.csv      # all organs, site=pooled
      prostate_pooled_ptv_mean_std.csv  # PTV only (primary organ)
  bladder/pfus1/
    by_seed/<method>/<run_name>.csv
    combined/
      bladder_pfus1_mean_std.csv           # all organs, site=pfus1
      bladder_pfus1_bladder_mean_std.csv   # Bladder only (primary organ)
```

Collated metrics: **Dice, MSD, HD95** only (units: mm for prostate; pixel-equivalent for PFUS1 paper tables).

Scratch dumps and audit/status CSVs are **not** shipped (see `.gitignore`).

## Populate after inference (recommended)

From the repository root:

```bash
bash scripts/populate_results_prostate.sh
# Legacy ladder only: LADDER_ONLY=1 bash scripts/populate_results_prostate.sh
# Rebuild combined/ only: BUILD_ONLY=1 bash scripts/populate_results_prostate.sh
```

See [`docs/RUNNING.md`](../docs/RUNNING.md) for per-method `python -m` ingest examples.

## Manual CLI (from the repository root)

```bash
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# box: run_name is pooled_seed42 (not box_pooled_seed42)
python -m bapmos.results.collate_seeds ingest-run \
  --corpus prostate --method box --run-name pooled_seed42 \
  --run-dir inference_output/prostate/pooled/box/box_pooled_seed42

# box_point: run_name matches inference folder
python -m bapmos.results.collate_seeds ingest-run \
  --corpus prostate --method box_point --run-name box50_point50_seed42 \
  --run-dir inference_output/prostate/pooled/box_point/box50_point50_seed42

# Repeat ingest-run for seeds 42, 43, and 44 before building
# the three-seed mean ± std tables.
python -m bapmos.results.collate_seeds build --corpus prostate
```

See [`docs/RESULTS.md`](../docs/RESULTS.md) and the [main README](../README.md) for additional details.
