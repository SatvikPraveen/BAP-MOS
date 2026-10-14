# Results tables

Running the pipeline writes per-seed tables to `by_seed/` and mean ± std tables to `combined/`,
under `results/prostate/` and `results/bladder/`. These files are generated locally and are
git-ignored. The repository ships no result tables; the paper's numbers are in the
[main README](../README.md#main-results).

## Example files

[`example/`](example/) shows every table a full run produces, with the same folders, file names,
columns and rows. Only the measured values are placeholders:

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

Prostate: trained on the pooled corpus, exported per site (`simulation`, `case1`, `case2`), and
aggregated as the slice-weighted `site=pooled` row. Distances are in millimetres.

PFUS1: distances are in pixel units, although the columns are still named `msd_mm` and
`hd95_mm`.

## Method IDs (`by_seed/<method>/`)

Pass these strings to `collate_seeds --method`. Source of truth: `src/bapmos/results/layout.py`.

- Prostate: `box`, `point`, `box_point`, `boxpoint_box_point`, `ucb1_global`, `ucb1_per_organ`,
  `epsilon_greedy_per_organ`, `epsilon_decay_per_organ`, `bapmos`, `bapmos_medsam`,
  `bapmos_random`, `bapmos_heuristic`, `bapmos_greedy`, `unet`, `nnunet`, `medsam`
- Bladder: `box`, `point`, `bapmos_sam`, `bapmos_medsam`, `unet`, `nnunet`, `medsam`

`example/` has no bladder `point` folder, because no bladder point-only run is part of the
shipped example layout. If you train one (`experiments/bladder/legacy/point/`), it collates
under `point` like any other method.

Which README row each id produces (for example, the prostate SAM row is `box`, and the PFUS1
BAP-MOS (MedSAM) row is bladder `bapmos_medsam`) is listed in
[`docs/RESULTS.md`](../docs/RESULTS.md#paper-tables-and-method-ids).

## Layout

```text
results/
  prostate/pooled/
    by_seed/<method>/<run_name>.csv
    combined/
      prostate_pooled_mean_std.csv         # all organs, site=pooled
      prostate_pooled_ptv_mean_std.csv     # PTV only (primary organ)
  bladder/pfus1/
    by_seed/<method>/<run_name>.csv
    combined/
      bladder_pfus1_mean_std.csv           # all organs
      bladder_pfus1_bladder_mean_std.csv   # Bladder only (primary organ)
```

Collated metrics are Dice, MSD and HD95.

## Building the tables

From the repository root, after the test exports:

```bash
bash scripts/populate_results_prostate.sh
bash scripts/populate_results_bladder.sh
```

Options and the manual `collate_seeds` commands are in
[`docs/RESULTS.md`](../docs/RESULTS.md#automated-collation).
