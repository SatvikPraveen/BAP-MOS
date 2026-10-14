# Results collation (mean ± std)

After the test exports, collate metrics into tables under `results/`. Per-seed rows go to
`by_seed/`; mean ± std tables go to `combined/`. Prostate and bladder are collated separately.
Every method (fixed schedules, policies, BAP-MOS variants, baselines) is one `method` folder
under `by_seed/`.

The repository ships no result tables. `results/` holds only a README and
[`results/example/`](../results/example/), which shows every table a full run produces with
placeholder values. The paper's numbers are in the [main README](../README.md#main-results).

## Paper tables and method ids

| README row | Method id | Where it comes from |
|------------|-----------|---------------------|
| Prostate: nnU-Net | `nnunet` | nnU-Net v2 with `nnUNetTrainerBapMosProtocol` |
| Prostate: U-Net | `unet` | `bapmos.external_baselines.unet` |
| Prostate: SAM | `box` | SAM decoder fine-tuned with box prompts (`bapmos.multiorgan.train_sam_multiorgan_decoder_box`; runs under `runs/prostate/legacy/box/`) |
| Prostate: MedSAM | `medsam` | MedSAM-initialised box decoder (`bapmos.external_baselines.medsam_init`) |
| Prostate: BAP-MOS (SAM) | `bapmos` | `--version bapmos` |
| Prostate: BAP-MOS (MedSAM) | `bapmos_medsam` | `--version bapmos_medsam_pooled` |
| PFUS1: BAP-MOS (MedSAM) | `bapmos_medsam` (bladder) | `--version bapmos_medsam --dataset pfus1` |
| PFUS1: FPN (Dice 0.710) | — | Quoted from García et al. (2024). Not reproduced by this code. |

The paper's fixed-schedule table (`box_point`, `boxpoint_box_point`), search-strategy
comparison (`bapmos`, `bapmos_random`, `bapmos_greedy`, `bapmos_heuristic`) and adaptive policy
variants (`ucb1_global`, `ucb1_per_organ`, `epsilon_greedy_per_organ`,
`epsilon_decay_per_organ`) are collated the same way. The bladder ids `bapmos_sam`, `unet`,
`nnunet`, `box` and `medsam` are additional experiments, not reported in the paper.

## Metric units

- Prostate TRUS: MSD and HD95 in millimetres, from native physical spacing.
- PFUS1: MSD and HD95 in pixel units.

The CSV columns are named `msd_mm` and `hd95_mm` for both corpora. For PFUS1, read them as
pixel units.

## Evaluation grid

Every method (BAP-MOS, fixed-prompt SAM / MedSAM, U-Net, nnU-Net and the legacy policies) is
scored on the **native grid of the reference annotation**, with the dataset's physical pixel
spacing (`bapmos.metrics.evaluation_grid`). A prediction made on a coarser grid (the 256×256 SAM
decoder output, the U-Net input) is upsampled to the reference grid with nearest-neighbour
interpolation; the reference itself is never resampled. Validation, the bandit probe reward, the
TPE objective and test export all use this one grid, and each export records `evaluation_grid`
and `pixel_spacing_mm` in its `evaluation_meta.json`.

An export that saved `*_pred_ids.png` for every test slice can be re-scored without inference:
`python -m bapmos.results.recompute_native_metrics run --corpus <prostate|bladder>`.

## Prostate multi-site rule

Training uses the pooled corpus. Test exports have one folder per site (`simulation`, `case1`,
`case2`). The pooled number is not an unweighted average of the three site means:

1. Load `per_slice_metrics.csv` from each site.
2. Keep rows with `valid_boundary`.
3. Concatenate across sites and take the mean per organ.

This equals the slice-weighted mean of the site means:

```math
\frac{\sum_s \mathrm{site\_mean}_s \cdot n_s}{\sum_s n_s}
```

Mean ± std across seeds 42, 43 and 44 is then taken over these pooled values.

## Test export layout

`run_test_inference` exports (BAP-MOS and every baseline) write to
`inference_output/<corpus>/<dataset>/<method>/<run_name>/`. Prostate exports have one subfolder
per site; PFUS1 exports write directly into the run folder:

```text
<run_name>/[<site>/]
  predictions/
    multiclass/*_pred_ids.png
    *_pred.png
  visualizations/
    *_viz.png|.pdf              # prediction overlays
    difference/*_diff.png|.pdf  # TP / FP / FN overlays
  metrics/                      # summary_metrics.csv, per_slice_metrics.csv
  evaluation_meta.json
```

Metrics cover the full test split; images and PDFs cover the first 10 slices. The writer is
`export_stratified_test_bundle` in `src/bapmos/inference_output/export_bundle.py`.

## Table layout

```text
results/prostate/pooled/
  by_seed/<method>/<run_name>.csv
  combined/
    prostate_pooled_mean_std.csv       # all organs, site=pooled
    prostate_pooled_ptv_mean_std.csv   # PTV only (primary prostate organ)

results/bladder/pfus1/
  by_seed/<method>/<run_name>.csv
  combined/
    bladder_pfus1_mean_std.csv
    bladder_pfus1_bladder_mean_std.csv # Bladder only (primary organ)
```

Per-seed columns:

`method, run_name, seed, site, organ, dice, msd_mm, hd95_mm, n_slices, source`

Mean ± std columns (`combined/`):

`method, run_name, organ, site, n_seeds, seeds, dice_mean, dice_std, dice_mean_pm_std, …`
(the same for `msd_mm` and `hd95_mm`).

`run_name` separates experiments within one method, such as the three `box_point` ratios or the
four `boxpoint_box_point` mixes. In `combined/`, `run_name` drops the seed suffix. Prostate
aggregates use `site=pooled` rows only. IoU appears in the raw export CSVs but is not collated.

## Run names

| Method | Export folder (`inference_output/prostate/pooled/…`) | `run_name` in `by_seed/` |
|--------|------------------------------------------------------|--------------------------|
| `box` | `box/box_pooled_seed42` | `pooled_seed42` |
| `point` | `point/point_pooled_seed42` | `pooled_seed42` |
| `unet`, `medsam`, `nnunet` | `unet/unet_pooled_seed42` (etc.) | `pooled_seed42` |
| `bapmos`, `bapmos_*` | `bapmos/pooled_seed42` | `pooled_seed42` |
| `box_point` | `box_point/box50_point50_seed42` | same as folder |
| `boxpoint_box_point` | `boxpoint_box_point/ratio_1_1_1_balanced_seed42` | same as folder |
| policies | `<policy>/ucb1_global_seed42` | same as folder |

Each experiment has three seeds: `*_seed42`, `*_seed43_rep2`, `*_seed44_rep3`
([`SEEDS.md`](SEEDS.md)). Bladder run names are `pfus1_seed42`, `pfus1_seed43_rep2`,
`pfus1_seed44_rep3`.

## Automated collation

```bash
bash scripts/populate_results_prostate.sh                  # all prostate exports
LADDER_ONLY=1 bash scripts/populate_results_prostate.sh    # fixed schedules and policies only
MAIN_ONLY=1 bash scripts/populate_results_prostate.sh      # BAP-MOS and baselines only
BUILD_ONLY=1 bash scripts/populate_results_prostate.sh     # rebuild combined/ only

bash scripts/populate_results_bladder.sh                   # all bladder exports
BUILD_ONLY=1 bash scripts/populate_results_bladder.sh
```

Both scripts skip missing runs with a warning. `./scripts/bapmos.sh populate-prostate` and
`./scripts/bapmos.sh populate-bladder` call the same scripts.

## Manual collation

Prostate: `ingest-run` reads the `simulation/`, `case1/` and `case2/` folders of one export and
adds the slice-weighted pooled row.

```bash
python -m bapmos.results.collate_seeds ingest-run \
  --corpus prostate --method box --run-name pooled_seed42 \
  --run-dir inference_output/prostate/pooled/box/box_pooled_seed42

python -m bapmos.results.collate_seeds ingest-run \
  --corpus prostate --method box_point --run-name box50_point50_seed42 \
  --run-dir inference_output/prostate/pooled/box_point/box50_point50_seed42

python -m bapmos.results.collate_seeds build --corpus prostate
```

Bladder: exports have no site folders, so ingest one summary file per run.

```bash
python -m bapmos.results.collate_seeds ingest \
  --corpus bladder --method bapmos_medsam --run-name pfus1_seed42 --site pfus1 \
  --summary inference_output/bladder/pfus1/bapmos_medsam/pfus1_seed42/metrics/summary_metrics.csv

python -m bapmos.results.collate_seeds build --corpus bladder
```

Ingest all three seeds before `build`. Training and export commands are in
[`RUNNING.md`](RUNNING.md).
