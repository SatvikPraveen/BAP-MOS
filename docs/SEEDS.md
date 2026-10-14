# Training seeds

Every reported method is trained with three seeds: 42, 43 and 44. Tables report mean ± std over
the three runs.

## Run names

| Replicate | Seed | Prostate | Bladder |
|-----------|------|----------|---------|
| primary | 42 | `pooled_seed42` | `pfus1_seed42` |
| rep2 | 43 | `pooled_seed43_rep2` | `pfus1_seed43_rep2` |
| rep3 | 44 | `pooled_seed44_rep3` | `pfus1_seed44_rep3` |

Baseline and fixed-schedule runs add a prefix (`unet_pooled_seed42`, `box50_point50_seed42`);
collation maps the suffix back to these names ([`RESULTS.md`](RESULTS.md#run-names)).
`bapmos.replicate_runs.all_study_run_names("pooled_seed42")` returns the three names.

## How each trainer takes the seed

| Trainer | Seed control |
|---------|--------------|
| BAP-MOS (`bapmos.method.bap_mos_trainer`) | The `--experiment` entry (`pooled_seed43_rep2` sets seed 43) |
| U-Net, MedSAM, SAM box / point | `--seed` |
| Fixed ratios (`legacy/box_point`, `legacy/boxpoint_box_point`) | `PI_TRAIN_SEED=43` / `44` environment variable |
| Legacy policies | The `--experiment` entry (`ucb1_global_seed43_rep2`) |
| nnU-Net | No seed option; a replicate is a separate `nnUNetv2_train` run |

Commands: [`RUNNING.md`](RUNNING.md).

## Rules

- The outer-loop search uses seed 42 only.
- Only the training seed changes between replicates. The checkpoint k-fold seed
  (`evaluation.kfold_seed`) and the BAP-MOS probe seed (`bandit.probe_seed`) stay at 42.
- Each seed has its own run folder under `runs/`.
- Test metrics come from `run_test_inference`, not from training. The export refuses
  non-seed-42 runs unless you pass `--force-inference-output`, so pass it for seeds 43 and 44.
- Export every seed to its own folder, `inference_output/…/<method>/<run_name>/`, then collate.
- A single seed-42 run is not the reported result.
