# Runtime artifact layout

Empty directories under `BAPMOS/` matching training / HPO / test-export paths.
Jobs fill them; empty folders are not completed runs.

## Three artifact trees

```text
runs/prostate/…/<run_folder>/best_checkpoint.pth     # training
inference_output/prostate/pooled/<method>/<export>/   # stratified test export
results/prostate/pooled/by_seed/<method>/<run>.csv    # per-seed metrics
results/prostate/pooled/combined/                     # paper mean±std (site=pooled)
```

Inference exports are written under `BAPMOS/inference_output/`.

Each prostate export folder contains `simulation/`, `case1/`, `case2/`, and `sites.json`.

## Directory map

```text
optuna_studies/
  prostate_bapmos_outer_loop_{tpe,random,greedy,heuristic,medsam}/
  bladder_bapmos_{sam,medsam}_outer_loop/

runs/
  prostate/
    baselines/{unet/unet_pooled_seed*, medsam_init/medsam_pooled_seed*, nnunet}/
    # SAM-box decoder is the same recipe as legacy/box — do not duplicate under baselines/sam_box
    bapmos/
      outer_loop/{tpe,random,greedy,heuristic,medsam}/   # trial_* after HPO
      inner_loop/{tpe,random,greedy,heuristic,medsam}/pooled_seed*/
    legacy/{box,point,box_point,boxpoint_box_point,ucb1_*,epsilon_*}/
      # box/ == SAM ViT-B + box prompts (CE+Dice); use this instead of baselines/sam_box
  bladder/
    baselines/…
    bapmos/{outer_loop,inner_loop}/{sam,medsam}/…

inference_output/
  prostate/pooled/<method>/{simulation,case1,case2}/
    predictions/multiclass | visualizations/difference | metrics
  bladder/pfus1/<method>/…

results/
  prostate/pooled/{by_seed/<method>,combined}/
  bladder/pfus1/{by_seed/<method>,combined}/
```

Canonical seed folder names: `pooled_seed42` / `pooled_seed43_rep2` / `pooled_seed44_rep3`
(and bladder `pfus1_seed*`). See `docs/SEEDS.md`.
