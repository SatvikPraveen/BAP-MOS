# Runtime layout

## Directories created at run time (not committed)

```text
optuna_studies/
  prostate_bapmos_outer_loop_{tpe,random,greedy,heuristic,medsam}/
  bladder_bapmos_{sam,medsam}_outer_loop/

runs/
  prostate/
    bapmos/
      outer_loop/{tpe,random,greedy,heuristic,medsam}/       # search trials
      inner_loop/{tpe,random,greedy,heuristic,medsam}/<run_name>/
    baselines/
      unet/<run_name>/
      medsam_init/<run_name>/
    legacy/
      box/<run_name>/                                         # also the SAM baseline
      point/<run_name>/
      box_point/<run_name>/
      boxpoint_box_point/<run_name>/
      {ucb1_global,ucb1_per_organ,epsilon_greedy_per_organ,epsilon_decay_per_organ}/<run_name>/
  bladder/
    bapmos/{outer_loop,inner_loop}/{sam,medsam}/
    baselines/{unet,medsam_init}/<run_name>/
    legacy/box/<run_name>/

inference_output/
  prostate/pooled/<method>/<run_name>/{simulation,case1,case2}/
  bladder/pfus1/<method>/<run_name>/

results/
  prostate/pooled/{by_seed/<method>,combined}/
  bladder/pfus1/{by_seed/<method>,combined}/
```

The BAP-MOS trainer takes its run folder from the suite's `run_root`. The baseline and legacy
trainers write where `--run_root` and `--run_name` point; the paths above assume the commands
in [`RUNNING.md`](RUNNING.md). nnU-Net writes to your `nnUNet_results` directory.

Each run folder holds `best_checkpoint.pth`, `last_checkpoint.pth` and `config.json`.
Test-export contents: [`RESULTS.md`](RESULTS.md#test-export-layout). Run names:
[`SEEDS.md`](SEEDS.md).
