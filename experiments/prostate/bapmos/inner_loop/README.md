# Inner loop — production (prostate)

Full training with **selected** outer-loop hyperparameters (kervadec-style loss + composite reward).
Schedule: **max_epochs 300 / patience 40**. Three seeds: 42 / 43 / 44 (`docs/SEEDS.md`).

## Required CLI (compose)

```bash
python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42
```

This merges `configs/prostate/common.yaml`, this suite’s `version.yaml`, and
`selected/tpe/pooled.yaml` (default). Checkpoints go under
`runs/prostate/bapmos/inner_loop/tpe/pooled_seed*`. Other search exports:

```bash
# bare names work too (normalized to selected/<method>)
BAPMOS_SELECTED_SUBDIR=random python -m bapmos.method.bap_mos_trainer \
  --version bapmos --dataset pooled --experiment pooled_seed42
# → runs/prostate/bapmos/inner_loop/random/pooled_seed42/
```

Compose **refuses** `selection_meta.generation: 0` placeholders unless you set
`BAPMOS_ALLOW_PLACEHOLDER_SELECTED=1` (debug / dry-run only).

MedSAM backbone (own outer TPE → this suite’s selected/):

```bash
python -m bapmos.hpo.study_runner run \
  --hpo-suite bapmos_bo_medsam_pooled --dataset pooled --n-trials 1
python -m bapmos.method.bap_mos_trainer \
  --version bapmos_medsam_pooled --dataset pooled --experiment pooled_seed42
```

## Do not use for production

- `--config …/inner_loop/version.yaml` — loads as-is; **skips** `selected/` merge. Advanced/debug only.

Start inner loop **only after** outer-loop export overwrites `selected/` placeholders.
