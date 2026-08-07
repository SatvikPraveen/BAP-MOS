# Baseline: UNet multiclass (main paper comparison).
# Entry: python -m bapmos.external_baselines.unet.train_multiclass
# Loss: regional CE+Dice only (not Kervadec — that is BAP-MOS method-only).
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md and docs/EXPERIMENT_LADDER.md
run_root: runs/prostate/baselines/unet
data_root: data/prostate/pooled
checkpoint_objective:
  metric: ptv_kfold_msd
  kfold_n: 5
  kfold_seed: 42
