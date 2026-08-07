# Baseline: UNet multiclass (PFUS1 bladder).
# Entry: python -m bapmos.external_baselines.unet.train_multiclass
# Loss: regional CE+Dice only (not Kervadec).
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md and docs/EXPERIMENT_LADDER.md
run_root: runs/bladder/baselines/unet
data_root: data/bladder/pfus1
checkpoint_objective:
  metric: ptv_kfold_msd
  objective_organ: Bladder
  kfold_n: 5
  kfold_seed: 42
