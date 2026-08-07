# Baseline: nnU-Net 2D (main paper comparison).
# Entry:
#   1) python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset ...
#   2) nnUNetv2_plan_and_preprocess -d <DATASET_ID>
#   3) nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerBapMosProtocol
# Protocol-constrained nnU-Net (stock Dice+CE; not Kervadec — that is BAP-MOS method-only).
# Flat LR / patient splits / checkpointing via nnUNetTrainerBapMosProtocol — not untouched stock nnU-Net.
# Export / train default Dataset505_BapMosPooledTrainVal — see docs/RUNNING.md and docs/EXPERIMENT_LADDER.md.
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md and docs/EXPERIMENT_LADDER.md
run_root: runs/prostate/baselines/nnunet
data_root: data/prostate/pooled
