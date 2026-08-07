# Baseline: nnU-Net 2d (PFUS1 bladder) — protocol-constrained, not untouched stock nnU-Net.
# Entry: python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset
# Train: nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerBapMosProtocol
# Loss: nnU-Net's own Dice+CE (not Kervadec).
# Set NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder for PFUS1.
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md
run_root: runs/bladder/baselines/nnunet
data_root: data/bladder/pfus1
