# Baseline: MedSAM-init decoder box (PFUS1 bladder).
# Not frozen MedSAM inference — MedSAM encoder init + decoder-only box fine-tuning.
# Preflight: python -m bapmos.external_baselines.medsam_init.verify_weights
# Train:     python -m bapmos.external_baselines.medsam_init.train_decoder_boxes
# Test export: python -m bapmos.external_baselines.medsam_init.run_test_inference
# Loss: regional CE+Dice only (not Kervadec).
# Defaults: configs/bladder/common.yaml (patience 20 / 100 epochs).
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md
run_root: runs/bladder/baselines/medsam_init
data_root: data/bladder/pfus1
