# Baseline: MedSAM-init + box decoder fine-tuning (main paper comparison).
# Not frozen MedSAM inference — MedSAM weights init the encoder; only the decoder is trained.
# Preflight: python -m bapmos.external_baselines.medsam_init.verify_weights
# Train:     python -m bapmos.external_baselines.medsam_init.train_decoder_boxes
# Test export: python -m bapmos.external_baselines.medsam_init.run_test_inference
# Loss: regional CE+Dice only (not Kervadec — that is BAP-MOS method-only).
# Weights: models/medsam/medsam_vit_b.pth — see docs/WEIGHTS.md (overlap logged in config.json)
# Three seeds: 42 / 43 / 44 — see docs/SEEDS.md and docs/EXPERIMENT_LADDER.md
run_root: runs/prostate/baselines/medsam_init
data_root: data/prostate/pooled
