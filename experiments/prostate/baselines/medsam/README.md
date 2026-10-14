# MedSAM baseline (prostate)

Paper row: MedSAM (method id `medsam`). SAM ViT-B initialised with MedSAM weights; only the mask
decoder is fine-tuned with box prompts. CE + Dice loss.

```bash
python -m bapmos.external_baselines.medsam_init.verify_weights

python -m bapmos.external_baselines.medsam_init.train_decoder_boxes \
  --data_root data/prostate/pooled \
  --medsam_checkpoint models/medsam/medsam_vit_b.pth \
  --run_root runs/prostate/baselines --run_name medsam_pooled_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.external_baselines.medsam_init.run_test_inference \
  --checkpoint runs/prostate/baselines/medsam_init/medsam_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/medsam/medsam_pooled_seed42
```

The trainer writes to `<run_root>/medsam_init/<run_name>/` (default `--run_root` is
`runs/ExternalBaselines`) and records the weight overlap in the run's `config.json`. For seeds 43
and 44, change `--seed` and the run name and add `--force-inference-output` to the export.
Weights: [`docs/WEIGHTS.md`](../../../../docs/WEIGHTS.md).
