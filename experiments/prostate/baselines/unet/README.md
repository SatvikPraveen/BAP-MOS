# U-Net baseline (prostate)

Paper row: U-Net (method id `unet`). Multiclass U-Net (`segmentation_models_pytorch`), CE + Dice
loss.

```bash
python -m bapmos.external_baselines.unet.train_multiclass \
  --data_root data/prostate/pooled \
  --run_root runs/prostate/baselines --run_name unet_pooled_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.external_baselines.unet.run_test_inference \
  --checkpoint runs/prostate/baselines/unet/unet_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/unet/unet_pooled_seed42
```

The trainer writes to `<run_root>/unet/<run_name>/` (default `--run_root` is
`runs/ExternalBaselines`) and picks the prostate schedule (300 epochs, patience 40, batch 1) from
the data root. For seeds 43 and 44, change `--seed` and the run name (`unet_pooled_seed43_rep2`,
`unet_pooled_seed44_rep3`) and add `--force-inference-output` to the export.
See [`docs/SEEDS.md`](../../../../docs/SEEDS.md).
