# U-Net baseline (PFUS1)

Additional experiment, not reported in the paper. Multiclass U-Net
(`segmentation_models_pytorch`), CE + Dice loss.

```bash
python -m bapmos.external_baselines.unet.train_multiclass \
  --data_root data/bladder/pfus1 \
  --run_root runs/bladder/baselines --run_name unet_pfus1_seed42 --seed 42 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder

python -m bapmos.external_baselines.unet.run_test_inference \
  --checkpoint runs/bladder/baselines/unet/unet_pfus1_seed42/best_checkpoint.pth \
  --output_dir inference_output/bladder/pfus1/unet/unet_pfus1_seed42
```

The trainer writes to `<run_root>/unet/<run_name>/` and picks the PFUS1 schedule (100 epochs,
patience 20, batch 128) from the data root. For seeds 43 and 44, change `--seed` and the run
name (`unet_pfus1_seed43_rep2`, `unet_pfus1_seed44_rep3`) and add `--force-inference-output` to
the export. See [`docs/SEEDS.md`](../../../../docs/SEEDS.md).
