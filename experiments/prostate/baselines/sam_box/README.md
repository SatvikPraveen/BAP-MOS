# SAM baseline (prostate)

Paper row: SAM (method id `box`). The SAM ViT-B mask decoder fine-tuned with box prompts, CE +
Dice loss. This is the same recipe as the box-only rung in `../../legacy/box/`, so runs go
under `runs/prostate/legacy/box/` and there is no separate `baselines/sam_box/` run folder.

```bash
python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box \
  --data_root data/prostate/pooled \
  --run_root runs/prostate/legacy/box --run_name box_pooled_seed42 --seed 42 \
  --box_jitter_px 2 \
  --checkpoint-objective-metric ptv_kfold_msd

python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/prostate/legacy/box/box_pooled_seed42/best_checkpoint.pth \
  --output_dir inference_output/prostate/pooled/box/box_pooled_seed42
```

Pass `--data_root` explicitly; its default is the single-site `case1` corpus. The trainer writes
to `<run_root>/<run_name>/` (default `--run_root` is `runs/<bundle>/Baseline`). For seeds 43 and
44, change `--seed` and the run name (`box_pooled_seed43_rep2`, `box_pooled_seed44_rep3`) and add
`--force-inference-output` to the export. Weights: `models/sam_base/sam_vit_b_01ec64.pth`.
