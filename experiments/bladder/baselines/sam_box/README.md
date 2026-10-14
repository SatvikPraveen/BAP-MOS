# SAM box baseline (PFUS1)

Additional experiment, not reported in the paper. SAM ViT-B mask decoder fine-tuned with box
prompts, CE + Dice loss. It is the same recipe as `../../legacy/box/`, so runs go under
`runs/bladder/legacy/box/` and collate as method `box`.

```bash
python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box \
  --data_root data/bladder/pfus1 \
  --run_root runs/bladder/legacy/box --run_name box_pfus1_seed42 --seed 42 \
  --max_epochs 100 --patience 20 --batch_size 128 --box_jitter_px 10 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder

python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/bladder/legacy/box/box_pfus1_seed42/best_checkpoint.pth \
  --output_dir inference_output/bladder/pfus1/box/box_pfus1_seed42
```

Pass `--data_root` explicitly; its default is the prostate `case1` corpus. The trainer writes to
`<run_root>/<run_name>/`. The schedule flags match `configs/bladder/common.yaml`.
