# Box-only and point-only prompts (PFUS1)

Additional experiments, not reported in the paper. They follow the earlier protocol described in
[`experiments/prostate/legacy/README.md`](../../prostate/legacy/README.md): CE + Dice loss and a
fixed 20 px negative-point ring.

| Folder | Trainer | Run folder |
|--------|---------|------------|
| `box/` | `bapmos.multiorgan.train_sam_multiorgan_decoder_box` | `runs/bladder/legacy/box/` |
| `point/` | `bapmos.multiorgan.train_sam_multiorgan_decoder_points` | `runs/bladder/legacy/point/` |

`box/` is also the bladder SAM box baseline; its command is in
[`../baselines/sam_box/README.md`](../baselines/sam_box/README.md). Point only:

```bash
python -m bapmos.multiorgan.train_sam_multiorgan_decoder_points \
  --data_root data/bladder/pfus1 \
  --run_root runs/bladder/legacy/point --run_name point_pfus1_seed42 --seed 42 \
  --max_epochs 100 --patience 20 --batch_size 128 \
  --checkpoint-objective-metric ptv_kfold_msd --checkpoint-objective-organ Bladder

python -m bapmos.multiorgan.run_test_inference \
  --checkpoint runs/bladder/legacy/point/point_pfus1_seed42/best_checkpoint.pth \
  --output_dir inference_output/bladder/pfus1/point/point_pfus1_seed42
```

The trainers take their settings from the command line; the `config.yaml` files record the
intended values. Bladder has no policy experiments.
