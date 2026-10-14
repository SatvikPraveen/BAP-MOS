# nnU-Net baseline (prostate)

Paper row: nnU-Net (method id `nnunet`). nnU-Net v2 2D with `nnUNetTrainerBapMosProtocol`:
nnU-Net's Dice + CE loss, flat learning rate, the repository's patient-level train/val split,
and validation-MSD checkpointing. It runs in a separate environment (`requirements_nnunet.txt`)
with `src/` on `PYTHONPATH`, and uses dataset id 505.

```bash
python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root data/prostate/pooled --output_parent "${nnUNet_raw}" \
  --dataset_id 505 --dataset_name BapMosPooledTrainVal
nnUNetv2_plan_and_preprocess -d 505 --verify_dataset_integrity
BAPMOS_NNUNET_DATA_ROOT=data/prostate/pooled NNUNET_MAX_EPOCHS=300 NNUNET_PATIENCE=40 \
  nnUNetv2_train 505 2d 0 -tr nnUNetTrainerBapMosProtocol
```

Environment variables, trainer registration, per-site prediction and test export:
[`docs/RUNNING.md`](../../../../docs/RUNNING.md#nnu-net-nnunet). There is no seed option; a
replicate is a separate training run.
