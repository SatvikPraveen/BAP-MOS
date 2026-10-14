# nnU-Net baseline (PFUS1)

Additional experiment, not reported in the paper. nnU-Net v2 2D with
`nnUNetTrainerBapMosProtocol`: nnU-Net's Dice + CE loss, flat learning rate, the repository's
patient-level train/val split, and validation-MSD checkpointing. It runs in a separate
environment (`requirements_nnunet.txt`) with `src/` on `PYTHONPATH`, and uses dataset id 503.

```bash
python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \
  --data_root data/bladder/pfus1 --output_parent "${nnUNet_raw}" \
  --dataset_id 503 --dataset_name BapMosPfus1TrainVal
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
BAPMOS_NNUNET_DATA_ROOT=data/bladder/pfus1 NNUNET_CHECKPOINT_OBJECTIVE_ORGAN=Bladder \
  nnUNetv2_train 503 2d 0 -tr nnUNetTrainerBapMosProtocol
```

Environment variables, trainer registration, prediction and test export:
[`docs/RUNNING.md`](../../../../docs/RUNNING.md#nnu-net-nnunet). There is no seed option; a
replicate is a separate training run.
