# Bladder baselines

U-Net, nnU-Net, SAM box and MedSAM on PFUS1. These are additional experiments, not reported in
the paper; the paper's PFUS1 table has BAP-MOS (MedSAM) and the FPN result quoted from García et
al. (2024).

All four use the prostate trainers with `--data_root data/bladder/pfus1`, CE + Dice loss
(nnU-Net: its own Dice + CE), checkpointing on fold-balanced validation MSD of the `Bladder`
organ, and three seeds (42, 43, 44).

| Folder | Trainer |
|--------|---------|
| `unet/` | `bapmos.external_baselines.unet.train_multiclass` |
| `medsam/` | `bapmos.external_baselines.medsam_init.train_decoder_boxes` |
| `sam_box/` | `bapmos.multiorgan.train_sam_multiorgan_decoder_box` |
| `nnunet/` | nnU-Net v2 with `nnUNetTrainerBapMosProtocol` |

Commands: [`docs/RUNNING.md`](../../../docs/RUNNING.md#baselines-additional-experiments).
