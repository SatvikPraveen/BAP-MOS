"""
External-style segmentation baselines (same ``data/`` corpora as SAM / BAP-MOS).

Loss policy (``bapmos.losses.loss_policy``):

- **BAP-MOS method** (``bapmos.method``, SAM or MedSAM): Kervadec-style.
- **These baselines**: regional CE+Dice only (nnU-Net: stock Dice+CE).
  Never Kervadec.

- ``unet`` — multiclass U-Net via ``segmentation_models_pytorch``
- ``medsam_init`` — **MedSAM-init** SAM ViT-B decoder-only box fine-tuning (not frozen MedSAM;
  MedSAM encoder re-applied on resume/inference)
- ``nnunet2d`` — **protocol-constrained** nnU-Net v2 (not untouched stock): export to
  ``nnUNet_raw``, trainer ``nnUNetTrainerBapMosProtocol``, stratified test re-export

Entrypoints (from the BAPMOS checkout, with ``PYTHONPATH=src``)::

    python -m bapmos.external_baselines.medsam_init.verify_weights
    python -m bapmos.external_baselines.medsam_init.train_decoder_boxes --data_root data/bladder/pfus1
    python -m bapmos.external_baselines.unet.train_multiclass --data_root data/prostate/pooled
    python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset --data_root data/prostate/pooled
"""
