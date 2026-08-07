"""
nnU-Net v2 dataset export, BAP-MOS protocol trainer, and stratified test re-export.

This is **protocol-constrained nnU-Net**, not untouched stock nnU-Net:
flat LR / batch / patience, patient-level splits, validation-MSD checkpointing.
Loss remains nnU-Net's stock Dice+CE (**not** Kervadec).

- ``export_nnunet_dataset`` — write ``nnUNet_raw/DatasetXXX_*`` from a BAPMOS ``data/`` bundle
- ``nnUNetTrainerBapMosProtocol`` — in ``nnunet_trainer_bap_mos.py``
- ``run_test_inference`` / ``evaluate_nnunet_predictions`` — re-export predictions into
  ``inference_output/`` with the same metrics + ``difference_v1`` layout as other baselines

Training still uses the official ``nnunetv2`` CLI in a separate environment
(see ``requirements_nnunet.txt``)::

    python -m bapmos.external_baselines.nnunet2d.export_nnunet_dataset \\
        --data_root data/prostate/pooled --dataset_id 501
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    nnUNetv2_train 501 2d 0 -tr nnUNetTrainerBapMosProtocol
"""
