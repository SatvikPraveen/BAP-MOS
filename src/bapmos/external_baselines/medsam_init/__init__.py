"""
MedSAM-init SAM ViT-B decoder-only box fine-tuning (external baseline).

**Not** frozen MedSAM inference: build SAM ViT-B, overlay MedSAM weights on the encoder,
fine-tune the mask decoder only with box prompts and CE+Dice loss (not Kervadec).

Public entrypoints:

- ``train_decoder_boxes`` — training / resume / test-only export
- ``run_test_inference`` — stratified ``inference_output/`` re-export (primary seed 42)
- ``verify_weights`` — quick MedSAM key-overlap check before training
- ``weight_loader`` — ``apply_medsam_encoder_init``, ``MedsamLoadReport``

MedSAM encoder weights are re-applied on resume and standalone inference so evaluation
uses the same backbone as training (Meta SAM arch + MedSAM encoder + fine-tuned decoder).

Examples::

    # Preflight overlap (expects ~0–10 missing keys for Zenodo medsam_vit_b.pth)
    python -m bapmos.external_baselines.medsam_init.verify_weights

    python -m bapmos.external_baselines.medsam_init.train_decoder_boxes \\
        --data_root data/bladder/pfus1 \\
        --medsam_checkpoint models/medsam/medsam_vit_b.pth

    python -m bapmos.external_baselines.medsam_init.run_test_inference \\
        --checkpoint runs/.../medsam_init/<run>/best_checkpoint.pth \\
        --output_dir inference_output/...
"""

from bapmos.external_baselines.medsam_init.weight_loader import (
    MedsamLoadReport,
    apply_medsam_encoder_init,
    is_medsam_init_config,
    verify_medsam_checkpoint,
)

__all__ = [
    "MedsamLoadReport",
    "apply_medsam_encoder_init",
    "is_medsam_init_config",
    "verify_medsam_checkpoint",
]
