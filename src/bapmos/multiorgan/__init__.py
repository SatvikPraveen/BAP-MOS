"""
Multi-organ SAM dataset and baseline decoder trainers (box / point prompts).

Canonical training entrypoints::

    python -m bapmos.multiorgan.train_sam_multiorgan_decoder_box
    python -m bapmos.multiorgan.train_sam_multiorgan_decoder_points

Dataset: ``MultiOrganDataset`` (simulation, clinical case1/2, pooled, PFUS1).
"""

from bapmos.multiorgan.dataset_multi_organ import (
    MultiOrganDataset,
    multi_organ_collate_fn,
    sample_per_organ_points_with_negatives,
)

__all__ = [
    "MultiOrganDataset",
    "multi_organ_collate_fn",
    "sample_per_organ_points_with_negatives",
]
