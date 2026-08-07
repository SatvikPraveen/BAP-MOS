"""PFUS1 compatibility shims — prefer ``bapmos.preprocess.bladder``.

Legacy module names map 1:1 onto the bladder package (identity re-exports)::

    bapmos.pfus1.constants                 → preprocess.bladder.constants
    bapmos.pfus1.convert_json_polygons_to_masks → …convert_json_polygons_to_masks
    bapmos.pfus1.create_pfus1_splits       → …create_splits
    bapmos.pfus1.dataset_pfus1             → …dataset
    bapmos.pfus1.analyze_pfus1_dataset     → …analyze_dataset
    bapmos.pfus1.visualize_pfus1_samples   → …visualize_samples
    bapmos.pfus1.verify_label_registry     → …verify_label_registry
    bapmos.pfus1.precompute_sam_embeddings → …precompute_sam_embeddings

Package orchestration CLI lives only under ``python -m bapmos.preprocess.bladder``
(not mirrored here).
"""

from bapmos.preprocess.bladder.constants import (
    LabelMode,
    PFUS1_ALL_LABELS,
    PFUS1_SUBSET_FIVE_LABELS,
)
from bapmos.preprocess.bladder.dataset import (
    PFUS1Dataset,
    parse_sample_line,
    remap_mask_to_subset,
)

__all__ = [
    "LabelMode",
    "PFUS1_ALL_LABELS",
    "PFUS1_SUBSET_FIVE_LABELS",
    "PFUS1Dataset",
    "parse_sample_line",
    "remap_mask_to_subset",
]
