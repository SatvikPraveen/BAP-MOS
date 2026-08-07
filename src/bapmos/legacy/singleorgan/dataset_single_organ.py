"""
Binary-mask dataset: one foreground organ per sample, same files as MultiOrganDataset.

Reads combined multiclass masks and keeps only the requested organ class as label 1;
all other pixels (including other organs) become 0.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset
from bapmos.training_taxonomy import get_baseline_taxonomy_profile


def binary_mask_from_multiclass(mc: np.ndarray, organ: str, organ_to_class: dict) -> np.ndarray:
    """Return uint8 mask in {0,1} where 1 indicates pixels of `organ` only."""
    if organ not in organ_to_class:
        raise ValueError(f"organ must be one of {list(organ_to_class)}, got {organ!r}")
    cid = organ_to_class[organ]
    return ((mc == cid).astype(np.uint8))


class SingleOrganDataset(Dataset):
    """
    Same directory layout as MultiOrganDataset (``images/`` or ``case*_dicom_png/``,
    ``masks/`` or ``masks/combined_masks/``, ``splits_stratified/``).
    Each sample returns a binary mask: 1 = target organ, 0 = background (and other organs).
    """

    def __init__(
        self,
        data_root,
        organ: str,
        split: str = "train",
        splits_subdir: str = "splits_stratified",
        image_embedding_dir: Optional[str] = None,
    ):
        profile = get_baseline_taxonomy_profile(data_root)
        organ_to_class = profile.organ_to_class
        if organ not in organ_to_class:
            raise ValueError(f"organ must be one of {list(organ_to_class)}, got {organ!r}")
        self.organ = organ
        self._organ_to_class = organ_to_class
        self.split = split
        self._base = MultiOrganDataset(
            data_root,
            split=split,
            splits_subdir=splits_subdir,
            image_embedding_dir=image_embedding_dir,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        binary = binary_mask_from_multiclass(sample["mask"], self.organ, self._organ_to_class)
        out = {
            "image": sample["image"],
            "mask": binary,
            "filename": sample["filename"],
            "organ": self.organ,
            "has_organ": bool(binary.any()),
        }
        if "embedding_path" in sample:
            out["embedding_path"] = sample["embedding_path"]
        if "sam_cached_pack" in sample:
            out["sam_cached_pack"] = sample["sam_cached_pack"]
        return out
