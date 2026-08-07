"""
PFUS1 PyTorch dataset: PNG frames + rasterized ``*_combined_mask.png``.

Typical layout:
  * images under ``data/bladder/pfus1_raw/Pxxx/frame_yyy.png``
  * masks under ``data/bladder/pfus1/masks/combined_masks/{Pxxx}_{frame_yyy}_combined_mask.png``
  * split lists under ``data/bladder/pfus1/splits_*/{train,val,test}.txt``

This module does **not** hook into ``MultiOrganDataset`` or prostate-case configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bapmos.paths import project_root
from bapmos.preprocess.bladder.constants import (
    ALL_CLASS_IDS,
    JSON_LABEL_TO_CLASS_ID,
    LabelMode,
    SUBSET_JSON_LABEL_TO_LOCAL_ID,
)


def _resolve(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def parse_sample_line(line: str) -> Tuple[str, str]:
    """
    ``line`` is ``Pxxx/frame_yyy`` (no extension), one per row.
    Returns (patient_dir, frame_stem) e.g. ``("P001", "frame_000")``.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        raise ValueError(f"Invalid non-sample line: {line!r}")
    if "/" not in s:
        raise ValueError(f"Expected 'Pxxx/frame_yyy', got: {line!r}")
    patient, stem = s.split("/", 1)
    return patient, stem


def remap_mask_to_subset(mask: np.ndarray) -> np.ndarray:
    """Map full 8-class combined mask to contiguous 1..5 subset; background = 0."""
    out = np.zeros_like(mask, dtype=np.uint8)
    for json_name, local_id in SUBSET_JSON_LABEL_TO_LOCAL_ID.items():
        src = JSON_LABEL_TO_CLASS_ID[json_name]
        out[mask == src] = local_id
    return out


class PFUS1Dataset(Dataset):
    """
    Required arguments (no implicit path defaults):

        image_root: Directory containing ``Pxxx/frame_yyy.png``.
        mask_root: Directory containing flat ``{patient}_{frame_stem}_combined_mask.png``.
        split_file: Path to ``train.txt`` / ``val.txt`` / ``test.txt`` listing ``Pxxx/frame_yyy``.
        label_mode: ``all`` (8 foreground classes) or ``subset_five`` (5 classes).
        return_rgb: If True, convert grayscale ultrasound to 3-channel uint8 (SAM-friendly).

    Tensor contract: ``image`` is unnormalized ``uint8`` (C×H×W); ``mask`` is ``int64`` (H×W).
    Downstream code is expected to normalize / cast as needed.
    """

    def __init__(
        self,
        image_root: Path | str,
        mask_root: Path | str,
        split_file: Path | str,
        label_mode: LabelMode | str = LabelMode.ALL,
        return_rgb: bool = True,
    ):
        self.image_root = _resolve(Path(image_root))
        self.mask_root = _resolve(Path(mask_root))
        self.split_file = _resolve(Path(split_file))
        if isinstance(label_mode, str):
            self.label_mode = LabelMode(label_mode)
        else:
            self.label_mode = label_mode
        self.return_rgb = return_rgb

        self._lines: List[str] = []
        with open(self.split_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self._lines.append(line)

        if not self._lines:
            raise ValueError(f"No samples in {self.split_file}")

    def __len__(self) -> int:
        return len(self._lines)

    def num_classes(self) -> int:
        """Foreground classes (excluding background)."""
        if self.label_mode is LabelMode.ALL:
            return len(ALL_CLASS_IDS)
        return len(SUBSET_JSON_LABEL_TO_LOCAL_ID)

    def _paths_for(self, sample_key: str) -> Tuple[Path, Path]:
        patient, stem = parse_sample_line(sample_key)
        img_path = self.image_root / patient / f"{stem}.png"
        mask_path = self.mask_root / f"{patient}_{stem}_combined_mask.png"
        return img_path, mask_path

    def __getitem__(self, idx: int) -> Dict[str, object]:
        key = self._lines[idx]
        img_path, mask_path = self._paths_for(key)

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Missing image: {img_path}")
        if self.return_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        else:
            image_t = torch.from_numpy(image).unsqueeze(0).contiguous()

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Missing mask: {mask_path}")
        if image_t.shape[1] != mask.shape[0] or image_t.shape[2] != mask.shape[1]:
            raise ValueError(
                f"Shape mismatch for {key}: image {tuple(image_t.shape)} vs mask {mask.shape[:2]}"
            )

        if self.label_mode is LabelMode.SUBSET_FIVE:
            mask = remap_mask_to_subset(mask)

        sample: Dict[str, object] = {
            "image": image_t,
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "sample_key": key,
            "label_mode": self.label_mode.value,
        }
        return sample
