"""
PFUS1 label definitions (JSON ``label`` strings and multiclass IDs).

Rasterization paints structures in **ascending ``class_id``** order so that
higher IDs overwrite overlaps (reproducible; document in docs/manifests).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, List, Tuple

# JSON strings exactly as in dataset annotations (see ``data/bladder/pfus1*`` READMEs).
# ``class_id`` 1..8; 0 = background in combined masks.
PFUS1_ALL_LABELS: Tuple[Tuple[str, int], ...] = (
    ("Pubis", 1),
    ("Urethra", 2),
    ("Bladder", 3),
    ("Uterus", 4),
    ("Vagina", 5),
    ("Anus", 6),
    ("Rectum", 7),
    ("Levator ani muscle", 8),
)

JSON_LABEL_TO_CLASS_ID: Dict[str, int] = {name: cid for name, cid in PFUS1_ALL_LABELS}
ALL_CLASS_IDS: Tuple[int, ...] = tuple(cid for _, cid in PFUS1_ALL_LABELS)

# Explicit subset (user-requested); use only after comparing to full-label stats.
PFUS1_SUBSET_FIVE_LABELS: Tuple[str, ...] = (
    "Bladder",
    "Rectum",
    "Urethra",
    "Levator ani muscle",
    "Pubis",
)

# Remap subset to contiguous 1..5 for training / metrics (subset mode).
SUBSET_JSON_LABEL_TO_LOCAL_ID: Dict[str, int] = {
    name: i + 1 for i, name in enumerate(PFUS1_SUBSET_FIVE_LABELS)
}


class LabelMode(str, Enum):
    """Which label set the dataset exposes in ``mask`` tensors."""

    ALL = "all"
    SUBSET_FIVE = "subset_five"


def labels_for_mode(mode: LabelMode) -> Tuple[str, ...]:
    if mode is LabelMode.ALL:
        return tuple(name for name, _ in PFUS1_ALL_LABELS)
    if mode is LabelMode.SUBSET_FIVE:
        return PFUS1_SUBSET_FIVE_LABELS
    raise ValueError(f"Unknown mode {mode!r}")


def json_labels_used_in_mode(mode: LabelMode) -> FrozenSet[str]:
    return frozenset(labels_for_mode(mode))
