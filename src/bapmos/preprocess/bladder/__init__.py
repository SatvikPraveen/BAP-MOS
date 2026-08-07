"""Bladder (PFUS1) preprocessing: JSON→masks, patient splits, verify, dataset helpers."""

from __future__ import annotations

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
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Orchestrate convert + splits + verify (``python -m bapmos.preprocess.bladder``)."""
    from bapmos.preprocess.bladder._cli import main as _main

    return _main(argv)
