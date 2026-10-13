"""
The grid on which every method is scored.

Dice, MSD and HD95 are computed on the reference annotation's **native** grid, with
the dataset's physical pixel spacing, for every method (BAP-MOS, SAM / MedSAM,
U-Net, nnU-Net). A prediction made on a coarser grid -- the 256x256 SAM decoder
output, the U-Net input size -- is brought to the reference grid by
nearest-neighbour label upsampling. The reference itself is never resampled.

This is what keeps millimetre distances meaningful and methods comparable: shrinking
prediction and reference to a common square and then applying the native spacing
reports distances at the wrong scale (by the resize factor, differently per axis
when the frame is not square), and would rank methods on a grid none of them was
annotated on.
"""

from __future__ import annotations

import cv2
import numpy as np

EVALUATION_GRID = "native_reference_grid"


def to_reference_grid(pred_labels: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    """
    Return ``pred_labels`` as a ``uint8`` label map on ``reference_labels``' grid.

    Nearest-neighbour interpolation keeps class ids intact. A prediction already on
    the reference grid is returned unchanged (as ``uint8``).
    """
    pred = np.asarray(pred_labels)
    ref = np.asarray(reference_labels)
    if pred.ndim != 2 or ref.ndim != 2:
        raise ValueError(
            f"expected 2-D label maps, got prediction {pred.shape} and reference {ref.shape}"
        )
    if pred.size and int(pred.max()) > 255:
        raise ValueError(f"class id {int(pred.max())} does not fit in uint8")
    pred = pred.astype(np.uint8, copy=False)
    if pred.shape == ref.shape:
        return pred
    height, width = ref.shape
    return cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
