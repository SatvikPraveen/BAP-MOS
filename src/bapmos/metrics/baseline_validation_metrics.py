"""
Shared validation metrics for baseline SAM trainers (multi-organ / single-organ).

Checkpoint selection and early stopping use **organ-balanced validation MSD**
(mean of per-organ MSD means), matching the optimization trainer philosophy.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.metrics.evaluation_grid import to_reference_grid


def organ_balanced_validation_summary(evaluator: MetricsEvaluator) -> Dict[str, Optional[float]]:
    """
    Organ-balanced means: for each evaluator organ, take ``aggregate_metrics(organ)``
    MSD/HD95 (slice-aggregated within that organ), then average those organ-level means
    with equal weight. Dice uses the same organ-balanced average of per-organ dice means.
    """
    msd_per_organ: list[float] = []
    hd_per_organ: list[float] = []
    dice_per_organ: list[float] = []

    for organ in evaluator.organs:
        agg = evaluator.aggregate_metrics(organ_name=organ)
        if agg is None:
            continue
        d = agg.get("dice_mean")
        if d is not None:
            dice_per_organ.append(float(d))
        m = agg.get("msd_mm_mean")
        if m is not None:
            msd_per_organ.append(float(m))
        h = agg.get("hd95_mm_mean")
        if h is not None:
            hd_per_organ.append(float(h))

    out: Dict[str, Optional[float]] = {
        "val_msd": float(np.mean(msd_per_organ)) if msd_per_organ else None,
        "val_hd95": float(np.mean(hd_per_organ)) if hd_per_organ else None,
        "val_dice": float(np.mean(dice_per_organ)) if dice_per_organ else None,
    }
    return out


def multiclass_pred_uint8_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Argmax over class dimension → (H, W) uint8 numpy."""
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def gt_uint8_from_tensor(gt_multi: torch.Tensor) -> np.ndarray:
    """(1,1,H,W) float tensor → (H,W) uint8."""
    return gt_multi.squeeze().detach().cpu().numpy().astype(np.uint8)


def append_multiclass_validation_slice(
    evaluator: MetricsEvaluator,
    *,
    pred_classes: np.ndarray,
    gt_classes: np.ndarray,
    image_id: str,
    class_mapping: Dict[int, str],
    slice_idx: int = 0,
) -> None:
    """Score one slice; ``gt_classes`` is the reference on its native grid.

    ``pred_classes`` may be on a coarser model grid; it is nearest-neighbour
    upsampled to the reference grid (:mod:`bapmos.metrics.evaluation_grid`).
    """
    gt_u8 = np.asarray(gt_classes).astype(np.uint8)
    evaluator.evaluate_multiclass_slice(
        to_reference_grid(pred_classes, gt_u8),
        gt_u8,
        slice_idx=slice_idx,
        image_id=image_id,
        class_mapping=class_mapping,
    )


def append_single_organ_validation_slice(
    evaluator: MetricsEvaluator,
    *,
    pred_logits: torch.Tensor,
    gt_binary: np.ndarray,
    organ_label: str,
    image_id: str,
    slice_idx: int = 0,
) -> None:
    """Binary slice: argmax on two-class logits, compare to the native GT mask.

    ``gt_binary`` is the reference on its native grid; the prediction is
    upsampled to it (:mod:`bapmos.metrics.evaluation_grid`).
    """
    probs = torch.softmax(pred_logits, dim=1)
    pred_c = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    gt_bin = (np.asarray(gt_binary) > 0).astype(np.uint8)
    pred_bin = to_reference_grid((pred_c > 0).astype(np.uint8), gt_bin)
    evaluator.evaluate_slice(pred_bin, gt_bin, organ_label, slice_idx=slice_idx, image_id=image_id)
