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
    evaluator.evaluate_multiclass_slice(
        pred_classes,
        gt_classes,
        slice_idx=slice_idx,
        image_id=image_id,
        class_mapping=class_mapping,
    )


def append_single_organ_validation_slice(
    evaluator: MetricsEvaluator,
    *,
    pred_logits: torch.Tensor,
    gt_binary: torch.Tensor,
    organ_label: str,
    image_id: str,
    slice_idx: int = 0,
) -> None:
    """Binary slice: argmax on two-class logits, compare to GT {0,1}."""
    probs = torch.softmax(pred_logits, dim=1)
    pred_c = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    gt_c = gt_binary.squeeze().detach().cpu().numpy().astype(np.uint8)
    pred_bin = (pred_c > 0).astype(np.uint8)
    gt_bin = (gt_c > 0).astype(np.uint8)
    evaluator.evaluate_slice(pred_bin, gt_bin, organ_label, slice_idx=slice_idx, image_id=image_id)
