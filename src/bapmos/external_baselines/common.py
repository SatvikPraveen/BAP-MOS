"""Shared helpers for external baseline trainers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resize_masks_nearest(
    pred: np.ndarray,
    gt: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize multiclass pred/GT to ``size×size`` for baseline metric parity (nearest labels)."""
    pred_u8 = cv2.resize(
        pred.astype(np.uint8),
        (int(size), int(size)),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_u8 = cv2.resize(
        gt.astype(np.uint8),
        (int(size), int(size)),
        interpolation=cv2.INTER_NEAREST,
    )
    return pred_u8, gt_u8


def resize_for_baseline(
    image_rgb: np.ndarray,
    mask_multi: np.ndarray,
    size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    image_rgb: (H, W, 3) uint8
    mask_multi: (H, W) uint8 class ids
    Returns:
        x: (1, 3, size, size) float32 in [0, 1] (apply ImageNet mean/std on device in trainer)
        y: (1, size, size) int64 class indices
    """
    h0, w0 = image_rgb.shape[:2]
    x = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0  # 3,H,W
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    m = torch.from_numpy(mask_multi.astype(np.float32)).view(1, 1, h0, w0)
    m = F.interpolate(m, size=(size, size), mode="nearest")
    y = m.squeeze(0).squeeze(0).long()
    return x, y.unsqueeze(0)


def imagenet_normalize_bchw(x: torch.Tensor) -> torch.Tensor:
    """x: (N, 3, H, W) on any device, float."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def multiclass_ce_dice_loss(
    logits: torch.Tensor,
    gt_long: torch.Tensor,
    num_classes: int,
    *,
    epoch: int = 0,
    max_epochs: int = 300,
    loss_mode: str = "ce_dice",
    kervadec_alpha_start: float = 1.0,
    kervadec_alpha_end: float = 0.5,
    kervadec_alpha_schedule: str = "linear",
    dist_maps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Regional CE + mean foreground Dice for external baselines.

    Kervadec is **not** used on the external-baseline path (UNet / MedSAM-box /
    nnU-Net). Extra keyword args are accepted for call-site compatibility and
    ignored. Passing ``loss_mode='kervadec'`` raises.
    """
    del epoch, max_epochs, kervadec_alpha_start, kervadec_alpha_end
    del kervadec_alpha_schedule, dist_maps
    mode = str(loss_mode or "ce_dice").lower()
    if mode != "ce_dice":
        raise ValueError(
            "external baselines use regional CE+Dice only "
            f"(got loss_mode={loss_mode!r}; Kervadec is BAP-MOS method-only)"
        )
    return ce_dice_regional_loss(logits, gt_long, num_classes)


def ce_dice_regional_loss(
    logits: torch.Tensor,
    gt_long: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Same spirit as SAM multi-organ trainer: CE + mean(1 - Dice_c) over foreground."""
    from bapmos.losses.kervadec_style import ce_dice_regional_loss as _regional

    return _regional(logits, gt_long, num_classes)


def mean_fg_dice_from_logits(logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int) -> float:
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        gt_c = (gt_long == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union > 0:
            dices.append(float((2.0 * inter + 1e-7) / (union + 1e-7)))
    return float(np.mean(dices)) if dices else 0.0


def external_run_dir(
    subpackage: str,
    run_name: str,
    *,
    external_baselines_parent: Optional[Union[str, Path]] = None,
) -> Path:
    """
    ``<parent>/<subpackage>/<run_name>/`` under project root by default.

    Default parent: ``runs/ExternalBaselines``. For PFUS1 cluster layout parity with
    internal baselines, pass e.g. ``runs/pfus1/ExternalBaselines``.
    """
    from bapmos.paths import project_root

    pr = project_root()
    if external_baselines_parent is None:
        base = pr / "runs" / "ExternalBaselines"
    else:
        p = Path(external_baselines_parent)
        base = p if p.is_absolute() else (pr / p)
    return base.resolve() / subpackage / run_name
