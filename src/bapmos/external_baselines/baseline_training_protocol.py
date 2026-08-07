"""Shared external-baseline training protocol (BAP-MOS production parity).

Loss policy (see ``bapmos.losses.loss_policy``):

- **BAP-MOS method** (``bapmos.method``, SAM or MedSAM init): Kervadec-style.
- **External / multiorgan / legacy baselines**: regional CE+Dice only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from bapmos.losses.loss_policy import (
    NON_METHOD_LOSS_MODE as EXTERNAL_BASELINE_LOSS_MODE,
    force_non_method_ce_dice_loss as force_external_baseline_ce_dice_loss,
)

# PFUS1 / clinical production defaults (match configs/bladder/common.yaml).
PFUS1_BATCH_SIZE = 128
PFUS1_LR = 1e-4
PFUS1_MAX_EPOCHS = 100
PFUS1_PATIENCE = 20
# Shared square for nearest-neighbor pred/GT resize before boundary MSD/HD95.
# Used by all stratified test exports (U-Net, SAM box/points, MedSAM, BAP-MOS,
# nnU-Net) so surface distances match decoder / baseline training resolution.
# Pass 0 / None via ``resolve_boundary_eval_size`` to score at native resolution.
PFUS1_BASELINE_EVAL_SIZE = 256
# Alias: same constant applies beyond PFUS1 (e.g. pooled prostate test exports).
BASELINE_BOUNDARY_EVAL_SIZE = PFUS1_BASELINE_EVAL_SIZE


def resolve_boundary_eval_size(eval_size: int | None) -> int | None:
    """Return square side for boundary metrics, or ``None`` for native resolution.

    ``None`` / ``0`` / negative → native. Positive → that square side.
    Callers that want the protocol default should pass ``PFUS1_BASELINE_EVAL_SIZE``
    (the default on ``export_stratified_test_bundle``).
    """
    if eval_size is None:
        return None
    size = int(eval_size)
    if size <= 0:
        return None
    return size

# Non-PFUS1 (prostate pooled / case / simulation) defaults.
CLINICAL_MAX_EPOCHS = 300
CLINICAL_PATIENCE = 40
CLINICAL_BATCH_SIZE = 1


def make_constant_lr_scheduler(optimizer: Optimizer) -> LRScheduler:
    """Flat LR schedule (no decay), matching BAP-MOS Adam @ fixed lr."""
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def production_wandb_tags(
    *,
    dataset_label: str,
    baseline: str,
    experiment: Optional[str] = None,
    seed: Optional[int] = None,
    extra: Optional[List[str]] = None,
) -> List[str]:
    if experiment is None:
        experiment = f"production_seed{int(seed) if seed is not None else 42}"
    tags = [
        f"dataset={dataset_label}",
        f"experiment={experiment}",
        "production",
        "external",
        f"baseline={baseline}",
    ]
    if extra:
        for tag in extra:
            if tag not in tags:
                tags.append(tag)
    return tags


def setup_wandb_metric_axes() -> None:
    import wandb

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")


def log_epoch_wandb(log_dict: Dict[str, Any]) -> None:
    """Log one epoch dict using ``epoch`` as the step axis (BAP-MOS style)."""
    import wandb

    if wandb.run is None:
        return
    wandb.log(dict(log_dict))
