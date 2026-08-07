"""
Shared epoch / test monitoring helpers for baseline trainers (UNet, SAM, legacy BAP-MOS).

Organ-balanced summaries match
``bapmos.metrics.baseline_validation_metrics.organ_balanced_validation_summary``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from bapmos.metrics.baseline_validation_metrics import organ_balanced_validation_summary
from bapmos.metrics.checkpoint_selection import (
    CheckpointScores,
    checkpoint_scores_from_evaluator,
    is_better_checkpoint,
    parse_checkpoint_objective_config,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator

BASELINE_METRICS_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "train_dice",
    "train_msd",
    "train_hd95",
    "train_dice_organ_balanced",
    "val_loss",
    "val_dice",
    "val_msd",
    "val_hd95",
    "val_dice_organ_balanced",
    "val_ptv_hd95",
    "val_ptv_dice",
    "best_val_msd",
    "lr",
]


def validation_checkpoint_scores(
    evaluator: MetricsEvaluator,
    config: dict,
    evaluator_organ_labels: list,
) -> Optional[CheckpointScores]:
    if not evaluator.per_slice_metrics:
        return None
    objective = parse_checkpoint_objective_config(config)
    return checkpoint_scores_from_evaluator(
        evaluator, objective, evaluator_organ_labels
    )


def summarize_evaluator_metrics(
    evaluator: MetricsEvaluator,
) -> Optional[Dict[str, Optional[float]]]:
    """Organ-balanced + slice-pooled summaries from filled ``MetricsEvaluator``."""
    if not evaluator.per_slice_metrics:
        return None
    organ_bal = organ_balanced_validation_summary(evaluator)
    overall = evaluator.aggregate_metrics(organ_name=None) or {}
    return {
        "msd_mm": overall.get("msd_mm_mean"),
        "hd95_mm": overall.get("hd95_mm_mean"),
        "msd_mm_organ_balanced": organ_bal.get("val_msd"),
        "hd95_mm_organ_balanced": organ_bal.get("val_hd95"),
        "dice_organ_balanced": organ_bal.get("val_dice"),
    }


def per_organ_wandb_dict(
    evaluator: MetricsEvaluator,
    organ_labels: List[str],
    prefix: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for organ in organ_labels:
        agg = evaluator.aggregate_metrics(organ_name=organ)
        if agg is None:
            continue
        base = f"{prefix}/organ/{organ}"
        for key, suffix in (
            ("msd_mm_mean", "msd_mm_mean"),
            ("hd95_mm_mean", "hd95_mm_mean"),
            ("dice_mean", "dice_mean"),
        ):
            val = agg.get(key)
            if val is not None:
                out[f"{base}/{suffix}"] = float(val)
    return out


def csv_metric_cell(
    metrics: Optional[Dict[str, Optional[float]]], key: str
) -> Any:
    if not metrics:
        return ""
    v = metrics.get(key)
    return v if v is not None else ""


def build_epoch_wandb_log(
    *,
    epoch: int,
    train_loss: float,
    train_dice: float,
    val_loss: float,
    val_dice: float,
    lr: float,
    time_epoch_s: float,
    train_metrics: Optional[Dict[str, Optional[float]]] = None,
    val_metrics: Optional[Dict[str, Optional[float]]] = None,
    val_per_organ_wandb: Optional[Dict[str, float]] = None,
    train_per_organ_wandb: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """WandB dict aligned with legacy ``bapmos.legacy.optimization.trainer`` epoch logging."""
    log: Dict[str, Any] = {
        "epoch": epoch,
        "train/loss": train_loss,
        "train/dice": train_dice,
        "val/loss": val_loss,
        "val/dice": val_dice,
        "lr": lr,
        "time/epoch_s": time_epoch_s,
    }
    if train_metrics:
        tr_msd = train_metrics.get("msd_mm_organ_balanced") or train_metrics.get("msd_mm")
        tr_hd = train_metrics.get("hd95_mm_organ_balanced") or train_metrics.get("hd95_mm")
        tr_dice = train_metrics.get("dice_organ_balanced")
        if tr_msd is not None:
            log["train/msd_mm"] = float(tr_msd)
        if tr_hd is not None:
            log["train/hd95_mm"] = float(tr_hd)
        if tr_dice is not None:
            log["train/dice_organ_balanced"] = float(tr_dice)
    if val_metrics:
        val_msd_ckpt = val_metrics.get("msd_mm")
        ob_msd = val_metrics.get("msd_mm_organ_balanced")
        ob_hd = val_metrics.get("hd95_mm_organ_balanced")
        ob_dice = val_metrics.get("dice_organ_balanced")
        if val_msd_ckpt is not None:
            log["val/msd_mm"] = float(val_msd_ckpt)
        if ob_msd is not None:
            log["val/msd_mm_organ_balanced"] = float(ob_msd)
        if ob_hd is not None:
            log["val/hd95_mm"] = float(ob_hd)
        if val_metrics.get("hd95_mm") is not None:
            log["val/hd95_mm_slice_weighted"] = float(val_metrics["hd95_mm"])
        if ob_dice is not None:
            log["val/dice_organ_balanced"] = float(ob_dice)
    if val_per_organ_wandb:
        log.update(val_per_organ_wandb)
    if train_per_organ_wandb:
        log.update(train_per_organ_wandb)
    if extra:
        log.update(extra)
    return log


def init_external_baseline_wandb(
    *,
    project: str,
    run_name: str,
    config: dict,
    run_dir: Path,
    resumed: bool = False,
    wandb_entity: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """W&B init aligned with BAP-MOS production runs (epoch x-axis for train/* and val/*)."""
    import os

    import wandb

    from bapmos.external_baselines.baseline_training_protocol import (
        production_wandb_tags,
        setup_wandb_metric_axes,
    )
    from bapmos.paths import dataset_bundle_tag

    ds_label = dataset_bundle_tag(str(config.get("data_root", "")))
    baseline = str(config.get("baseline", "external")).replace(" ", "_")
    init_tags = production_wandb_tags(
        dataset_label=ds_label,
        baseline=baseline,
        seed=config.get("seed"),
        experiment=config.get("experiment_name") or config.get("experiment"),
        extra=list(tags or []),
    )

    entity = wandb_entity or os.environ.get("WANDB_ENTITY")
    init_kw = {
        "project": project,
        "name": run_name,
        "config": config,
        "dir": str(run_dir),
        "tags": init_tags,
        "group": group or f"{baseline}_{ds_label}",
    }
    if entity:
        init_kw["entity"] = entity
    wandb.init(**init_kw)
    setup_wandb_metric_axes()
    if resumed:
        wandb.summary["resumed"] = True


def build_test_wandb_log(
    result: Dict[str, Any],
    evaluator: Optional[MetricsEvaluator] = None,
    organ_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    log: Dict[str, float] = {}
    mapping = {
        "test_loss": "test/loss",
        "test_dice": "test/dice",
        "test_msd": "test/msd_mm",
        "test_msd_mm": "test/msd_mm",
        "test_hd95": "test/hd95_mm",
        "test_hd95_mm": "test/hd95_mm",
        "test_dice_organ_balanced": "test/dice_organ_balanced",
    }
    for src, dst in mapping.items():
        v = result.get(src)
        if v is not None:
            log[dst] = float(v)
    if evaluator is not None and organ_labels:
        log.update(per_organ_wandb_dict(evaluator, organ_labels, "test"))
    return log
