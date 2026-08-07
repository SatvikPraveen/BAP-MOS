"""Shared validation checkpoint objectives (PTV k-fold MSD, organ-balanced MSD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from bapmos.metrics.baseline_validation_metrics import organ_balanced_validation_summary
from bapmos.legacy.optimization.metrics import MetricsEvaluator


def resolve_ptv_evaluator_organ(
    evaluator_organ_labels: Sequence[str],
    *,
    override: Optional[str] = None,
) -> Optional[str]:
    """Return PTV organ label (``PTV`` clinical, ``PTV1`` simulation)."""
    if override:
        label = str(override)
        if label not in evaluator_organ_labels:
            raise ValueError(
                f"objective_organ={label!r} not in evaluator organs {list(evaluator_organ_labels)}"
            )
        return label
    for label in evaluator_organ_labels:
        if "PTV" in label.upper():
            return label
    return None


def ptv_kfold_msd_mm(
    evaluator: MetricsEvaluator,
    organ_name: str,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> Optional[float]:
    """
    K-fold mean PTV MSD on validation slices (equal weight per fold).

    Each fold mean is computed over slices with valid boundary MSD; folds with no
    valid slices are skipped. ``n_folds`` is capped at the number of valid slices.
    """
    metrics_list = [
        m
        for m in evaluator.per_slice_metrics
        if m["organ"] == organ_name
        and m.get("valid_boundary")
        and m.get("msd_mm") is not None
    ]
    if not metrics_list:
        return None

    slice_ids = sorted({str(m["image_id"]) for m in metrics_list})
    n_slices = len(slice_ids)
    if n_slices == 0:
        return None

    k = min(max(int(n_folds), 1), n_slices)
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(slice_ids))

    folds = [[] for _ in range(k)]
    for idx, sid in enumerate(perm):
        folds[idx % k].append(str(sid))

    id_to_msd = {str(m["image_id"]): float(m["msd_mm"]) for m in metrics_list}
    fold_means: list[float] = []
    for fold_ids in folds:
        msds = [id_to_msd[sid] for sid in fold_ids if sid in id_to_msd]
        if msds:
            fold_means.append(float(np.mean(msds)))

    return float(np.mean(fold_means)) if fold_means else None


def ptv_full_val_metrics(
    evaluator: MetricsEvaluator,
    organ_name: str,
) -> Dict[str, Optional[float]]:
    """Full-validation PTV secondary metrics (mean over PTV slices with valid boundary)."""
    metrics_list = [
        m
        for m in evaluator.per_slice_metrics
        if m["organ"] == organ_name and m.get("valid_boundary")
    ]
    if not metrics_list:
        return {"ptv_hd95": None, "ptv_dice": None}

    hd95_values = [m["hd95_mm"] for m in metrics_list if m.get("hd95_mm") is not None]
    dice_values = [m["dice"] for m in metrics_list if m.get("dice") is not None]
    return {
        "ptv_hd95": float(np.mean(hd95_values)) if hd95_values else None,
        "ptv_dice": float(np.mean(dice_values)) if dice_values else None,
    }


@dataclass(frozen=True)
class CheckpointObjectiveConfig:
    metric: str = "organ_balanced_msd"
    kfold_n: int = 5
    kfold_seed: int = 42
    objective_organ: Optional[str] = None
    min_delta: float = 1e-6


@dataclass(frozen=True)
class CheckpointScores:
    primary_msd: Optional[float]
    metric: str
    ptv_organ: Optional[str] = None
    ptv_hd95: Optional[float] = None
    ptv_dice: Optional[float] = None
    organ_balanced_msd: Optional[float] = None
    organ_balanced_hd95: Optional[float] = None
    organ_balanced_dice: Optional[float] = None


def parse_checkpoint_objective_config(cfg: dict) -> CheckpointObjectiveConfig:
    """Parse checkpoint metric from ``checkpoint_objective`` or ``evaluation`` blocks.

    Precedence (first hit wins for the metric string):

    1. ``checkpoint_objective.metric``
    2. top-level ``checkpoint_objective_metric`` / ``checkpoint_metric``
    3. ``evaluation.objective_metric`` (BAP-MOS version.yaml style)
    4. default ``organ_balanced_msd``
    """
    block = cfg.get("checkpoint_objective") or {}
    eval_block = cfg.get("evaluation") or {}
    metric = (
        block.get("metric")
        or cfg.get("checkpoint_objective_metric")
        or ("ptv_kfold_msd" if cfg.get("checkpoint_metric") == "ptv_kfold_msd" else None)
        or eval_block.get("objective_metric")
        or "organ_balanced_msd"
    )
    return CheckpointObjectiveConfig(
        metric=str(metric),
        kfold_n=int(
            block.get(
                "kfold_n",
                cfg.get("checkpoint_kfold_n", eval_block.get("kfold_n", 5)),
            )
        ),
        kfold_seed=int(
            block.get(
                "kfold_seed",
                cfg.get("checkpoint_kfold_seed", eval_block.get("kfold_seed", 42)),
            )
        ),
        objective_organ=(
            block.get("objective_organ")
            or cfg.get("checkpoint_objective_organ")
            or eval_block.get("objective_organ")
        ),
        min_delta=float(
            block.get(
                "min_delta",
                cfg.get(
                    "val_msd_min_delta",
                    (cfg.get("common") or {}).get("val_msd_min_delta", 1e-6),
                ),
            )
        ),
    )


def checkpoint_scores_from_evaluator(
    evaluator: MetricsEvaluator,
    objective: CheckpointObjectiveConfig,
    evaluator_organ_labels: Sequence[str],
) -> CheckpointScores:
    organ_bal = organ_balanced_validation_summary(evaluator)
    ptv_organ = resolve_ptv_evaluator_organ(
        evaluator_organ_labels, override=objective.objective_organ
    )
    ptv_sec = (
        ptv_full_val_metrics(evaluator, ptv_organ)
        if ptv_organ is not None
        else {"ptv_hd95": None, "ptv_dice": None}
    )

    primary: Optional[float]
    if objective.metric == "ptv_kfold_msd":
        if not ptv_organ:
            primary = None
        else:
            primary = ptv_kfold_msd_mm(
                evaluator,
                ptv_organ,
                n_folds=objective.kfold_n,
                seed=objective.kfold_seed,
            )
    elif objective.metric == "organ_balanced_msd":
        primary = organ_bal.get("val_msd")
    else:
        raise ValueError(
            f"Unknown checkpoint objective metric {objective.metric!r}; "
            "use 'ptv_kfold_msd' or 'organ_balanced_msd'."
        )

    return CheckpointScores(
        primary_msd=primary,
        metric=objective.metric,
        ptv_organ=ptv_organ,
        ptv_hd95=ptv_sec["ptv_hd95"],
        ptv_dice=ptv_sec["ptv_dice"],
        organ_balanced_msd=organ_bal.get("val_msd"),
        organ_balanced_hd95=organ_bal.get("val_hd95"),
        organ_balanced_dice=organ_bal.get("val_dice"),
    )


def merge_checkpoint_objective_config(target: dict, source: dict) -> None:
    """Copy ``checkpoint_objective`` block from *source* into trainer *target* config."""
    block = source.get("checkpoint_objective")
    if block:
        target["checkpoint_objective"] = dict(block)


def add_checkpoint_objective_cli(parser) -> None:
    parser.add_argument(
        "--checkpoint-objective-metric",
        type=str,
        default=None,
        choices=["ptv_kfold_msd", "organ_balanced_msd"],
        help="Validation checkpoint primary metric (default: organ_balanced_msd).",
    )
    parser.add_argument(
        "--checkpoint-objective-organ",
        type=str,
        default=None,
        help="Evaluator organ label for k-fold MSD (e.g. PTV, Bladder).",
    )
    parser.add_argument("--checkpoint-kfold-n", type=int, default=None)
    parser.add_argument("--checkpoint-kfold-seed", type=int, default=None)


def apply_checkpoint_objective_cli(config: dict, args) -> None:
    metric = getattr(args, "checkpoint_objective_metric", None)
    if metric:
        config.setdefault("checkpoint_objective", {})
        config["checkpoint_objective"]["metric"] = metric
    organ = getattr(args, "checkpoint_objective_organ", None)
    if organ:
        config.setdefault("checkpoint_objective", {})
        config["checkpoint_objective"]["objective_organ"] = str(organ)
    kfold_n = getattr(args, "checkpoint_kfold_n", None)
    if kfold_n is not None:
        config.setdefault("checkpoint_objective", {})
        config["checkpoint_objective"]["kfold_n"] = int(kfold_n)
    kfold_seed = getattr(args, "checkpoint_kfold_seed", None)
    if kfold_seed is not None:
        config.setdefault("checkpoint_objective", {})
        config["checkpoint_objective"]["kfold_seed"] = int(kfold_seed)


def is_better_checkpoint(
    candidate_msd: Optional[float],
    best_msd: float,
    *,
    min_delta: float,
) -> bool:
    if candidate_msd is None:
        return False
    return float(candidate_msd) < float(best_msd) - float(min_delta)
