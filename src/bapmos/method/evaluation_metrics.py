"""
Epoch / block / test metrics for ``bapmos.method`` (shared evaluator with legacy baselines).

Loss stays at the SAM decoder grid (256×256); MSD/HD95 upsample class maps to native GT
resolution via :func:`upsample_pred_classes_to_gt` (nearest-neighbor), matching
``run_test_inference._resize_pred``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch

from bapmos.metrics.baseline_validation_metrics import (
    append_multiclass_validation_slice,
    organ_balanced_validation_summary,
)
from bapmos.metrics.checkpoint_selection import (
    ptv_kfold_msd_mm,
    resolve_ptv_evaluator_organ,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.train.training_taxonomy import BaselineTaxonomyProfile, get_baseline_taxonomy_profile

from bapmos.method.data_adapter import resolve_path
from bapmos.method.reward import OrganProbeMetrics
from bapmos.legacy.optimization.metrics.boundary_metrics import compute_all_metrics

if TYPE_CHECKING:
    from bapmos.method.bap_mos_trainer import BAPMOSTrainer

logger = logging.getLogger(__name__)


def multiclass_dice_from_logits(
    logits: torch.Tensor, gt_multi: torch.Tensor, num_classes: int
) -> float:
    """Mean Dice over foreground classes (matches legacy optimization trainer)."""
    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(probs, dim=1, keepdim=True)
    dices: List[float] = []
    for c in range(1, num_classes):
        pred_c = (pred_classes == c).float()
        gt_c = (gt_multi == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union > 0:
            dice = (2.0 * inter + 1e-7) / (union + 1e-7)
            dices.append(float(dice.item()))
    return float(np.mean(dices)) if dices else 0.0


def resolve_taxonomy(data_root: str) -> BaselineTaxonomyProfile:
    return get_baseline_taxonomy_profile(resolve_path(data_root))


def per_organ_metric_dict(
    evaluator: MetricsEvaluator, prefix: str
) -> Dict[str, float]:
    """Flatten per-organ aggregates into WandB keys ``{prefix}/organ/{Organ}/msd_mm_mean`` etc.

    Public helper for scripts/tests; the trainer's epoch W&B path uses
    ``bapmos.evaluation.baseline_epoch_monitoring.per_organ_wandb_dict``.
    """
    out: Dict[str, float] = {}
    for organ in evaluator.organs:
        agg = evaluator.aggregate_metrics(organ_name=organ)
        if agg is None:
            continue
        base = f"{prefix}/organ/{organ}"
        for key, wandb_suffix in (
            ("msd_mm_mean", "msd_mm_mean"),
            ("hd95_mm_mean", "hd95_mm_mean"),
            ("dice_mean", "dice_mean"),
        ):
            val = agg.get(key)
            if val is not None:
                out[f"{base}/{wandb_suffix}"] = float(val)
    return out


def split_summary_to_log_dict(
    summary: Dict[str, Optional[float]], prefix: str
) -> Dict[str, float]:
    """Map organ_balanced_validation_summary keys to WandB names (public helper)."""
    out: Dict[str, float] = {}
    msd = summary.get("val_msd")
    hd = summary.get("val_hd95")
    dice = summary.get("val_dice")
    if msd is not None:
        out[f"{prefix}/msd_mm"] = float(msd)
    if hd is not None:
        out[f"{prefix}/hd95_mm"] = float(hd)
    if dice is not None:
        out[f"{prefix}/dice_organ_balanced"] = float(dice)
    return out


def validation_objective_msd(
    evaluator: MetricsEvaluator,
    *,
    metric: str,
    organ_name: Optional[str] = None,
    kfold_n: int = 5,
    kfold_seed: int = 42,
) -> Optional[float]:
    """Score used for checkpoint selection and BO (may differ from organ-balanced MSD)."""
    if metric == "organ_balanced_msd":
        return organ_balanced_validation_summary(evaluator).get("val_msd")
    if metric == "ptv_kfold_msd":
        if not organ_name:
            return None
        return ptv_kfold_msd_mm(
            evaluator,
            organ_name,
            n_folds=kfold_n,
            seed=kfold_seed,
        )
    raise ValueError(
        f"Unknown evaluation.objective_metric {metric!r}; "
        "use 'organ_balanced_msd' or 'ptv_kfold_msd'"
    )


def upsample_pred_classes_to_gt(
    pred_classes: np.ndarray,
    gt_full: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample low-res SAM class predictions to full GT slice shape.

    Used by train/val metrics passes (native-resolution MSD/HD95). Stratified
    **test** export instead resizes pred+GT to ``PFUS1_BASELINE_EVAL_SIZE`` (256)
    via ``export_stratified_test_bundle(eval_size=...)`` for cross-method parity;
    keep this helper on native until an HPO cutover aligns train/val with test.
    Loss remains at 256×256 either way.
    """
    gt_np = np.asarray(gt_full).astype(np.uint8)
    pred = np.asarray(pred_classes).astype(np.uint8)
    if pred.shape != gt_np.shape[:2]:
        pred = cv2.resize(
            pred,
            (gt_np.shape[1], gt_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return pred, gt_np


@torch.no_grad()
def run_split_metrics_pass(
    trainer: "BAPMOSTrainer",
    loader,
    *,
    split: str,
    arms: Optional[Mapping[str, str]] = None,
) -> Tuple[float, float, Dict[str, Optional[float]], MetricsEvaluator, Dict[str, int]]:
    """
    One full pass over *loader* (train or val): loss, batch-mean dice, organ-balanced MSD/HD95.

    MSD/HD95 currently use native GT resolution (see ``upsample_pred_classes_to_gt``).
    Published test bundles use ``eval_size=256`` for method/baseline parity.
    """
    trainer.model.eval()
    evaluator = MetricsEvaluator(
        pixel_spacing=tuple(trainer.pixel_spacing),
        organs=list(trainer.evaluator_organ_labels),
    )
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    use_arms = dict(arms) if arms is not None else trainer._arms_for_eval(split)
    prompt_counts = empty_prompt_counts()

    for batch in loader:
        images = batch["image"]
        masks = batch["mask"]
        filenames = batch["filename"]
        emb_paths = batch.get("embedding_path")
        cached_packs = batch.get("sam_cached_pack")
        for i in range(len(images)):
            image_rgb = images[i]
            mask_multi = masks[i]
            if hasattr(mask_multi, "numpy"):
                mask_multi = mask_multi.numpy()
            sid = str(filenames[i])
            if int(np.asarray(mask_multi).max()) == 0:
                continue
            ep = emb_paths[i] if emb_paths is not None else None
            cp = cached_packs[i] if cached_packs is not None else None
            out = trainer._forward_sample(
                image_rgb,
                mask_multi,
                sid,
                use_arms,
                train=False,
                embedding_path=ep,
                cached_embedding_pack=cp,
            )
            if out is None:
                continue
            merge_prompt_counts(prompt_counts, use_arms)
            logits, gt = out
            total_loss += float(trainer.loss_fn(logits, gt).item())
            total_dice += multiclass_dice_from_logits(logits, gt, trainer.num_classes)

            pred_cls = (
                torch.argmax(torch.softmax(logits, dim=1), dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            pred_cls, gt_np = upsample_pred_classes_to_gt(pred_cls, mask_multi)
            append_multiclass_validation_slice(
                evaluator,
                pred_classes=pred_cls,
                gt_classes=gt_np,
                image_id=sid,
                class_mapping=trainer.multiclass_eval_mapping,
                slice_idx=n,
            )
            n += 1

    trainer.model.train()
    avg_loss = total_loss / max(n, 1)
    avg_dice = total_dice / max(n, 1)
    summary = organ_balanced_validation_summary(evaluator) if n else {
        "val_msd": None,
        "val_hd95": None,
        "val_dice": None,
    }
    return avg_loss, avg_dice, summary, evaluator, prompt_counts


@torch.no_grad()
def validate_block_per_organ_probe_metrics(
    trainer: "BAPMOSTrainer",
    val_dataset,
    probe_indices: np.ndarray,
) -> Dict[str, OrganProbeMetrics]:
    """Block-end Dice / MSD / HD95 per organ on a fixed val probe (bandit reward signal)."""
    trainer.model.eval()
    sums: Dict[str, Dict[str, float]] = {
        k: {"dice": 0.0, "msd_mm": 0.0, "hd95_mm": 0.0} for k in trainer.organ_keys
    }
    counts: Dict[str, int] = {k: 0 for k in trainer.organ_keys}
    arms = dict(trainer.scheduler.current_arms)
    if not arms:
        # Should not happen mid-block; fallback selects arms without committing rewards.
        logger.warning(
            "validate_block_per_organ_probe_metrics: scheduler.current_arms empty; "
            "falling back to select_arm() (does not increment bandit t/N_a)."
        )
        arms = {o: trainer.policy.select_arm(o) for o in trainer.organ_keys}

    for idx in probe_indices:
        sample = val_dataset[int(idx)]
        image_rgb = sample["image"]
        mask_multi = sample["mask"]
        if hasattr(image_rgb, "numpy"):
            image_rgb = image_rgb.numpy()
        if hasattr(mask_multi, "numpy"):
            mask_multi = mask_multi.numpy()
        sid = str(sample["filename"])
        out = trainer._forward_sample(
            image_rgb,
            mask_multi,
            sid,
            arms,
            train=False,
            embedding_path=sample.get("embedding_path"),
            cached_embedding_pack=sample.get("sam_cached_pack"),
        )
        if out is None:
            continue
        logits, _ = out
        pred_cls = (
            torch.argmax(torch.softmax(logits, dim=1), dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        for organ_key, class_id in trainer.organ_to_class.items():
            gt_bin = (mask_multi == class_id).astype(np.uint8)
            if gt_bin.sum() == 0:
                continue
            pred_bin = (pred_cls == class_id).astype(np.uint8)
            pred_rs = cv2.resize(
                pred_bin,
                (gt_bin.shape[1], gt_bin.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            metrics = compute_all_metrics(
                pred_rs,
                gt_bin,
                trainer.pixel_spacing,
                organ_name=organ_key,
            )
            if not metrics.get("valid_boundary"):
                continue
            msd = metrics.get("msd_mm")
            hd95 = metrics.get("hd95_mm")
            dice = metrics.get("dice")
            if msd is None or hd95 is None or dice is None:
                continue
            if not all(np.isfinite(float(x)) for x in (msd, hd95, dice)):
                continue
            sums[organ_key]["dice"] += float(dice)
            sums[organ_key]["msd_mm"] += float(msd)
            sums[organ_key]["hd95_mm"] += float(hd95)
            counts[organ_key] += 1

    trainer.model.train()
    out: Dict[str, OrganProbeMetrics] = {}
    for k in trainer.organ_keys:
        if counts[k] > 0:
            out[k] = OrganProbeMetrics(
                dice=sums[k]["dice"] / counts[k],
                msd_mm=sums[k]["msd_mm"] / counts[k],
                hd95_mm=sums[k]["hd95_mm"] / counts[k],
            )
    return out


@torch.no_grad()
def validate_block_per_organ_msd_probe(
    trainer: "BAPMOSTrainer",
    val_dataset,
    probe_indices: np.ndarray,
) -> Dict[str, float]:
    """Block-end MSD per organ key on a fixed val probe (legacy MSD-only bandit signal)."""
    organ_metrics = validate_block_per_organ_probe_metrics(trainer, val_dataset, probe_indices)
    return {
        k: (organ_metrics[k]["msd_mm"] if k in organ_metrics else float("inf"))
        for k in trainer.organ_keys
    }


def build_block_wandb_dict(
    organ_msds: Mapping[str, float],
    rewards: Mapping[str, float],
    total_blocks: int,
    *,
    organ_metrics: Optional[Mapping[str, OrganProbeMetrics]] = None,
) -> Dict[str, Any]:
    # Separate top-level namespace from epoch ``bandit/*`` (avoids W&B step-metric overlap).
    log: Dict[str, Any] = {"bandit_block/total_blocks": int(total_blocks)}
    metrics_by_organ = organ_metrics or {}
    for k, v in organ_msds.items():
        if np.isfinite(v):
            log[f"bandit_block/{k}_msd_mm"] = float(v)
    for k, m in metrics_by_organ.items():
        log[f"bandit_block/{k}_hd95_mm"] = float(m["hd95_mm"])
        log[f"bandit_block/{k}_dice"] = float(m["dice"])
    for k, v in rewards.items():
        log[f"bandit_block/{k}_reward"] = float(v)
    return log


_ARM_INDEX = {"box": 0, "point": 1, "both": 2}


def build_bandit_epoch_wandb_dict(
    policy: "PerOrganBanditPolicy",
    *,
    total_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Epoch-level bandit stats aligned with legacy optimization trainer W&B keys."""
    log: Dict[str, Any] = {}
    all_stats = policy.get_all_statistics()
    agg = all_stats["aggregated"]
    log["bandit/total_blocks"] = int(
        total_blocks if total_blocks is not None else agg.get("total_blocks", 0)
    )
    log["bandit/total_pulls"] = int(agg.get("total_pulls", 0))
    for organ, organ_stats in all_stats["per_organ"].items():
        organ_key = organ.lower()
        avg_rewards = organ_stats.get("arm_avg_rewards") or {}
        best_arm = organ_stats.get("best_arm")
        if best_arm is None and avg_rewards:
            best_arm = max(avg_rewards.items(), key=lambda x: x[1])[0]
        if best_arm is not None:
            log[f"bandit/{organ_key}_best_arm"] = _ARM_INDEX.get(best_arm, -1)
        for arm in policy.arms:
            log[f"bandit/{organ_key}_arm_{arm}_count"] = int(
                organ_stats.get("arm_counts", {}).get(arm, 0)
            )
            log[f"bandit/{organ_key}_arm_{arm}_avg_reward"] = float(
                avg_rewards.get(arm, 0.0)
            )
            log[f"bandit/{organ_key}_arm_{arm}_selection_rate"] = float(
                organ_stats.get("arm_selection_rates", {}).get(arm, 0.0)
            )
    return log


def empty_prompt_counts() -> Dict[str, int]:
    return {"box": 0, "point": 0, "both": 0}


def merge_prompt_counts(
    base: Dict[str, int], arms: Mapping[str, str], *, multiplier: int = 1
) -> None:
    for arm in arms.values():
        if arm in base:
            base[arm] += int(multiplier)


def _write_test_metric_csvs(evaluator: MetricsEvaluator, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
    evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
    evaluator.export_failure_analysis_csv(metrics_dir / "failure_analysis.csv", top_n=20)
    _export_per_organ_csv(evaluator, metrics_dir / "per_organ_metrics.csv")


def export_test_split_metrics(
    trainer: "BAPMOSTrainer",
    test_loader,
    output_dir: Optional[Path],
    *,
    canonical_method_dir: Optional[Path] = None,
    method_slug: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate test split with best checkpoint; write JSON + optional CSV bundle."""
    import torch as _torch

    run_dir = Path(trainer.run_dir) if trainer.run_dir else Path(".")
    best_path = run_dir / "best_checkpoint.pth"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}")

    try:
        ckpt = _torch.load(best_path, map_location=trainer.device, weights_only=False)
    except TypeError:
        ckpt = _torch.load(best_path, map_location=trainer.device)
    trainer.model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
    if ckpt.get("bandit_state"):
        trainer.policy.load_state(ckpt["bandit_state"])
    trainer.epoch = int(ckpt.get("epoch_index", ckpt.get("epoch", 0)))

    test_arms = trainer.policy.get_best_arms_per_organ()
    avg_loss, avg_dice, summary, evaluator, _ = run_split_metrics_pass(
        trainer, test_loader, split="test", arms=test_arms
    )
    test_msd = summary.get("val_msd")
    test_hd95 = summary.get("val_hd95")
    test_rep_dice = summary.get("val_dice")

    if output_dir is not None:
        _write_test_metric_csvs(evaluator, Path(output_dir))

    if canonical_method_dir is not None:
        canon = Path(canonical_method_dir)
        canon.mkdir(parents=True, exist_ok=True)
        _write_test_metric_csvs(evaluator, canon / "metrics")

    out: Dict[str, Any] = {
        "test_loss": float(avg_loss),
        "test_dice": float(avg_dice),
        "test_msd_mm": float(test_msd) if test_msd is not None else None,
        "test_hd95_mm": float(test_hd95) if test_hd95 is not None else None,
        "test_dice_organ_balanced": float(test_rep_dice) if test_rep_dice is not None else None,
        "best_val_msd_mm": float(trainer.best_val_msd),
        "best_epoch": int(ckpt.get("epoch_index", ckpt.get("epoch", 0))) + 1,
        "test_prompt_arms": dict(test_arms),
        "_evaluator": evaluator,
    }
    with open(run_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in out.items() if k != "_evaluator"}, f, indent=2)

    if canonical_method_dir is not None:
        canon = Path(canonical_method_dir)
        with open(canon / "test_results.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in out.items() if k != "_evaluator"}, f, indent=2)
        if method_slug and cfg is not None:
            from bapmos.method.paths import write_bapmos_evaluation_meta

            common = cfg.get("common", cfg)
            write_bapmos_evaluation_meta(
                canon,
                checkpoint=best_path,
                data_root=common["data_root"],
                method_slug=method_slug,
                cfg=cfg,
                extra={"test_prompt_arms": out.get("test_prompt_arms")},
            )

    return out


def _export_per_organ_csv(evaluator: MetricsEvaluator, path: Path) -> None:
    import pandas as pd

    rows = []
    for organ in evaluator.organs:
        agg = evaluator.aggregate_metrics(organ_name=organ)
        if agg is not None:
            rows.append(agg)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def format_test_results_wandb(out: Dict[str, Any]) -> Dict[str, float]:
    log: Dict[str, float] = {}
    if out.get("test_loss") is not None:
        log["test/loss"] = float(out["test_loss"])
    if out.get("test_dice") is not None:
        log["test/dice"] = float(out["test_dice"])
    if out.get("test_msd_mm") is not None:
        log["test/msd_mm"] = float(out["test_msd_mm"])
    if out.get("test_hd95_mm") is not None:
        log["test/hd95_mm"] = float(out["test_hd95_mm"])
    if out.get("test_dice_organ_balanced") is not None:
        log["test/dice_organ_balanced"] = float(out["test_dice_organ_balanced"])
    return log
