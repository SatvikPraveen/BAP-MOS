"""Log train/val MSD/HD95 to W&B during nnU-Net training (BAP-MOS / U-Net parity)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bapmos.evaluation.baseline_epoch_monitoring import (
    build_epoch_wandb_log,
    per_organ_wandb_dict,
    summarize_evaluator_metrics,
)
from bapmos.external_baselines.baseline_training_protocol import log_epoch_wandb
from bapmos.external_baselines.nnunet2d.evaluate_nnunet_predictions import (
    case_entries_for_ids,
    evaluate_nnunet_predictions_for_entries,
)
from bapmos.external_baselines.nnunet2d.export_nnunet_dataset import dataset_folder


def extend_ddp_timeout_for_boundary_eval(*, seconds: int | None = None) -> None:
    """Raise NCCL process-group timeout so rank-0 boundary infer can finish.

    Default PG timeout is ~10 minutes. Val-only sliding-window on ~2k PFUS1 cases
    routinely exceeds that; ``dist.barrier()`` then times out even when ranks are
    correctly waiting. Call on every rank after ``init_process_group``.
    """
    try:
        import torch.distributed as dist
        from datetime import timedelta

        if not (dist.is_available() and dist.is_initialized()):
            return
        if seconds is None:
            raw = os.environ.get("NNUNET_DDP_TIMEOUT_SEC", "7200").strip()
            seconds = max(600, int(raw or "7200"))
        timeout = timedelta(seconds=int(seconds))
        # Public helper when present (PyTorch 2.x).
        setter = getattr(dist, "distributed_c10d", None)
        set_fn = getattr(setter, "_set_pg_timeout", None) if setter is not None else None
        if set_fn is None:
            from torch.distributed.distributed_c10d import _set_pg_timeout as set_fn  # type: ignore

        set_fn(timeout)
    except Exception:
        pass


def _ddp_barrier() -> None:
    """Keep DDP ranks aligned while rank 0 runs sliding-window boundary eval."""
    try:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_world_size() <= 1:
            return
        extend_ddp_timeout_for_boundary_eval()
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception as exc:
        # Do not swallow: a failed barrier leaves ranks desynchronized.
        raise RuntimeError(f"DDP barrier after/before boundary eval failed: {exc}") from exc


def _boundary_interval(case_count: int) -> int:
    explicit = os.environ.get("NNUNET_BOUNDARY_METRICS_EVERY", "").strip()
    if explicit:
        return max(1, int(explicit))
    # Default: every epoch. Prefer val-only (see NNUNET_BOUNDARY_VAL_ONLY).
    if os.environ.get("NNUNET_BAPMOS_PROTOCOL", "1").strip().lower() not in ("0", "false", "no"):
        return 1
    return 1 if case_count <= 100 else 10


def _env_truthy(name: str, *, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


def _boundary_val_only() -> bool:
    """When true, skip train-cohort sliding-window infer each epoch (val still runs).

    Default **on**: full train+val (~12k PFUS1 cases) writing PNGs every epoch is
    prohibitively slow and quota-hostile. Opt into train+val with
    ``NNUNET_BOUNDARY_VAL_ONLY=0``.
    """
    return _env_truthy("NNUNET_BOUNDARY_VAL_ONLY", default="1")


def _boundary_train_every() -> int:
    """Optional train boundary metrics every N epochs (0 = never when val-only)."""
    raw = os.environ.get("NNUNET_BOUNDARY_TRAIN_EVERY", "0").strip()
    if not raw:
        return 0
    return max(0, int(raw))


def _splits_for_boundary_epoch(
    *,
    epoch: int,
    interval: int,
    tr_keys: list,
    val_keys: list,
) -> list[tuple[str, list, str]]:
    """Which cohorts to run sliding-window boundary eval on this epoch."""
    if epoch % interval != 0:
        return []

    splits: list[tuple[str, list, str]] = []
    if val_keys:
        splits.append(("val", list(val_keys), "boundary_val"))

    if not _boundary_val_only():
        if tr_keys:
            splits.append(("train", list(tr_keys), "boundary_train"))
        return splits

    train_every = _boundary_train_every()
    if train_every > 0 and epoch % train_every == 0 and tr_keys:
        splits.append(("train", list(tr_keys), "boundary_train"))
    return splits


def _resolve_case_mapping(dataset_id: int, dataset_name: str) -> Optional[Path]:
    env = os.environ.get("BAPMOS_NNUNET_CASE_MAPPING", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    raw = os.environ.get("nnUNet_raw", "").strip()
    if not raw:
        return None
    p = dataset_folder(Path(raw), dataset_id, dataset_name) / "case_mapping.json"
    return p if p.is_file() else None


def _resolve_data_root() -> Optional[str]:
    from bapmos.external_baselines.nnunet2d.bapmos_env import resolve_bapmos_data_root

    return resolve_bapmos_data_root()


def _evaluate_split_predictions(
    *,
    trainer: Any,
    predictions_dir: Path,
    case_mapping: Path,
    data_root: str,
    case_ids: list,
    split_label: str,
    tmp_root: Path,
) -> Tuple[Optional[Dict[str, Optional[float]]], Optional[Any]]:
    entries = case_entries_for_ids(case_mapping, list(case_ids))
    split_out = tmp_root / split_label
    evaluator, split_results = evaluate_nnunet_predictions_for_entries(
        predictions_dir=predictions_dir,
        entries=entries,
        data_root=Path(data_root),
        output_dir=split_out,
        split_label=split_label,
    )

    tax_organs = []
    try:
        from bapmos.training_taxonomy import get_baseline_taxonomy_profile

        tax_organs = list(get_baseline_taxonomy_profile(data_root).evaluator_organ_labels)
    except Exception:
        pass

    metrics_summary: Optional[Dict[str, Optional[float]]] = None
    if tax_organs:
        per_slice = split_out / "metrics" / "per_slice_metrics.csv"
        if per_slice.is_file():
            import pandas as pd

            from bapmos.legacy.optimization.metrics import MetricsEvaluator

            df = pd.read_csv(per_slice)
            ev = MetricsEvaluator(pixel_spacing=(1.0, 1.0), organs=tax_organs)
            for _, row in df.iterrows():
                ev.per_slice_metrics.append(row.to_dict())
            metrics_summary = summarize_evaluator_metrics(ev)
            evaluator = ev

    if metrics_summary is None and split_results:
        metrics_summary = {
            "msd_mm_organ_balanced": split_results.get(f"{split_label}_msd_mm"),
            "hd95_mm_organ_balanced": split_results.get(f"{split_label}_hd95_mm"),
            "dice_organ_balanced": split_results.get(f"{split_label}_dice_organ_balanced"),
            "msd_mm": split_results.get(f"{split_label}_msd_mm"),
            "hd95_mm": split_results.get(f"{split_label}_hd95_mm"),
        }

    return metrics_summary, evaluator


def log_epoch_boundary_wandb(
    trainer: Any,
    *,
    epoch: int,
    dataset_id: int,
    dataset_name: str,
    train_loss: float,
    val_loss: float,
    val_dice: float,
    lr: float,
    time_epoch_s: float,
) -> bool:
    """Full train+val inference; log BAP-MOS-style W&B metrics for this epoch."""
    if os.environ.get("NNUNET_BOUNDARY_METRICS", "1").strip().lower() in ("0", "false", "no"):
        return False
    if getattr(trainer, "local_rank", 0) != 0:
        return False

    tr_keys, val_keys = trainer.do_split()
    interval = _boundary_interval(max(len(tr_keys), len(val_keys)))
    split_plan = _splits_for_boundary_epoch(
        epoch=epoch, interval=interval, tr_keys=tr_keys, val_keys=val_keys
    )
    if not split_plan:
        return False

    case_mapping = _resolve_case_mapping(dataset_id, dataset_name)
    data_root = _resolve_data_root()
    if case_mapping is None or data_root is None:
        trainer.print_to_log_file(
            "Skipping boundary W&B metrics: set nnUNet_raw + BAPMOS_NNUNET_DATA_ROOT "
            "(or BAPMOS_NNUNET_CASE_MAPPING).",
            also_print_to_console=True,
        )
        return False

    infer = getattr(trainer, "perform_inference_on_split_keys", None)
    if infer is None:
        trainer.print_to_log_file(
            "Skipping boundary W&B metrics: trainer lacks perform_inference_on_split_keys.",
            also_print_to_console=True,
        )
        return False

    val_only = _boundary_val_only()
    eval_train_cases = next((len(keys) for label, keys, _ in split_plan if label == "train"), 0)
    eval_val_cases = next((len(keys) for label, keys, _ in split_plan if label == "val"), 0)
    mode = "val-only" if val_only and eval_train_cases == 0 else "train+val"
    trainer.print_to_log_file(
        f"Boundary metrics (epoch {epoch}, {mode}): "
        f"infer train={eval_train_cases} val={eval_val_cases} cases "
        f"(cohort sizes train={len(tr_keys)} val={len(val_keys)})",
        also_print_to_console=True,
    )

    train_metrics: Optional[Dict[str, Optional[float]]] = None
    val_metrics: Optional[Dict[str, Optional[float]]] = None
    val_evaluator = None
    train_per_organ: Dict[str, float] = {}
    val_per_organ: Dict[str, float] = {}
    pred_dirs: list[Path] = []
    t_boundary = time.time()

    try:
        from bapmos.external_baselines.nnunet2d.bapmos_env import resolve_boundary_tmpdir

        boundary_scratch = resolve_boundary_tmpdir()
        trainer.print_to_log_file(
            f"Boundary scratch dir: {boundary_scratch}",
            also_print_to_console=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="nnunet_boundary_eval_", dir=boundary_scratch
        ) as tmp:
            tmp_root = Path(tmp)
            for split_label, keys, folder in split_plan:
                if not keys:
                    continue
                pred_dir = infer(list(keys), folder)
                pred_dirs.append(pred_dir)
                split_metrics, evaluator = _evaluate_split_predictions(
                    trainer=trainer,
                    predictions_dir=pred_dir,
                    case_mapping=case_mapping,
                    data_root=data_root,
                    case_ids=list(keys),
                    split_label=split_label,
                    tmp_root=tmp_root,
                )
                if split_label == "train":
                    train_metrics = split_metrics
                    if evaluator is not None:
                        try:
                            from bapmos.training_taxonomy import get_baseline_taxonomy_profile

                            organs = list(
                                get_baseline_taxonomy_profile(data_root).evaluator_organ_labels
                            )
                            train_per_organ = per_organ_wandb_dict(evaluator, organs, "train")
                        except Exception:
                            pass
                else:
                    val_metrics = split_metrics
                    if evaluator is not None:
                        val_evaluator = evaluator
                        try:
                            from bapmos.training_taxonomy import get_baseline_taxonomy_profile

                            organs = list(
                                get_baseline_taxonomy_profile(data_root).evaluator_organ_labels
                            )
                            val_per_organ = per_organ_wandb_dict(evaluator, organs, "val")
                        except Exception:
                            pass
    except Exception as exc:
        trainer.print_to_log_file(
            f"Boundary W&B metrics failed @ epoch {epoch}: {exc}",
            also_print_to_console=True,
        )
        return False
    finally:
        if os.environ.get("NNUNET_KEEP_VALIDATION_DIR", "0") != "1":
            for pred_dir in pred_dirs:
                shutil.rmtree(pred_dir, ignore_errors=True)

    train_dice = 0.0
    if train_metrics and train_metrics.get("dice_organ_balanced") is not None:
        train_dice = float(train_metrics["dice_organ_balanced"])

    epoch_no = int(epoch) + 1
    log_dict = build_epoch_wandb_log(
        epoch=epoch_no,
        train_loss=float(train_loss),
        train_dice=train_dice,
        val_loss=float(val_loss),
        val_dice=float(val_dice),
        lr=float(lr),
        time_epoch_s=float(time_epoch_s) + (time.time() - t_boundary),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_per_organ_wandb=val_per_organ or None,
        train_per_organ_wandb=train_per_organ or None,
    )
    log_epoch_wandb(log_dict)

    trainer.print_to_log_file(
        f"W&B boundary metrics @ epoch {epoch}: "
        f"train_msd={log_dict.get('train/msd_mm')} train_hd95={log_dict.get('train/hd95_mm')} "
        f"val_msd={log_dict.get('val/msd_mm')} val_hd95={log_dict.get('val/hd95_mm')}",
        also_print_to_console=True,
    )
    if val_evaluator is not None:
        trainer._bapmos_val_evaluator = val_evaluator  # type: ignore[attr-defined]
    return True


def maybe_log_boundary_metrics(
    trainer: Any,
    *,
    epoch: int,
    dataset_id: int,
    dataset_name: str,
) -> None:
    """Backward-compatible entry: infer losses from nnU-Net logger and log full metrics."""
    try:
        train_loss = float(trainer.logger.get_value("train_losses", step=-1))
        val_loss = float(trainer.logger.get_value("val_losses", step=-1))
        val_dice = float(trainer.logger.get_value("mean_fg_dice", step=-1))
    except Exception:
        train_loss = val_loss = val_dice = 0.0

    try:
        t0 = float(trainer.logger.get_value("epoch_start_timestamps", step=-1))
        t1 = float(trainer.logger.get_value("epoch_end_timestamps", step=-1))
        time_epoch_s = max(0.0, t1 - t0)
    except Exception:
        time_epoch_s = 0.0

    lr = float(trainer.optimizer.param_groups[0]["lr"])
    # All ranks must enter/leave together. Extend timeout first — the barrier itself
    # is an NCCL collective subject to the same 10m default that killed prior jobs.
    extend_ddp_timeout_for_boundary_eval()
    _ddp_barrier()
    try:
        log_epoch_boundary_wandb(
            trainer,
            epoch=epoch,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=val_dice,
            lr=lr,
            time_epoch_s=time_epoch_s,
        )
    finally:
        _ddp_barrier()


def maybe_log_val_boundary_metrics(
    trainer: Any,
    *,
    epoch: int,
    dataset_id: int,
    dataset_name: str,
) -> None:
    maybe_log_boundary_metrics(
        trainer, epoch=epoch, dataset_id=dataset_id, dataset_name=dataset_name
    )
