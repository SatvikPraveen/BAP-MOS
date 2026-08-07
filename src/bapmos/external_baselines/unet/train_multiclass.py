"""
Train a multiclass U-Net (``segmentation_models_pytorch``) on ``MultiOrganDataset``.

Example:

    python -m bapmos.external_baselines.unet.train_multiclass \\
        --data_root data/prostate/pooled \\
        --encoder resnet34

Checkpoints: ``runs/ExternalBaselines/unet/<run_name>/`` by default, or under
``--run_root`` (e.g. ``runs/pfus1/ExternalBaselines/unet/``).

Resume with ``--resume path/to/last_checkpoint.pth`` or ``--resume auto`` (requires
``--run_name``). Optional ``--max_epochs`` / ``--patience`` override values stored in the
checkpoint (useful when extending a run that already hit the original ``max_epochs``).

Optional test figures (same layout as multi-organ SAM baselines): pass
``--save-test-visualizations`` (and ``--test-viz-*`` to subsample).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from bapmos.baseline_validation_metrics import (
    append_multiclass_validation_slice,
    gt_uint8_from_tensor,
    multiclass_pred_uint8_from_logits,
    organ_balanced_validation_summary,
)
from bapmos.checkpoint_selection import (
    add_checkpoint_objective_cli,
    apply_checkpoint_objective_cli,
    is_better_checkpoint,
    parse_checkpoint_objective_config,
)
from bapmos.evaluation.baseline_epoch_monitoring import (
    BASELINE_METRICS_CSV_FIELDS,
    build_epoch_wandb_log,
    build_test_wandb_log,
    csv_metric_cell,
    init_external_baseline_wandb,
    per_organ_wandb_dict,
    summarize_evaluator_metrics,
    validation_checkpoint_scores,
)
from bapmos.external_baselines.baseline_training_protocol import (
    CLINICAL_BATCH_SIZE,
    CLINICAL_MAX_EPOCHS,
    CLINICAL_PATIENCE,
    PFUS1_BATCH_SIZE,
    PFUS1_MAX_EPOCHS,
    PFUS1_PATIENCE,
    force_external_baseline_ce_dice_loss,
    log_epoch_wandb,
    make_constant_lr_scheduler,
)
from bapmos.external_baselines.common import (
    external_run_dir,
    imagenet_normalize_bchw,
    mean_fg_dice_from_logits,
    multiclass_ce_dice_loss,
    resize_for_baseline,
    seed_everything,
    seed_worker,
)
from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset, multi_organ_collate_fn
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.paths import (
    inference_output_dir_for_checkpoint,
    method_slug_from_checkpoint,
    write_method_evaluation_meta,
    project_root,
    resolve_training_data_root,
    resolve_under_project,
)
from bapmos.training_taxonomy import default_splits_subdir, get_baseline_taxonomy_profile, log_baseline_taxonomy_startup


def apply_unet_training_cli_overrides(
    config: dict,
    *,
    max_epochs: Optional[int],
    patience: Optional[int],
) -> None:
    """Mutate ``config`` when CLI explicitly sets training length (fresh or resume)."""
    if max_epochs is not None:
        config["max_epochs"] = int(max_epochs)
    if patience is not None:
        config["patience"] = int(patience)


class UNetMultiOrganTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._taxonomy = get_baseline_taxonomy_profile(config["data_root"])
        self.num_classes = int(config["num_classes"])
        log_baseline_taxonomy_startup(self._taxonomy, prefix="UNetMultiOrganTrainer")

        enc = config.get("encoder", "resnet34")
        self.model = smp.Unet(
            encoder_name=enc,
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config["lr"]))
        if bool(config.get("flat_lr", True)):
            self.scheduler = make_constant_lr_scheduler(self.optimizer)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(config["max_epochs"]), eta_min=1e-7
            )

        self.best_val_msd = float("inf")
        self.patience = int(config.get("patience", 40))
        self._checkpoint_objective = parse_checkpoint_objective_config(config)
        self.val_msd_min_delta = float(self._checkpoint_objective.min_delta)
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.compute_train_boundary_metrics = bool(
            config.get("compute_train_boundary_metrics", True)
        )
        self.run_test_after_train = bool(config.get("run_test_after_train", True))
        # External baselines: regional CE+Dice only (never Kervadec).
        force_external_baseline_ce_dice_loss(self.cfg)

    def _forward_one(self, image_rgb, mask_multi, filename, is_train: bool):
        """Same interface as ``SAMMultiOrganTrainer._forward_one`` for shared test viz."""
        del filename
        if mask_multi.max() == 0:
            return None
        x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
        x = imagenet_normalize_bchw(x.to(self.device))
        y_dev = y.to(self.device)
        if is_train:
            logits = self.model(x)
        else:
            with torch.no_grad():
                logits = self.model(x)
        gt = y_dev.unsqueeze(1).float()
        return logits, gt

    @torch.no_grad()
    def _collect_boundary_metrics_from_loader(
        self, loader
    ) -> tuple[Dict[str, Optional[float]], MetricsEvaluator]:
        """Full-split MSD/Dice/HD95 (no grad), matching legacy BAP-MOS epoch monitoring."""
        was_training = self.model.training
        self.model.eval()
        evaluator = MetricsEvaluator(
            pixel_spacing=self._taxonomy.pixel_spacing_mm,
            organs=list(self._taxonomy.evaluator_organ_labels),
        )
        n = 0
        for batch in loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]
            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            fns: list[str] = []
            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                if mask_multi.max() == 0:
                    continue
                x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
                xs.append(x)
                ys.append(y)
                fns.append(str(filenames[i]))
            if not xs:
                continue
            xb = torch.cat(xs, dim=0).to(self.device)
            yb = torch.cat(ys, dim=0).to(self.device)
            xb = imagenet_normalize_bchw(xb)
            logits = self.model(xb)
            bs = int(logits.shape[0])
            for j in range(bs):
                logits_j = logits[j : j + 1]
                y_j = yb[j : j + 1]
                pred_u8 = multiclass_pred_uint8_from_logits(logits_j)
                gt_u8 = gt_uint8_from_tensor(y_j.unsqueeze(1).float())
                append_multiclass_validation_slice(
                    evaluator,
                    pred_classes=pred_u8,
                    gt_classes=gt_u8,
                    image_id=fns[j],
                    class_mapping=self._taxonomy.multiclass_eval_mapping,
                    slice_idx=n,
                )
                n += 1
        if was_training:
            self.model.train()
        summary = summarize_evaluator_metrics(evaluator)
        return (summary if summary is not None else {}), evaluator

    def run_epoch(self, loader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_dice, n = 0.0, 0.0, 0
        val_evaluator: Optional[MetricsEvaluator] = None
        if not train:
            val_evaluator = MetricsEvaluator(
                pixel_spacing=self._taxonomy.pixel_spacing_mm,
                organs=list(self._taxonomy.evaluator_organ_labels),
            )

        for batch in loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]

            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            fns: list = []

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                if mask_multi.max() == 0:
                    continue
                x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
                xs.append(x)
                ys.append(y)
                fns.append(fn)

            if not xs:
                continue

            xb = torch.cat(xs, dim=0).to(self.device)
            yb = torch.cat(ys, dim=0).to(self.device)
            xb = imagenet_normalize_bchw(xb)

            if train:
                logits = self.model(xb)
            else:
                with torch.no_grad():
                    logits = self.model(xb)

            loss = multiclass_ce_dice_loss(
                logits,
                yb,
                self.num_classes,
            )

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            bs = int(logits.shape[0])
            batch_dice = 0.0
            for j in range(bs):
                logits_j = logits[j : j + 1]
                y_j = yb[j : j + 1]
                batch_dice += mean_fg_dice_from_logits(logits_j, y_j, self.num_classes)
                if not train and val_evaluator is not None:
                    pred_u8 = multiclass_pred_uint8_from_logits(logits_j)
                    gt_u8 = gt_uint8_from_tensor(y_j.unsqueeze(1).float())
                    append_multiclass_validation_slice(
                        val_evaluator,
                        pred_classes=pred_u8,
                        gt_classes=gt_u8,
                        image_id=str(fns[j]),
                        class_mapping=self._taxonomy.multiclass_eval_mapping,
                    )
            batch_dice /= bs

            total_loss += float(loss.item()) * bs
            total_dice += batch_dice * bs
            n += bs

        if n == 0:
            if not train and val_evaluator is not None:
                return 0.0, 0.0, summarize_evaluator_metrics(val_evaluator), val_evaluator
            return 0.0, 0.0, None, None

        avg_loss = total_loss / n
        avg_dice = total_dice / n
        if not train and val_evaluator is not None:
            return avg_loss, avg_dice, summarize_evaluator_metrics(val_evaluator), val_evaluator
        return avg_loss, avg_dice, None, None

    def save_checkpoint(self, path: Path, val_msd: Optional[float] = None) -> None:
        ckpt = {
            "format_version": 1,
            "epoch": self.current_epoch,
            "backbone": "smp_unet",
            "num_classes": self.num_classes,
            "best_val_msd": float(self.best_val_msd),
            "val_msd_at_save": float(val_msd) if val_msd is not None else None,
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def apply_checkpoint(self, ckpt: dict) -> int:
        """Restore trainable state. Returns 0-based **next** epoch index to run."""
        fv = ckpt.get("format_version")
        if fv is not None and int(fv) != 1:
            raise ValueError(
                f"Unsupported checkpoint format_version={fv!r}; expected 1 or legacy (missing key)."
            )
        if int(ckpt.get("num_classes", -1)) != int(self.num_classes):
            raise ValueError(
                f"Checkpoint num_classes={ckpt.get('num_classes')} does not match trainer {self.num_classes}."
            )
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if bool(self.cfg.get("flat_lr", True)):
            self.scheduler = make_constant_lr_scheduler(self.optimizer)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(self.cfg["max_epochs"]), eta_min=1e-7
            )
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        best = ckpt.get("best_val_msd")
        self.best_val_msd = float(best) if best is not None else float("inf")
        self.epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        last_done = int(ckpt.get("epoch", -1))
        if last_done < 0:
            raise ValueError("Checkpoint missing or invalid 'epoch' (last completed epoch index).")
        start = last_done + 1
        print(
            f"[resume] Loaded last_checkpoint | resume from epoch index {start} "
            f"(completed through epoch index {last_done}) | best_val_msd={self.best_val_msd:.4f}"
        )
        return start

    def train_loop(self, train_loader, val_loader, *, start_epoch: int = 0) -> None:
        run_dir = Path(self.cfg["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=BASELINE_METRICS_CSV_FIELDS).writeheader()

        organ_labels = list(self._taxonomy.evaluator_organ_labels)
        max_ep = int(self.cfg["max_epochs"])
        if start_epoch >= max_ep:
            print(f"[resume] start_epoch={start_epoch} >= max_epochs={max_ep}; nothing to train.")
            return

        for epoch in range(start_epoch, max_ep):
            self.current_epoch = epoch
            t0 = time.time()
            train_loss, train_dice, _, _ = self.run_epoch(train_loader, train=True)
            val_loss, val_dice, val_metrics, val_evaluator = self.run_epoch(
                val_loader, train=False
            )

            val_per_organ_wandb = (
                per_organ_wandb_dict(val_evaluator, organ_labels, "val")
                if val_evaluator is not None
                else {}
            )

            train_metrics: Dict[str, Optional[float]] = {}
            train_per_organ_wandb: Dict[str, float] = {}
            if self.compute_train_boundary_metrics:
                train_metrics, train_evaluator = self._collect_boundary_metrics_from_loader(
                    train_loader
                )
                train_per_organ_wandb = per_organ_wandb_dict(
                    train_evaluator, organ_labels, "train"
                )

            val_msd = None
            ckpt_scores = (
                validation_checkpoint_scores(val_evaluator, self.cfg, organ_labels)
                if val_evaluator is not None
                else None
            )
            if ckpt_scores is not None:
                val_msd = ckpt_scores.primary_msd
            elif val_metrics:
                val_msd = val_metrics.get("msd_mm_organ_balanced") or val_metrics.get("msd_mm")
            lr = self.optimizer.param_groups[0]["lr"]

            self.save_checkpoint(run_dir / "last_checkpoint.pth", val_msd=val_msd)

            if val_loss == 0.0 and val_dice == 0.0:
                print("WARNING: val had 0 valid samples this epoch")
            elif is_better_checkpoint(val_msd, self.best_val_msd, min_delta=self.val_msd_min_delta):
                self.best_val_msd = float(val_msd)
                self.save_checkpoint(run_dir / "best_checkpoint.pth", val_msd=val_msd)
                self.epochs_without_improvement = 0
                print(f"✓ New best {self._checkpoint_objective.metric}: {val_msd:.4f} mm")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_msd": csv_metric_cell(train_metrics, "msd_mm_organ_balanced")
                or csv_metric_cell(train_metrics, "msd_mm"),
                "train_hd95": csv_metric_cell(train_metrics, "hd95_mm_organ_balanced")
                or csv_metric_cell(train_metrics, "hd95_mm"),
                "train_dice_organ_balanced": csv_metric_cell(
                    train_metrics, "dice_organ_balanced"
                ),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_msd": val_msd if val_msd is not None else "",
                "val_hd95": (
                    ckpt_scores.organ_balanced_hd95
                    if ckpt_scores is not None
                    else csv_metric_cell(val_metrics, "hd95_mm_organ_balanced")
                    or csv_metric_cell(val_metrics, "hd95_mm")
                ),
                "val_dice_organ_balanced": (
                    ckpt_scores.organ_balanced_dice
                    if ckpt_scores is not None
                    else csv_metric_cell(val_metrics, "dice_organ_balanced")
                ),
                "val_ptv_hd95": ckpt_scores.ptv_hd95 if ckpt_scores is not None else "",
                "val_ptv_dice": ckpt_scores.ptv_dice if ckpt_scores is not None else "",
                "best_val_msd": self.best_val_msd,
                "lr": lr,
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=BASELINE_METRICS_CSV_FIELDS).writerow(row)

            log_dict = build_epoch_wandb_log(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_dice=train_dice,
                val_loss=val_loss,
                val_dice=val_dice,
                lr=lr,
                time_epoch_s=time.time() - t0,
                train_metrics=train_metrics or None,
                val_metrics=val_metrics,
                val_per_organ_wandb=val_per_organ_wandb,
                train_per_organ_wandb=train_per_organ_wandb,
            )
            log_epoch_wandb(log_dict)

            ob_msd = (val_metrics or {}).get("msd_mm_organ_balanced")
            ob_hd = (val_metrics or {}).get("hd95_mm_organ_balanced")
            msd_s = f"{ob_msd:.4f}" if ob_msd is not None else "N/A"
            hd_s = f"{ob_hd:.4f}" if ob_hd is not None else "N/A"
            print(
                f"[{epoch+1}/{self.cfg['max_epochs']}] "
                f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"val loss={val_loss:.4f} dice={val_dice:.4f} msd={msd_s} hd95={hd_s} mm | lr={lr:.2e}"
            )
            self.scheduler.step()

    def export_test_split_metrics(self, test_loader, output_dir: Optional[Path] = None) -> dict:
        """Run test split, optional CSV export under ``output_dir`` (e.g. ``output/pfus1/.../test``)."""
        run_dir = Path(self.cfg["run_dir"])
        best_path = run_dir / "best_checkpoint.pth"
        if not best_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
        try:
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])

        self.model.eval()
        evaluator = MetricsEvaluator(
            pixel_spacing=self._taxonomy.pixel_spacing_mm,
            organs=list(self._taxonomy.evaluator_organ_labels),
        )
        total_loss, total_dice, n = 0.0, 0.0, 0

        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]
            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                if mask_multi.max() == 0:
                    continue
                x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
                xb = imagenet_normalize_bchw(x.to(self.device))
                with torch.no_grad():
                    logits = self.model(xb)
                y_dev = y.to(self.device)  # (1, H, W) from resize_for_baseline
                loss = multiclass_ce_dice_loss(
                    logits,
                    y_dev,
                    self.num_classes,
                )
                pred_u8 = multiclass_pred_uint8_from_logits(logits)
                gt_u8 = gt_uint8_from_tensor(y_dev.unsqueeze(1).float())
                append_multiclass_validation_slice(
                    evaluator,
                    pred_classes=pred_u8,
                    gt_classes=gt_u8,
                    image_id=str(filenames[i]),
                    class_mapping=self._taxonomy.multiclass_eval_mapping,
                )
                dice = mean_fg_dice_from_logits(logits, y_dev, self.num_classes)
                total_loss += float(loss.item())
                total_dice += float(dice)
                n += 1

        test_summ = organ_balanced_validation_summary(evaluator) if n else None
        test_msd = test_summ.get("val_msd") if test_summ else None
        test_hd95 = test_summ.get("val_hd95") if test_summ else None
        test_rep_dice = test_summ.get("val_dice") if test_summ else None
        avg_loss = total_loss / max(n, 1)
        avg_dice = total_dice / max(n, 1)

        organ_labels = list(self._taxonomy.evaluator_organ_labels)
        if output_dir is not None:
            output_dir = Path(output_dir)
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
            evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
            evaluator.export_failure_analysis_csv(metrics_dir / "failure_analysis.csv", top_n=20)
            rows = []
            for organ in organ_labels:
                agg = evaluator.aggregate_metrics(organ_name=organ)
                if agg is not None:
                    rows.append(agg)
            if rows:
                import pandas as pd

                pd.DataFrame(rows).to_csv(output_dir / "per_organ_metrics.csv", index=False)

        metrics = summarize_evaluator_metrics(evaluator) or {}
        out = {
            "test_loss": float(avg_loss),
            "test_dice": float(avg_dice),
            "test_msd": float(test_msd) if test_msd is not None else None,
            "test_msd_mm": metrics.get("msd_mm_organ_balanced") or test_msd,
            "test_hd95": float(test_hd95) if test_hd95 is not None else None,
            "test_hd95_mm": metrics.get("hd95_mm_organ_balanced") or test_hd95,
            "test_dice_organ_balanced": float(test_rep_dice) if test_rep_dice is not None else None,
            "best_val_msd": float(self.best_val_msd),
        }
        with open(run_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return out, evaluator

    def evaluate_test_once(self, test_loader) -> dict:
        out, _ = self.export_test_split_metrics(test_loader, output_dir=None)
        return out


def wandb_dataset_label(data_root: str) -> str:
    """Short dataset label for W&B project/group (matches clinical scripts)."""
    from bapmos.paths import dataset_bundle_tag

    tag = dataset_bundle_tag(str(data_root))
    return {"case_1": "case1", "case_2": "case2"}.get(tag, tag)


def default_wandb_project(data_root: str) -> str:
    return f"{wandb_dataset_label(data_root)}_external_unet"


def init_unet_wandb(
    *,
    config: Dict[str, Any],
    wandb_project: str,
    run_dir: Path,
    resumed: bool,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
) -> None:
    """Initialize W&B with tags/group layout aligned to BAP-MOS production runs."""
    ds_label = wandb_dataset_label(str(config["data_root"]))
    tags = list(wandb_tags or [])
    tags.extend(
        [
            "multiclass",
            f"encoder={config.get('encoder', 'resnet34')}",
        ]
    )
    init_external_baseline_wandb(
        project=wandb_project,
        run_name=str(config["run_name"]),
        config={**config, "baseline": config.get("baseline", "unet")},
        run_dir=run_dir,
        resumed=resumed,
        wandb_entity=wandb_entity,
        group=wandb_group or f"unet_{ds_label}",
        tags=tags,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="SMP U-Net multiclass baseline")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--run_root",
        type=str,
        default=None,
        help=(
            "Parent directory for ExternalBaselines runs (subfolders unet/, medsam_init/). "
            "Default: runs/ExternalBaselines. Use runs/pfus1/ExternalBaselines for PFUS1 layout parity."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from last_checkpoint.pth: pass a path, or 'auto' to use "
            "<run_root>/unet/<run_name>/last_checkpoint.pth (requires --run_name)."
        ),
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Default: 100 (PFUS1) or 300 (prostate). When resuming, overrides checkpoint if set.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Default: 128 (PFUS1) or 1 (prostate).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--cosine_lr",
        action="store_true",
        help="Cosine LR decay (default: flat lr matching BAP-MOS production).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Default: 20 (PFUS1) or 40 (prostate). When resuming, overrides checkpoint if set.",
    )
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY env if set).",
    )
    p.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group (default: unet_<dataset> from data_root).",
    )
    p.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Extra comma-separated W&B tags.",
    )
    p.add_argument(
        "--splits_subdir",
        type=str,
        default=None,
        help="Split folder under data_root (default: taxonomy-specific).",
    )
    p.add_argument(
        "--save-test-visualizations",
        action="store_true",
        help="After testing, save 4-panel figures under run_dir/test_results/visualizations.",
    )
    p.add_argument(
        "--test-viz-selection",
        choices=["all", "random", "worst_msd", "best_msd", "per_patient_even"],
        default="all",
    )
    p.add_argument("--test-viz-max", type=int, default=None)
    p.add_argument("--test-viz-seed", type=int, default=42)
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Load best checkpoint and export test metrics only (no training).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Export test CSVs here (default for PFUS1: output/pfus1/.../test from checkpoint path).",
    )
    p.add_argument(
        "--no-train-boundary-metrics",
        action="store_true",
        help="Skip per-epoch train MSD/HD95 pass (faster; val boundary metrics still logged).",
    )
    p.add_argument(
        "--skip-test-after-train",
        action="store_true",
        help="Do not run test split export after training completes.",
    )
    add_checkpoint_objective_cli(p)
    args = p.parse_args()

    resume_arg = (args.resume or "").strip()
    ckpt_obj = None
    pr = project_root()

    if resume_arg:
        if resume_arg.lower() == "auto":
            if not args.run_name:
                raise ValueError("--resume auto requires --run_name matching the prior run directory.")
            rr = (args.run_root or "runs/ExternalBaselines").strip()
            base = Path(rr) if Path(rr).is_absolute() else (pr / rr)
            ckpt_fp = base.resolve() / "unet" / args.run_name / "last_checkpoint.pth"
            if not ckpt_fp.is_file():
                raise FileNotFoundError(f"--resume auto: missing checkpoint {ckpt_fp}")
        else:
            ckpt_fp = resolve_under_project(resume_arg)
            if not ckpt_fp.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
        try:
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu")
        config = copy.deepcopy(ckpt_obj["config"])
        apply_unet_training_cli_overrides(
            config,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        run_dir = Path(config["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (pr / run_dir).resolve()
        else:
            run_dir = run_dir.resolve()
        config["run_dir"] = str(run_dir)
        data_root = config["data_root"]
        print(f"[resume] Continuing run {config['run_name']!r} | run_dir={run_dir}")
    else:
        data_root = args.data_root or str(resolve_training_data_root("case1"))
        tax = get_baseline_taxonomy_profile(data_root)
        splits_subdir = args.splits_subdir or default_splits_subdir(data_root)
        distance_unit = "px" if "pfus1" in tax.taxonomy_name else "mm"
        is_pfus1 = "pfus1" in tax.taxonomy_name
        if args.max_epochs is not None:
            max_epochs = args.max_epochs
        else:
            max_epochs = PFUS1_MAX_EPOCHS if is_pfus1 else CLINICAL_MAX_EPOCHS
        if args.patience is not None:
            patience = args.patience
        else:
            patience = PFUS1_PATIENCE if is_pfus1 else CLINICAL_PATIENCE
        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            batch_size = PFUS1_BATCH_SIZE if is_pfus1 else CLINICAL_BATCH_SIZE
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"unet_{args.encoder}_{tax.taxonomy_name}_{ts}"
        run_dir = external_run_dir("unet", run_name, external_baselines_parent=args.run_root)
        config = {
            "baseline": "smp_unet",
            "data_root": str(resolve_under_project(data_root)),
            "encoder": args.encoder,
            "external_baselines_run_root": args.run_root,
            "run_dir": str(run_dir),
            "run_name": run_name,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "flat_lr": not args.cosine_lr,
            "seed": args.seed,
            "splits_subdir": splits_subdir,
            "patience": patience,
            "num_classes": tax.num_classes,
            "taxonomy": tax.taxonomy_name,
            "distance_unit": distance_unit,
            "pixel_spacing": list(tax.pixel_spacing_mm),
            "test_visualizations": {
                "enabled": args.save_test_visualizations,
                "selection": args.test_viz_selection,
                "max": args.test_viz_max,
                "seed": args.test_viz_seed,
            },
            "compute_train_boundary_metrics": not args.no_train_boundary_metrics,
            "run_test_after_train": not args.skip_test_after_train,
        }
        apply_checkpoint_objective_cli(config, args)

    force_external_baseline_ce_dice_loss(config)
    seed_everything(int(config["seed"]), deterministic=True)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    wandb_project = args.wandb_project or default_wandb_project(config["data_root"])
    extra_tags = (
        [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        if args.wandb_tags
        else None
    )
    init_unet_wandb(
        config=config,
        wandb_project=wandb_project,
        run_dir=run_dir,
        resumed=ckpt_obj is not None,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=extra_tags,
    )
    print(f"WandB project={wandb_project} run={config['run_name']} group=unet_{wandb_dataset_label(config['data_root'])}")

    train_ds = MultiOrganDataset(
        config["data_root"], split="train", splits_subdir=config["splits_subdir"]
    )
    val_ds = MultiOrganDataset(
        config["data_root"], split="val", splits_subdir=config["splits_subdir"]
    )
    skip_test_after_train = bool(args.skip_test_after_train)
    from bapmos.paths import should_skip_in_train_global_test

    # Pooled / missing global test.txt: never build MultiOrganDataset(split="test").
    # --test-only still needs a loader → use site-aware build_test_dataloader.
    auto_skip = should_skip_in_train_global_test(
        config["data_root"],
        config["splits_subdir"],
        force_skip=skip_test_after_train,
    )
    test_loader = None
    if args.test_only or not auto_skip:
        if auto_skip and args.test_only:
            from bapmos.method.data_adapter import build_test_dataloader

            test_loader = build_test_dataloader(
                config["data_root"],
                splits_subdir=config["splits_subdir"],
                batch_size=int(config["batch_size"]),
                num_workers=0,
            )
        elif not auto_skip:
            test_loader = DataLoader(
                MultiOrganDataset(
                    config["data_root"], split="test", splits_subdir=config["splits_subdir"]
                ),
                batch_size=int(config["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                collate_fn=multi_organ_collate_fn,
            )
    elif not skip_test_after_train and auto_skip:
        print(
            f"[info] Skipping in-train test eval "
            f"(no global {config['splits_subdir']}/test.txt; "
            "pooled uses site_tests/; run stratified inference_output separately)."
        )
        skip_test_after_train = True
        config["run_test_after_train"] = False


    g = torch.Generator()
    g.manual_seed(int(config["seed"]))
    bs = int(config["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=multi_organ_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_organ_collate_fn,
    )

    trainer = UNetMultiOrganTrainer(config)

    if args.test_only:
        if test_loader is None:
            raise ValueError("--test-only requires a test split (omit --skip-test-after-train).")
        if not resume_arg:
            raise ValueError("--test-only requires --resume auto or a checkpoint .pth path")
        out_dir = None
        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser()
            if not out_dir.is_absolute():
                out_dir = (pr / out_dir).resolve()
        else:
            ckpt_for_out = (
                (base / "unet" / args.run_name / "best_checkpoint.pth")
                if resume_arg.lower() == "auto"
                else ckpt_fp
            )
            out_dir = inference_output_dir_for_checkpoint(
                ckpt_for_out, config["data_root"], split="test"
            )
        result, evaluator = trainer.export_test_split_metrics(test_loader, output_dir=out_dir)
        organ_labels = list(trainer._taxonomy.evaluator_organ_labels)
        test_log = build_test_wandb_log(result, evaluator, organ_labels)
        if test_log:
            wandb.log(test_log)
        print(f"[TEST] {result}")
        if out_dir:
            print(f"Metrics exported to: {out_dir}")
        wandb.finish()
        return

    start_epoch = 0
    if ckpt_obj is not None:
        start_epoch = trainer.apply_checkpoint(ckpt_obj)
    trainer.train_loop(train_loader, val_loader, start_epoch=start_epoch)

    if trainer.run_test_after_train and test_loader is not None:
        test_out_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else inference_output_dir_for_checkpoint(
                run_dir / "best_checkpoint.pth", config["data_root"], split="test"
            )
        )
        if args.output_dir and not test_out_dir.is_absolute():
            test_out_dir = (pr / test_out_dir).resolve()
        write_method_evaluation_meta(
            test_out_dir,
            checkpoint=run_dir / "best_checkpoint.pth",
            data_root=config["data_root"],
            method_slug=method_slug_from_checkpoint(run_dir / "best_checkpoint.pth"),
            split="test",
            extra={"baseline": "smp_unet", "run_name": config.get("run_name")},
        )
        test_out, test_evaluator = trainer.export_test_split_metrics(
            test_loader, output_dir=test_out_dir
        )
        organ_labels = list(trainer._taxonomy.evaluator_organ_labels)
        test_log = build_test_wandb_log(test_out, test_evaluator, organ_labels)
        if test_log:
            wandb.log(test_log)
        tv = config.get("test_visualizations") or {}
        if tv.get("enabled"):
            from bapmos.evaluation.baseline_multiorgan_viz import (
                run_multiorgan_baseline_test_visualizations,
            )

            run_multiorgan_baseline_test_visualizations(
                trainer, test_loader, run_dir, tv, output_root=test_out_dir
            )
        print(f"[TEST] {test_out}")
    wandb.finish()
    print(f"Done. Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
