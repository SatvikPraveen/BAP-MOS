"""
Taxonomy-aware SAM decoder-only fine-tuning with per-organ box prompts.

Supports simulation, real clinical (Case 1/2), and PFUS1 taxonomies via
``get_baseline_taxonomy_profile``. Checkpoint selection: minimum ``val_msd``.
"""

import json
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
import random
from typing import Optional

# Path setup not needed - using package-relative imports

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import wandb

from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from ..metrics.baseline_validation_metrics import (
    append_multiclass_validation_slice,
    gt_uint8_from_tensor,
    multiclass_pred_uint8_from_logits,
    organ_balanced_validation_summary,
)
from bapmos.metrics.checkpoint_selection import (
    add_checkpoint_objective_cli,
    apply_checkpoint_objective_cli,
    is_better_checkpoint,
    parse_checkpoint_objective_config,
)
from ..evaluation.baseline_epoch_monitoring import (
    BASELINE_METRICS_CSV_FIELDS,
    build_epoch_wandb_log,
    build_test_wandb_log,
    csv_metric_cell,
    init_external_baseline_wandb,
    per_organ_wandb_dict,
    summarize_evaluator_metrics,
    validation_checkpoint_scores,
)
from ..external_baselines.baseline_training_protocol import (
    force_external_baseline_ce_dice_loss,
    log_epoch_wandb,
    make_constant_lr_scheduler,
)
from ..optimization.metrics import MetricsEvaluator
from ..losses.kervadec_style import BoundaryDistMapCache, DecoderLossComputer, DecoderLossConfig
from ..paths import (
    dataset_bundle_tag,
    project_root,
    resolve_model_checkpoint,
    resolve_training_data_root,
    resolve_under_project,
)
from ..train.training_taxonomy import PFUS1_SPLITS_SUBDIR, get_baseline_taxonomy_profile, log_baseline_taxonomy_startup
from bapmos.method.prompt_geometry import jitter_box as bapmos_jitter_box
from .dataset_multi_organ import (
    MultiOrganDataset,
    compute_per_organ_boxes,
    multi_organ_collate_fn,
)

# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Seed worker for deterministic DataLoader with num_workers>0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def stable_rng_from_id(sample_id: str, base_seed: int):
    # stable hash (FNV-1a)
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)

def _box_jitter_px(cfg: dict) -> int:
    if cfg.get("box_jitter_px") is not None:
        return int(cfg["box_jitter_px"])
    return int(cfg.get("box_noise_pixels", 0))


def multi_class_dice_from_logits(logits, gt_multi, num_classes=5):
    """
    Compute mean dice across all classes (except background).
    
    Args:
        logits: (1, num_classes, 256, 256) - logits for each class
        gt_multi: (1, 1, 256, 256) - multi-class ground truth {0,1,2,3,4}
        num_classes: Number of classes (5)
    
    Returns:
        Mean dice score across foreground classes (1,2,3,4)
    """
    # Convert logits to predictions
    probs = torch.softmax(logits, dim=1)  # (1, 5, 256, 256)
    pred_classes = torch.argmax(probs, dim=1, keepdim=True)  # (1, 1, 256, 256)
    
    dices = []
    
    # Compute dice for each foreground class (skip background=0)
    for c in range(1, num_classes):
        pred_c = (pred_classes == c).float()
        gt_c = (gt_multi == c).float()
        
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        
        if union > 0:
            dice = (2.0 * inter + 1e-7) / (union + 1e-7)
            dices.append(dice.item())
    
    return np.mean(dices) if dices else 0.0

# -----------------------------
# Trainer
# -----------------------------
class SAMMultiOrganTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = sam_model_registry["vit_b"](checkpoint=config["sam_checkpoint"]).to(self.device)
        self.model.train()

        # Freeze encoder + prompt encoder; train decoder only
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False
        for p in self.model.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True

        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

        # Optimizer (decoder params only)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=config["lr"], weight_decay=1e-4)
        if bool(config.get("flat_lr", True)):
            self.scheduler = make_constant_lr_scheduler(self.optimizer)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["max_epochs"], eta_min=1e-7
            )

        self._taxonomy = get_baseline_taxonomy_profile(config["data_root"])
        self.num_classes = self._taxonomy.num_classes
        log_baseline_taxonomy_startup(self._taxonomy, prefix="SAMMultiOrganBoxTrainer")

        self.best_val_msd = float("inf")
        self.patience = config.get("patience", 40)
        self._checkpoint_objective = parse_checkpoint_objective_config(config)
        self.val_msd_min_delta = float(self._checkpoint_objective.min_delta)
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.decoder_loss = DecoderLossComputer(
            DecoderLossConfig.from_training_config(
                config, max_epochs=int(config["max_epochs"])
            )
        )
        self._boundary_dist_cache = (
            BoundaryDistMapCache(self.num_classes)
            if self.decoder_loss.mode == "kervadec"
            else None
        )
        self._resume_start_epoch = 0
        self._train_generator_ref = None
        self.compute_train_boundary_metrics = bool(
            config.get("compute_train_boundary_metrics", True)
        )

    @torch.no_grad()
    def _collect_boundary_metrics_from_loader(
        self, loader
    ) -> tuple[dict, MetricsEvaluator]:
        """Full-split MSD/HD95 on train data (eval forward, no grad)."""
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
            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                if mask_multi.max() == 0:
                    continue
                out = self._forward_one(image_rgb, mask_multi, fn, is_train=False)
                if out is None:
                    continue
                logits, gt = out
                pred_u8 = multiclass_pred_uint8_from_logits(logits)
                gt_u8 = gt_uint8_from_tensor(gt)
                append_multiclass_validation_slice(
                    evaluator,
                    pred_classes=pred_u8,
                    gt_classes=gt_u8,
                    image_id=str(fn),
                    class_mapping=self._taxonomy.multiclass_eval_mapping,
                    slice_idx=n,
                )
                n += 1
        if was_training:
            self.model.train()
        summary = summarize_evaluator_metrics(evaluator)
        return (summary if summary is not None else {}), evaluator

    def loss_fn(self, logits, gt_multi, dist_maps=None):
        """Multi-class segmentation loss (CE+Dice or scheduled Kervadec)."""
        return self.decoder_loss(
            logits,
            gt_multi,
            self.num_classes,
            epoch=self.current_epoch,
            dist_maps=dist_maps,
        )

    def _prepare_boundary_dist_maps(self, gt, sample_id=None):
        """Return cached signed φ_G for resized GT (Kervadec); None for CE+Dice."""
        if self._boundary_dist_cache is None:
            return None
        label_hw = gt.detach().squeeze().long().cpu().numpy().astype(np.int64, copy=False)
        key = None if sample_id is None else f"{sample_id}|{label_hw.shape[0]}x{label_hw.shape[1]}"
        dist = self._boundary_dist_cache.get_or_compute(
            label_hw,
            cache_key=key,
            device=self.device,
            dtype=torch.float32,
        )
        return dist.unsqueeze(0)

    def resume_from(self, ckpt: dict, train_generator=None):
        """Restore decoder training state (epoch-boundary checkpoints only)."""
        if ckpt.get("format_version") != 2:
            raise ValueError(
                "Resume requires format_version=2 checkpoints produced by this trainer version."
            )
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if bool(self.cfg.get("flat_lr", True)):
            self.scheduler = make_constant_lr_scheduler(self.optimizer)
        else:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        best = ckpt.get("best_val_msd_tracked")
        if best is None:
            best = ckpt.get("best_val_msd")
        self.best_val_msd = float(best) if best is not None else float("inf")
        self.epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        ei = int(ckpt.get("epoch_index", 0))
        self._resume_start_epoch = ei + 1

        rs = ckpt.get("rng_state")
        if rs:
            torch.set_rng_state(rs["torch"])
            random.setstate(rs["random"])
            np.random.set_state(rs["numpy"])
            if rs.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda"])
        if train_generator is not None and ckpt.get("train_generator_state") is not None:
            train_generator.set_state(ckpt["train_generator_state"])
        self._train_generator_ref = train_generator
        print(
            f"[resume] Loaded epoch-boundary checkpoint after epoch {ei + 1} "
            f"(continuing from epoch index {self._resume_start_epoch})."
        )

    def _prepare_image(self, image_rgb):
        original_size = image_rgb.shape[:2]
        image_resized = self.transform.apply_image(image_rgb)
        x = torch.as_tensor(image_resized, device=self.device).float()
        x = x.permute(2, 0, 1).contiguous()

        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3, 1, 1)
        x = (x - pixel_mean) / pixel_std

        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        x = x.unsqueeze(0)  # (1,3,1024,1024)
        return x, original_size

    def _prepare_mask(self, mask_multi):
        """Resize multi-class mask to 256x256."""
        mask_resized = cv2.resize(
            mask_multi.astype(np.float32), 
            (256, 256), 
            interpolation=cv2.INTER_NEAREST
        )
        gt = torch.as_tensor(mask_resized, device=self.device, dtype=torch.float32)
        gt = gt.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
        return gt

    def _box_prompt(self, mask_multi, original_size, sample_id, is_train: bool):
        """Per-organ tight boxes from GT, with optional train-time jitter."""
        organ_boxes_dict = compute_per_organ_boxes(
            mask_multi, organ_to_class=self._taxonomy.organ_to_class
        )
        base = self.cfg["seed"]
        epoch_offset = (self.current_epoch * 1000) if is_train else 0
        rng = stable_rng_from_id(sample_id, base + epoch_offset)

        jitter_px = _box_jitter_px(self.cfg)
        boxes_list = []
        present_organs = []
        for organ_name in self._taxonomy.organ_keys:
            box = organ_boxes_dict[organ_name]
            if box is not None:
                if is_train and jitter_px > 0:
                    box = bapmos_jitter_box(
                        np.asarray(box, dtype=np.int64),
                        jitter_px,
                        original_size,
                        rng,
                    ).tolist()
                boxes_list.append(box)
                present_organs.append(organ_name)

        if len(boxes_list) == 0:
            return None

        boxes_array = np.array(boxes_list)
        boxes_t = self.transform.apply_boxes(boxes_array, original_size)
        boxes_t = torch.as_tensor(boxes_t, device=self.device, dtype=torch.float32)
        return boxes_t, present_organs

    def _forward_one(self, image_rgb, mask_multi, filename, is_train: bool):
        """Forward pass for one sample with per-organ box prompts."""
        x, original_size = self._prepare_image(image_rgb)
        gt = self._prepare_mask(mask_multi)

        with torch.no_grad():
            img_emb = self.model.image_encoder(x)

        prompt_result = self._box_prompt(mask_multi, original_size, str(filename), is_train=is_train)
        if prompt_result is None:
            return None

        boxes_t, present_organs = prompt_result
        num_organs = len(present_organs)

        unique_classes = np.unique(mask_multi)
        allowed = set(range(self.num_classes))
        assert set(unique_classes).issubset(allowed), f"Invalid mask classes: {unique_classes}"

        organ_masks = []
        for i in range(num_organs):
            box_i = boxes_t[i : i + 1, :]
            with torch.no_grad():
                sparse, dense = self.model.prompt_encoder(
                    points=None,
                    boxes=box_i.unsqueeze(1),
                    masks=None,
                )
            low_res_mask, _ = self.model.mask_decoder(
                image_embeddings=img_emb,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            organ_masks.append(low_res_mask.squeeze(1))

        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        logits = torch.zeros(1, self.num_classes, 256, 256, device=self.device, dtype=torch.float32)
        for idx, organ_name in enumerate(present_organs):
            class_id = self._taxonomy.organ_to_class[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]

        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        return logits, gt

    def run_epoch(self, loader, train: bool, return_evaluator: bool = False):
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

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]

                if mask_multi.max() == 0:
                    continue

                if train:
                    out = self._forward_one(image_rgb, mask_multi, fn, is_train=train)
                else:
                    with torch.no_grad():
                        out = self._forward_one(image_rgb, mask_multi, fn, is_train=train)
                if out is None:
                    continue
                logits, gt = out

                loss = self.loss_fn(
                    logits, gt, dist_maps=self._prepare_boundary_dist_maps(gt, sample_id=fn)
                )
                d = multi_class_dice_from_logits(logits, gt, self.num_classes)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.mask_decoder.parameters(), 1.0)
                    self.optimizer.step()
                elif val_evaluator is not None:
                    pred_u8 = multiclass_pred_uint8_from_logits(logits)
                    gt_u8 = gt_uint8_from_tensor(gt)
                    append_multiclass_validation_slice(
                        val_evaluator,
                        pred_classes=pred_u8,
                        gt_classes=gt_u8,
                        image_id=str(fn),
                        class_mapping=self._taxonomy.multiclass_eval_mapping,
                    )

                total_loss += float(loss.item())
                total_dice += float(d)
                n += 1

        if n == 0:
            if not train and val_evaluator is not None:
                summ = organ_balanced_validation_summary(val_evaluator)
                if return_evaluator:
                    return 0.0, 0.0, summ, val_evaluator
                return 0.0, 0.0, summ
            return (0.0, 0.0, None, None) if return_evaluator else (0.0, 0.0, None)

        avg_loss = total_loss / n
        avg_dice = total_dice / n
        if not train and val_evaluator is not None:
            summ = organ_balanced_validation_summary(val_evaluator)
            if return_evaluator:
                return avg_loss, avg_dice, summ, val_evaluator
            return avg_loss, avg_dice, summ
        return (avg_loss, avg_dice, None, None) if return_evaluator else (avg_loss, avg_dice, None)

    def save_checkpoint(self, path: Path, val_msd: Optional[float] = None):
        rng_bundle = {
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_bundle["cuda"] = torch.cuda.get_rng_state_all()
        ckpt = {
            "format_version": 2,
            "completed_full_epoch": True,
            "epoch_index": int(self.current_epoch),
            "prompt_type": "box",
            "num_classes": self.num_classes,
            "best_val_msd_tracked": float(self.best_val_msd),
            "val_msd_at_save": float(val_msd) if val_msd is not None else None,
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "model_state": {
                "mask_decoder": self.model.mask_decoder.state_dict(),
            },
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.cfg,
            "rng_state": rng_bundle,
        }
        gen = getattr(self, "_train_generator_ref", None)
        if gen is not None:
            ckpt["train_generator_state"] = gen.get_state()
        torch.save(ckpt, path)

    def train_loop(self, train_loader, val_loader):
        run_dir = Path(self.cfg["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=BASELINE_METRICS_CSV_FIELDS).writeheader()

        organ_labels = list(self._taxonomy.evaluator_organ_labels)
        start_epoch = int(getattr(self, "_resume_start_epoch", 0))
        for epoch in range(start_epoch, self.cfg["max_epochs"]):
            self.current_epoch = epoch
            t0 = time.time()

            train_loss, train_dice, _ = self.run_epoch(train_loader, train=True)
            val_loss, val_dice, val_summ, val_evaluator = self.run_epoch(
                val_loader, train=False, return_evaluator=True
            )

            train_metrics: dict = {}
            train_per_organ_wandb: dict = {}
            if self.compute_train_boundary_metrics:
                train_metrics, train_evaluator = self._collect_boundary_metrics_from_loader(
                    train_loader
                )
                train_per_organ_wandb = per_organ_wandb_dict(
                    train_evaluator, organ_labels, "train"
                )

            val_metrics = summarize_evaluator_metrics(val_evaluator) if val_evaluator else None
            ckpt_scores = (
                validation_checkpoint_scores(val_evaluator, self.cfg, organ_labels)
                if val_evaluator is not None
                else None
            )
            val_msd = ckpt_scores.primary_msd if ckpt_scores is not None else None
            val_hd95 = None
            val_rep_dice = None
            if ckpt_scores is not None:
                val_hd95 = ckpt_scores.organ_balanced_hd95
                val_rep_dice = ckpt_scores.organ_balanced_dice
            elif val_metrics:
                val_msd = val_metrics.get("msd_mm_organ_balanced") or val_metrics.get("msd_mm")
                val_hd95 = val_metrics.get("hd95_mm_organ_balanced") or val_metrics.get("hd95_mm")
                val_rep_dice = val_metrics.get("dice_organ_balanced")
            elif val_summ:
                val_msd = val_summ.get("val_msd")
                val_hd95 = val_summ.get("val_hd95")
                val_rep_dice = val_summ.get("val_dice")
            val_per_organ_wandb = (
                per_organ_wandb_dict(val_evaluator, organ_labels, "val")
                if val_evaluator is not None
                else {}
            )
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
                msd_s = f"{val_msd:.4f}" if val_msd is not None else "N/A"
                print(
                    f"No MSD improvement for {self.epochs_without_improvement}/{self.patience} epochs "
                    f"(val_msd={msd_s} mm, best={self.best_val_msd:.4f} mm)"
                )

                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"Best val_msd: {self.best_val_msd:.4f} mm")
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
                "val_msd": csv_metric_cell(val_metrics, "msd_mm_organ_balanced")
                or csv_metric_cell(val_metrics, "msd_mm")
                or (val_msd if val_msd is not None else ""),
                "val_hd95": csv_metric_cell(val_metrics, "hd95_mm_organ_balanced")
                or csv_metric_cell(val_metrics, "hd95_mm")
                or (val_hd95 if val_hd95 is not None else ""),
                "val_dice_organ_balanced": csv_metric_cell(val_metrics, "dice_organ_balanced")
                or (val_rep_dice if val_rep_dice is not None else ""),
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
                train_per_organ_wandb=train_per_organ_wandb or None,
            )
            log_epoch_wandb(log_dict)

            tr_msd = (
                train_metrics.get("msd_mm_organ_balanced") or train_metrics.get("msd_mm")
                if train_metrics
                else None
            )
            tr_hd = (
                train_metrics.get("hd95_mm_organ_balanced") or train_metrics.get("hd95_mm")
                if train_metrics
                else None
            )
            msd_s = f"{val_msd:.4f}" if val_msd is not None else "N/A"
            hd_s = f"{val_hd95:.4f}" if val_hd95 is not None else "N/A"
            tr_msd_s = f"{tr_msd:.4f}" if tr_msd is not None else "N/A"
            tr_hd_s = f"{tr_hd:.4f}" if tr_hd is not None else "N/A"
            print(
                f"[{epoch+1}/{self.cfg['max_epochs']}] "
                f"Train: loss={train_loss:.4f}, dice={train_dice:.4f}, msd={tr_msd_s} mm, hd95={tr_hd_s} mm | "
                f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}, msd={msd_s} mm, hd95={hd_s} mm | "
                f"LR={lr:.2e} | {time.time()-t0:.1f}s"
            )

            self.scheduler.step()

    def evaluate_test_once(self, test_loader):
        """Evaluate on test set using best checkpoint."""
        run_dir = Path(self.cfg["run_dir"])
        best_path = run_dir / "best_checkpoint.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")

        try:
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=self.device)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        ei = int(ckpt.get("epoch_index", ckpt.get("epoch", 0)))
        print(f"Loaded best checkpoint from epoch {ei + 1}")

        test_loss, test_dice, test_summ = self.run_epoch(test_loader, train=False)
        test_msd = test_summ.get("val_msd") if test_summ else None
        test_hd95 = test_summ.get("val_hd95") if test_summ else None
        test_rep_dice = test_summ.get("val_dice") if test_summ else None

        out = {
            "test_loss": float(test_loss),
            "test_dice": float(test_dice),
            "test_msd": float(test_msd) if test_msd is not None else None,
            "test_hd95": float(test_hd95) if test_hd95 is not None else None,
            "test_dice_organ_balanced": float(test_rep_dice) if test_rep_dice is not None else None,
            "best_val_msd": float(self.best_val_msd),
            "best_epoch": int(ei + 1),
        }
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(out, f, indent=2)

        from bapmos.paths import (
            method_slug_from_checkpoint,
            method_test_output_dir_from_checkpoint,
            write_method_evaluation_meta,
        )

        out_root = method_test_output_dir_from_checkpoint(best_path, self.cfg["data_root"])
        write_method_evaluation_meta(
            out_root,
            checkpoint=best_path,
            data_root=self.cfg["data_root"],
            method_slug=method_slug_from_checkpoint(best_path),
            split="test",
            extra={"baseline": "multiorgan_sam", "prompt_type": "box"},
        )
        with open(out_root / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        tv = self.cfg.get("test_visualizations") or {}
        if tv.get("enabled"):
            from bapmos.evaluation.baseline_multiorgan_viz import (
                run_multiorgan_baseline_test_visualizations,
            )

            run_multiorgan_baseline_test_visualizations(
                self, test_loader, run_dir, tv, output_root=out_root
            )

        test_log = build_test_wandb_log(
            {
                "test_loss": test_loss,
                "test_dice": test_dice,
                "test_msd": test_msd,
                "test_hd95": test_hd95,
                "test_dice_organ_balanced": test_rep_dice,
            }
        )
        if test_log:
            wandb.log(test_log)

        print(
            f"[TEST] loss={test_loss:.4f} dice={test_dice:.4f} "
            f"msd={test_msd} hd95={test_hd95} organ_balanced_dice={test_rep_dice}"
        )
        print(f"Test results saved to: {run_dir / 'test_results.json'}")
        return out

    def export_test_split_metrics(self, test_loader, output_dir: Optional[Path] = None) -> dict:
        """Evaluate test split with best checkpoint; optional CSV export under ``output_dir``."""
        run_dir = Path(self.cfg["run_dir"])
        best_path = run_dir / "best_checkpoint.pth"
        if not best_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")

        try:
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=self.device)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        ei = int(ckpt.get("epoch_index", ckpt.get("epoch", 0)))
        print(f"Loaded best checkpoint from epoch {ei + 1}")

        test_loss, test_dice, test_summ, evaluator = self.run_epoch(
            test_loader, train=False, return_evaluator=True
        )
        test_msd = test_summ.get("val_msd") if test_summ else None
        test_hd95 = test_summ.get("val_hd95") if test_summ else None
        test_rep_dice = test_summ.get("val_dice") if test_summ else None

        if output_dir is not None and evaluator is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluator.export_per_slice_csv(output_dir / "per_slice_metrics.csv")
            evaluator.export_summary_csv(output_dir / "summary_metrics.csv")
            evaluator.export_failure_analysis_csv(output_dir / "failure_analysis.csv", top_n=20)
            print(f"Metrics exported to: {output_dir}")

        out = {
            "test_loss": float(test_loss),
            "test_dice": float(test_dice),
            "test_msd": float(test_msd) if test_msd is not None else None,
            "test_hd95": float(test_hd95) if test_hd95 is not None else None,
            "test_dice_organ_balanced": float(test_rep_dice) if test_rep_dice is not None else None,
            "best_val_msd": float(self.best_val_msd),
            "best_epoch": int(ei + 1),
        }
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(out, f, indent=2)
        return out

# -----------------------------
# Main
# -----------------------------
def main():
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root (images/, masks/combined_masks/, splits/). Default: bundled case1 real data.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="SAM ViTk-B checkpoint. Default: models/sam_base/sam_vit_b_01ec64.pth under project root.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--box_jitter_px", type=int, default=None, help="BAP-MOS train box jitter (0..N px per edge).")
    parser.add_argument("--box_noise_pixels", type=int, default=10, help="Legacy alias for --box_jitter_px.")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--skip_test_eval",
        action="store_true",
        help="Skip post-train test export (Table I runs infer separately).",
    )
    parser.add_argument("--wandb_project", type=str, default="sam-medical-multiple-segmentation")
    parser.add_argument(
        "--splits_subdir",
        type=str,
        default=None,
        help="Split lists under data_root (default: splits_stratified; PFUS1: splits_patient_70_15_15_seed42).",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default=None,
        help=(
            "Checkpoint parent directory. Default: runs/<bundle>/Baseline where <bundle> "
            "is inferred from --data_root (case_1, case_2, simulation, pfus1, other)."
        ),
    )
    parser.add_argument(
        "--save-test-visualizations",
        action="store_true",
        help="After testing, save qualitative panels under run_dir/test_results/ (same style as optimization inference).",
    )
    parser.add_argument(
        "--test-viz-selection",
        choices=["all", "random", "worst_msd", "best_msd", "per_patient_even"],
        default="all",
    )
    parser.add_argument("--test-viz-max", type=int, default=None)
    parser.add_argument("--test-viz-seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from last_checkpoint.pth path, or 'auto' with fixed --run_name under --run_root.",
    )
    add_checkpoint_objective_cli(parser)
    args = parser.parse_args()

    data_root = args.data_root or str(resolve_training_data_root("case1"))
    bundle = dataset_bundle_tag(data_root)

    ckpt_obj = None
    resume_arg = (args.resume or "").strip()
    run_root_base = args.run_root or f"runs/{bundle}/Baseline"

    if resume_arg:
        if resume_arg.lower() == "auto":
            if not args.run_name:
                raise ValueError("--resume auto requires --run_name matching the interrupted job.")
            ckpt_fp = Path(run_root_base) / args.run_name / "last_checkpoint.pth"
        else:
            ckpt_fp = resolve_under_project(resume_arg)
        if not ckpt_fp.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
        ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
        config = ckpt_obj["config"]
        run_dir = Path(config["run_dir"])
        run_name = config["run_name"]
        data_root = config["data_root"]
        sam_ckpt = config["sam_checkpoint"]
        bundle = dataset_bundle_tag(data_root)
    else:
        sam_ckpt = args.sam_checkpoint or "models/sam_base/sam_vit_b_01ec64.pth"
        sam_resolved = resolve_model_checkpoint(sam_ckpt)
        if not sam_resolved.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_ckpt!r} (tried under {root}/models and alternate model roots)"
            )
        args.sam_checkpoint = str(sam_resolved)

        seed_everything(args.seed, deterministic=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"{bundle}_multiorgan_decoder_box_realdata_{timestamp}"
        run_dir = Path(run_root_base) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        _tax_profile = get_baseline_taxonomy_profile(data_root)
        splits_subdir = args.splits_subdir
        if splits_subdir is None:
            splits_subdir = (
                PFUS1_SPLITS_SUBDIR
                if _tax_profile.taxonomy_name == "pfus1_eight_organ"
                else "splits_stratified"
            )
        jitter_px = (
            int(args.box_jitter_px)
            if args.box_jitter_px is not None
            else int(args.box_noise_pixels)
        )
        config = {
            "data_root": data_root,
            "sam_checkpoint": args.sam_checkpoint,
            "run_dir": str(run_dir),
            "run_name": run_name,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "splits_subdir": splits_subdir,
            "box_jitter_px": jitter_px,
            "box_noise_pixels": jitter_px,
            "patience": args.patience,
            "compute_train_boundary_metrics": True,
            "prompt_type": "box",
            "num_classes": _tax_profile.num_classes,
            "taxonomy": _tax_profile.taxonomy_name,
            "dataset": bundle,
            "baseline": "sam_multiorgan_box",
            "test_visualizations": {
                "enabled": args.save_test_visualizations,
                "selection": args.test_viz_selection,
                "max": args.test_viz_max,
                "seed": args.test_viz_seed,
            },
        }
        # Paper SAM+box baseline: regional CE+Dice only (Kervadec is BAP-MOS method-only).
        force_external_baseline_ce_dice_loss(config)
        apply_checkpoint_objective_cli(config, args)

    # Always pin CE+Dice (including resume from older configs that may have carried kervadec).
    force_external_baseline_ce_dice_loss(config)

    if ckpt_obj is not None:
        seed_everything(config["seed"], deterministic=True)

    sam_resolved = resolve_model_checkpoint(config["sam_checkpoint"])
    if not sam_resolved.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {config['sam_checkpoint']!r} (tried under {root}/models)"
        )
    config["sam_checkpoint"] = str(sam_resolved)

    if resume_arg:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    ds_tag = bundle.replace("-", "_")
    init_external_baseline_wandb(
        project=args.wandb_project,
        run_name=run_name,
        config=config,
        run_dir=run_dir,
        resumed=ckpt_obj is not None,
        group=f"sam_box_{ds_tag}",
        tags=[
            f"dataset={bundle}",
            "baseline=sam_multiorgan_box",
            "prompt=box",
        ],
    )

    print("\n" + "="*60)
    print(f"Multi-Organ SAM Training (Box Prompts) — bundle={bundle}")
    print("="*60)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("="*60 + "\n")

    # Datasets
    train_dataset = MultiOrganDataset(
        data_root, split="train", splits_subdir=config["splits_subdir"]
    )
    val_dataset = MultiOrganDataset(
        data_root, split="val", splits_subdir=config["splits_subdir"]
    )
    from bapmos.paths import should_skip_in_train_global_test

    skip_test_eval = should_skip_in_train_global_test(
        data_root,
        config["splits_subdir"],
        force_skip=bool(args.skip_test_eval),
    )
    if skip_test_eval and not args.skip_test_eval:
        print(
            f"[info] Skipping in-train test eval "
            f"(no global {config['splits_subdir']}/test.txt; "
            "pooled uses site_tests/; run stratified inference_output separately)."
        )

    test_loader = None
    if not skip_test_eval:
        test_dataset = MultiOrganDataset(
            data_root, split="test", splits_subdir=config["splits_subdir"]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=multi_organ_collate_fn,
        )

    # Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=multi_organ_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_organ_collate_fn,
    )

    # Train
    trainer = SAMMultiOrganTrainer(config)
    trainer._train_generator_ref = g
    if ckpt_obj is not None:
        trainer.resume_from(ckpt_obj, g)
    trainer.train_loop(train_loader, val_loader)
    
    if not skip_test_eval and test_loader is not None:
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        trainer.evaluate_test_once(test_loader)

    wandb.finish()
    print(f"\nTraining complete! Best val MSD: {trainer.best_val_msd:.4f} mm")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
