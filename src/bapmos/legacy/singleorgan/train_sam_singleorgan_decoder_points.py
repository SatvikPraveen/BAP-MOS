"""
SAM decoder-only fine-tuning for a single organ (binary segmentation).

Checkpoint layout matches multi-organ baselines: mask_decoder, config with num_classes=2,
prompt_type points, plus organ name and task marker.
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

from bapmos.baseline_validation_metrics import (
    append_single_organ_validation_slice,
    organ_balanced_validation_summary,
)
from bapmos.multiorgan.dataset_multi_organ import (
    sample_negative_points_from_ring,
    sample_positive_points,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.paths import project_root, resolve_training_data_root, resolve_under_project
from bapmos.training_taxonomy import get_baseline_taxonomy_profile, log_baseline_taxonomy_startup
from .dataset_single_organ import SingleOrganDataset


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stable_rng_from_id(sample_id: str, base_seed: int):
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def multi_class_dice_from_logits(logits, gt_multi, num_classes: int):
    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(probs, dim=1, keepdim=True)
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred_classes == c).float()
        gt_c = (gt_multi == c).float()
        inter = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        if union > 0:
            dices.append(((2.0 * inter + 1e-7) / (union + 1e-7)).item())
    return np.mean(dices) if dices else 0.0


class SAMSingleOrganTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.organ = config["organ"]
        self.num_classes = 2
        self._taxonomy = get_baseline_taxonomy_profile(config["data_root"])
        if self.organ not in self._taxonomy.organ_to_class:
            raise ValueError(
                f"organ {self.organ!r} not in taxonomy {self._taxonomy.taxonomy_name}; "
                f"expected one of {list(self._taxonomy.organ_keys)}"
            )
        self._organ_to_class = self._taxonomy.organ_to_class
        self._eval_organ_label = {
            o.key: o.evaluator_label for o in self._taxonomy.organ_definitions
        }[self.organ]
        log_baseline_taxonomy_startup(self._taxonomy, prefix=f"SAMSingleOrganPointsTrainer({self.organ})")

        self.model = sam_model_registry["vit_b"](checkpoint=config["sam_checkpoint"]).to(self.device)
        self.model.train()
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False
        for p in self.model.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True

        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=config["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["max_epochs"], eta_min=1e-7
        )

        self.best_val_msd = float("inf")
        self.patience = config.get("patience", 40)
        self.val_msd_min_delta = float(config.get("val_msd_min_delta", 1e-6))
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self._resume_start_epoch = 0
        self._train_generator_ref = None

    def loss_fn(self, logits, gt_bin):
        gt_flat = gt_bin.squeeze(1).long()
        ce_loss = nn.functional.cross_entropy(logits, gt_flat, reduction="mean")
        probs = torch.softmax(logits, dim=1)
        dice_losses = []
        for c in range(1, self.num_classes):
            pred_c = probs[:, c : c + 1, :, :]
            gt_c = (gt_bin == c).float()
            inter = (pred_c * gt_c).sum(dim=(2, 3))
            union = pred_c.sum(dim=(2, 3)) + gt_c.sum(dim=(2, 3))
            dice = (2.0 * inter + 1e-7) / (union + 1e-7)
            dice_losses.append(1.0 - dice.mean())
        dice_loss = torch.stack(dice_losses).mean()
        return ce_loss + dice_loss

    def resume_from(self, ckpt: dict, train_generator=None):
        """Restore decoder training state (epoch-boundary checkpoints only)."""
        if ckpt.get("format_version") != 2:
            raise ValueError(
                "Resume requires format_version=2 checkpoints produced by this trainer version."
            )
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
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
        return x.unsqueeze(0), original_size

    def _prepare_mask(self, mask_bin):
        mask_resized = cv2.resize(
            mask_bin.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST
        )
        gt = torch.as_tensor(mask_resized, device=self.device, dtype=torch.float32)
        return gt.unsqueeze(0).unsqueeze(0)

    def _points_prompt(self, mask_bin, original_size, sample_id, is_train: bool):
        base = self.cfg["seed"]
        epoch_offset = (self.current_epoch * 1000) if is_train else 0
        rng = stable_rng_from_id(sample_id, base + epoch_offset)
        organ_u8 = (mask_bin > 0).astype(np.uint8)
        if organ_u8.sum() == 0:
            return None
        pos = sample_positive_points(
            organ_u8, num_points=self.cfg.get("num_pos_per_organ", 1), rng=rng
        )
        if pos is None:
            return None
        neg = sample_negative_points_from_ring(
            organ_u8,
            ring_width=self.cfg.get("ring_width", 20),
            num_neg=self.cfg.get("num_neg_points", 3),
            rng=rng,
        )
        if neg is not None:
            all_points = np.vstack([pos, neg])
            all_labels = np.concatenate(
                [np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)]
            )
        else:
            all_points = pos
            all_labels = np.ones(len(pos), dtype=np.int32)
        return {"points": all_points, "labels": all_labels}

    def _forward_one(self, image_rgb, mask_bin, filename, is_train: bool):
        x, original_size = self._prepare_image(image_rgb)
        gt = self._prepare_mask(mask_bin)
        with torch.no_grad():
            img_emb = self.model.image_encoder(x)
        point_set = self._points_prompt(mask_bin, original_size, str(filename), is_train=is_train)
        if point_set is None:
            return None
        pts_t = self.transform.apply_coords(point_set["points"], original_size)
        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
        lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)

        with torch.no_grad():
            sparse, dense = self.model.prompt_encoder(points=(pts_t, lab_t), boxes=None, masks=None)

        low_res_mask, _ = self.model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = torch.zeros(1, 2, 256, 256, device=self.device, dtype=torch.float32)
        logits[0, 1, :, :] = low_res_mask.squeeze(1).squeeze(0)
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        return logits, gt

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
                organs=[self._eval_organ_label],
            )
        for batch in loader:
            images, masks, filenames = batch["image"], batch["mask"], batch["filename"]
            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_bin = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                if mask_bin.max() == 0:
                    continue
                if train:
                    out = self._forward_one(image_rgb, mask_bin, fn, is_train=train)
                else:
                    with torch.no_grad():
                        out = self._forward_one(image_rgb, mask_bin, fn, is_train=train)
                if out is None:
                    continue
                logits, gt = out
                loss = self.loss_fn(logits, gt)
                d = multi_class_dice_from_logits(logits, gt, self.num_classes)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.mask_decoder.parameters(), 1.0)
                    self.optimizer.step()
                elif val_evaluator is not None:
                    append_single_organ_validation_slice(
                        val_evaluator,
                        pred_logits=logits,
                        gt_binary=gt,
                        organ_label=self._eval_organ_label,
                        image_id=str(fn),
                    )
                total_loss += float(loss.item())
                total_dice += float(d)
                n += 1
        if n == 0:
            if not train and val_evaluator is not None:
                return 0.0, 0.0, organ_balanced_validation_summary(val_evaluator)
            return 0.0, 0.0, None
        avg_loss = total_loss / n
        avg_dice = total_dice / n
        if not train and val_evaluator is not None:
            return avg_loss, avg_dice, organ_balanced_validation_summary(val_evaluator)
        return avg_loss, avg_dice, None

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
            "prompt_type": "points",
            "num_classes": self.num_classes,
            "task": "single_organ",
            "organ": self.organ,
            "source_class_id": self._organ_to_class[self.organ],
            "best_val_msd_tracked": float(self.best_val_msd),
            "val_msd_at_save": float(val_msd) if val_msd is not None else None,
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "model_state": {"mask_decoder": self.model.mask_decoder.state_dict()},
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
        fields = [
            "epoch",
            "train_loss",
            "train_dice",
            "val_loss",
            "val_dice",
            "val_msd",
            "val_hd95",
            "best_val_msd",
            "lr",
        ]
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
        start_epoch = int(getattr(self, "_resume_start_epoch", 0))
        for epoch in range(start_epoch, self.cfg["max_epochs"]):
            self.current_epoch = epoch
            t0 = time.time()
            train_loss, train_dice, _ = self.run_epoch(train_loader, train=True)
            val_loss, val_dice, val_summ = self.run_epoch(val_loader, train=False)
            val_msd = val_summ.get("val_msd") if val_summ else None
            val_hd95 = val_summ.get("val_hd95") if val_summ else None
            val_rep_dice = val_summ.get("val_dice") if val_summ else None
            lr = self.optimizer.param_groups[0]["lr"]
            self.save_checkpoint(run_dir / "last_checkpoint.pth", val_msd=val_msd)
            if val_loss == 0.0 and val_dice == 0.0:
                print("WARNING: val had 0 valid samples this epoch")
            elif val_msd is not None and val_msd < self.best_val_msd - self.val_msd_min_delta:
                self.best_val_msd = val_msd
                self.save_checkpoint(run_dir / "best_checkpoint.pth", val_msd=val_msd)
                self.epochs_without_improvement = 0
                print(f"✓ New best val_msd: {val_msd:.4f} mm")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_msd": val_msd if val_msd is not None else "",
                "val_hd95": val_hd95 if val_hd95 is not None else "",
                "best_val_msd": self.best_val_msd,
                "lr": lr,
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/dice": train_dice,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "lr": lr,
                "time/epoch_s": time.time() - t0,
            }
            if val_msd is not None:
                log_dict["val/msd_mm"] = val_msd
            if val_hd95 is not None:
                log_dict["val/hd95_mm"] = val_hd95
            if val_rep_dice is not None:
                log_dict["val/dice_organ_balanced"] = val_rep_dice
            wandb.log(log_dict)
            msd_s = f"{val_msd:.4f}" if val_msd is not None else "N/A"
            hd_s = f"{val_hd95:.4f}" if val_hd95 is not None else "N/A"
            print(
                f"[{epoch+1}/{self.cfg['max_epochs']}] "
                f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"val loss={val_loss:.4f} dice={val_dice:.4f} msd={msd_s} mm hd95={hd_s} mm"
            )
            self.scheduler.step()

    def evaluate_test_once(self, test_loader):
        run_dir = Path(self.cfg["run_dir"])
        best_path = run_dir / "best_checkpoint.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
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
            "organ": self.organ,
            "task": "single_organ",
        }
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(out, f, indent=2)

        tv = self.cfg.get("test_visualizations") or {}
        if tv.get("enabled"):
            from bapmos.evaluation.baseline_singleorgan_viz import (
                run_singleorgan_baseline_test_visualizations,
            )

            run_singleorgan_baseline_test_visualizations(self, test_loader, run_dir, tv)

        wandb.log(
            {
                "test/loss": test_loss,
                "test/dice": test_dice,
                "test/msd_mm": test_msd,
                "test/hd95_mm": test_hd95,
            }
        )
        print(
            f"[TEST] loss={test_loss:.4f} dice={test_dice:.4f} msd={test_msd} hd95={test_hd95} "
            f"organ_balanced_dice={test_rep_dice}"
        )
        return out


def main():
    root = project_root()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--organ",
        type=str,
        required=True,
        help="Organ key for this dataset taxonomy (binary foreground)",
    )
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--sam_checkpoint", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_pos_per_organ", type=int, default=1)
    p.add_argument("--num_neg_points", type=int, default=3)
    p.add_argument("--ring_width", type=int, default=20)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument(
        "--skip_test_eval",
        action="store_true",
        help="Skip post-train test export (Table I runs infer separately).",
    )
    p.add_argument("--wandb_project", type=str, default="sam-single-organ")
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY env if set; omit entity when unset).",
    )
    p.add_argument(
        "--splits_subdir",
        type=str,
        default="splits_stratified",
        help="Split lists under data_root (default: splits_stratified).",
    )
    p.add_argument(
        "--save-test-visualizations",
        action="store_true",
        help="After testing, save qualitative panels under run_dir/test_results/.",
    )
    p.add_argument(
        "--test-viz-selection",
        choices=["all", "random", "worst_msd", "best_msd"],
        default="all",
    )
    p.add_argument("--test-viz-max", type=int, default=None)
    p.add_argument("--test-viz-seed", type=int, default=42)
    p.add_argument(
        "--run_root",
        type=str,
        default=None,
        help=(
            "Directory containing per-run folders (each with last_checkpoint.pth). "
            "Default: runs/SingleOrgan/Baseline/<organ>_points"
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from last_checkpoint.pth path, or 'auto' with fixed --run_name under --run_root.",
    )
    args = p.parse_args()

    ckpt_obj = None
    resume_arg = (args.resume or "").strip()
    default_run_root = Path("runs") / "SingleOrgan" / "Baseline" / f"{args.organ}_points"
    run_root_base = Path(args.run_root) if args.run_root else default_run_root

    if resume_arg:
        if resume_arg.lower() == "auto":
            if not args.run_name:
                raise ValueError("--resume auto requires --run_name matching the interrupted job.")
            ckpt_fp = run_root_base / args.run_name / "last_checkpoint.pth"
        else:
            ckpt_fp = resolve_under_project(resume_arg)
        if not ckpt_fp.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
        ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
        config = ckpt_obj["config"]
        if config.get("organ") != args.organ:
            raise ValueError(
                f"Checkpoint organ {config.get('organ')!r} does not match --organ {args.organ!r}"
            )
        run_dir = Path(config["run_dir"])
        run_name = config["run_name"]
        data_root = config["data_root"]
        sam_ckpt = config["sam_checkpoint"]
    else:
        data_root = args.data_root or str(resolve_training_data_root("case1"))
        sam_ckpt = args.sam_checkpoint or str(root / "models" / "sam_base" / "sam_vit_b_01ec64.pth")
        sam_resolved = resolve_under_project(sam_ckpt)
        if not sam_resolved.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt!r}")

        seed_everything(args.seed, deterministic=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"{args.organ}_single_points_{ts}"
        run_dir = run_root_base / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "data_root": data_root,
            "sam_checkpoint": str(sam_resolved),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "organ": args.organ,
            "task": "single_organ",
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "num_pos_per_organ": args.num_pos_per_organ,
            "num_neg_points": args.num_neg_points,
            "ring_width": args.ring_width,
            "patience": args.patience,
            "prompt_type": "points",
            "num_classes": 2,
            "splits_subdir": args.splits_subdir,
            "test_visualizations": {
                "enabled": args.save_test_visualizations,
                "selection": args.test_viz_selection,
                "max": args.test_viz_max,
                "seed": args.test_viz_seed,
            },
        }

    if ckpt_obj is not None:
        seed_everything(config["seed"], deterministic=True)

    sam_resolved = resolve_under_project(config["sam_checkpoint"])
    if not sam_resolved.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {config['sam_checkpoint']!r}")
    config["sam_checkpoint"] = str(sam_resolved)

    if resume_arg:
        run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("splits_subdir", "splits_stratified")

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    wandb_kw = {
        "project": args.wandb_project,
        "name": run_name,
        "config": config,
    }
    entity = (args.wandb_entity or os.environ.get("WANDB_ENTITY") or "").strip()
    if entity:
        wandb_kw["entity"] = entity
    wandb.init(**wandb_kw)

    train_ds = SingleOrganDataset(
        data_root, organ=config["organ"], split="train", splits_subdir=config["splits_subdir"]
    )
    val_ds = SingleOrganDataset(
        data_root, organ=config["organ"], split="val", splits_subdir=config["splits_subdir"]
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
        test_ds = SingleOrganDataset(
            data_root, organ=config["organ"], split="test", splits_subdir=config["splits_subdir"]
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )

    trainer = SAMSingleOrganTrainer(config)
    trainer._train_generator_ref = g
    if ckpt_obj is not None:
        trainer.resume_from(ckpt_obj, g)
    trainer.train_loop(train_loader, val_loader)
    if not skip_test_eval and test_loader is not None:
        print("Evaluating test...")
        trainer.evaluate_test_once(test_loader)
    wandb.finish()
    print(f"Done. Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
