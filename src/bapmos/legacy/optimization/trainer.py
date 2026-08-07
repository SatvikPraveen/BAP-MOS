"""
Shared Trainer for Multi-Organ Segmentation Optimization

Supports three prompt strategies:
1. Box vs Point (mutually exclusive)
2. Three-way (box, point, both)
3. Adaptive UCB1 (UCB1 bandit)

Key features:
- MSD-based checkpoint selection (not Dice)
- Boundary-aware metrics during validation
- WandB logging with prompt distributions
- Decoder-only fine-tuning
"""

import os
import sys
import json
import time
import csv
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import random
import pickle
import yaml

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

# Import dataset and prompt utilities
from bapmos.multiorgan.dataset_multi_organ import (
    MultiOrganDataset,
    compute_per_organ_boxes,
    multi_organ_collate_fn,
    sample_per_organ_points_with_negatives,
)
from bapmos.losses.kervadec_style import BoundaryDistMapCache, DecoderLossComputer, DecoderLossConfig
from bapmos.paths import (
    dataset_bundle_tag,
    project_root,
    resolve_model_checkpoint,
    resolve_under_project,
)
from bapmos.training_taxonomy import get_baseline_taxonomy_profile

# Import optimization components
from .prompts import (
    BoxPointSampler, 
    ThreeWaySampler, 
    UCB1Bandit,
    UCB1PerOrganBandit,
    EpsilonGreedyPerOrganBandit,
    EpsilonDecayPerOrganBandit,
    UCBTunedPerOrganBandit
)
from bapmos.baseline_validation_metrics import organ_balanced_validation_summary
from bapmos.checkpoint_selection import (
    checkpoint_scores_from_evaluator,
    is_better_checkpoint,
    parse_checkpoint_objective_config,
)
from .metrics import MetricsEvaluator

LEGACY_METRICS_CSV_FIELDS = [
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
    "train_box_count",
    "train_point_count",
    "train_both_count",
    "val_box_count",
    "val_point_count",
    "val_both_count",
]

# Prompt strategies with independent bandits per organ (see bandit.collapse_per_organ_arms_to_majority).
PER_ORGAN_PROMPT_STRATEGIES = frozenset({
    "ucb1_per_organ",
    "epsilon_greedy_per_organ",
    "epsilon_decay_per_organ",
    "ucb_tuned_per_organ",
    "bap_mos_tuned",
})


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


def setup_logging(log_dir: Path, run_name: str):
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def seed_worker(worker_id):
    """Seed worker for deterministic DataLoader with num_workers>0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stable_rng_from_id(sample_id: str, base_seed: int):
    """Create stable RNG from sample ID."""
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def _resolved_dataset_root(data_root) -> Path:
    p = Path(data_root)
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def jitter_box(box, noise_pixels, rng):
    """Add jitter to bounding box."""
    if noise_pixels == 0 or box is None:
        return box
    
    x_min, y_min, x_max, y_max = box
    jitter = rng.integers(-noise_pixels, noise_pixels + 1, size=4)
    
    x_min_j = max(0, x_min + jitter[0])
    y_min_j = max(0, y_min + jitter[1])
    x_max_j = x_max + jitter[2]
    y_max_j = y_max + jitter[3]
    
    if x_max_j <= x_min_j + 1:
        x_max_j = x_min_j + 2
    if y_max_j <= y_min_j + 1:
        y_max_j = y_min_j + 2
    
    return np.array([x_min_j, y_min_j, x_max_j, y_max_j])


def multi_class_dice_from_logits(logits, gt_multi, num_classes=5):
    """Compute mean dice across foreground classes."""
    probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(probs, dim=1, keepdim=True)
    
    dices = []
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
class OptimizationTrainer:
    """
    Trainer for multi-organ segmentation optimization experiments.
    
    Args:
        config (dict): Configuration dictionary containing:
            - prompt_strategy: "box_point", "three_way", or "adaptive"
            - Strategy-specific params (ratios, weights, bandit config)
            - Training params (lr, epochs, etc.)
    """
    
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
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

        self.use_cached_image_embeddings = bool(config.get("use_cached_image_embeddings", False))
        self._sam_checkpoint_name = Path(config["sam_checkpoint"]).name
        if self.use_cached_image_embeddings:
            self.logger.info(
                "use_cached_image_embeddings=True: skipping SAM image_encoder in forward; "
                "ensure train/val MultiOrganDataset provides embedding_path and caches "
                "were built with the same sam_checkpoint (Meta SAM vs MedSAM are not interchangeable). "
                "Trainer SAM weights file: %s",
                self._sam_checkpoint_name,
            )

        tax = get_baseline_taxonomy_profile(config["data_root"])
        self._taxonomy_name = tax.taxonomy_name
        self._is_simulation = tax.is_simulation
        self._organ_to_class = dict(tax.organ_to_class)
        self._organ_keys = list(tax.organ_keys)
        self._multiclass_eval_mapping = dict(tax.multiclass_eval_mapping)
        self._evaluator_organ_labels = list(tax.evaluator_organ_labels)
        self._organs_for_per_organ_metrics = tax.organ_definitions
        self.num_classes = tax.num_classes
        self.logger.info(
            "OptimizationTrainer: taxonomy=%s | num_classes=%s | evaluator organs=%s",
            tax.taxonomy_name,
            self.num_classes,
            self._evaluator_organ_labels,
        )
        from bapmos.legacy.pfus1_advanced.scale_aware_prompts import (
            is_scale_aware_prompt_geometry,
            prompt_geometry_summary,
        )

        if is_scale_aware_prompt_geometry(config):
            # Letterboxed PFUS1-advanced canvas (768x576) when data_root is pfus1_advanced
            self.logger.info(
                "Prompt geometry: scale-aware | profile=%s | %s",
                config.get("prompt_geometry_profile", "unspecified"),
                prompt_geometry_summary(config, image_hw=(576, 768)),
            )
        else:
            self.logger.info(
                "Prompt geometry: fixed | ring_width=%s",
                config.get("ring_width", 20),
            )

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=config["lr"], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["max_epochs"], eta_min=1e-7
        )
        
        # Initialize prompt sampler
        self.prompt_strategy = config["prompt_strategy"]
        self._init_prompt_sampler()
        
        # Metrics evaluator for validation
        pixel_spacing = config.get("pixel_spacing", None)
        if pixel_spacing is None:
            pixel_spacing = tax.pixel_spacing_mm
        pixel_spacing = tuple(pixel_spacing)
        self.evaluator = MetricsEvaluator(
            pixel_spacing=pixel_spacing,
            organs=self._evaluator_organ_labels,
        )
        dr = _resolved_dataset_root(config["data_root"])
        self.logger.info(
            "Trainer taxonomy check: data_root_resolved=%s | taxonomy=%s | "
            "num_classes=%s | organ_to_class=%s | pixel_spacing_mm=%s | evaluator_organs=%s",
            dr,
            self._taxonomy_name,
            self.num_classes,
            dict(self._organ_to_class),
            tuple(pixel_spacing),
            list(self._evaluator_organ_labels),
        )
        
        # Checkpoint tracking (MSD-based)
        self._checkpoint_objective = parse_checkpoint_objective_config(config)
        self.best_val_msd = float('inf')
        self.patience = config.get("patience", 40)
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        # Fixed-ratio / policy / legacy prompting is outside BAP-MOS method → CE+Dice only.
        from bapmos.losses.loss_policy import force_non_method_ce_dice_loss

        force_non_method_ce_dice_loss(config)
        self.decoder_loss = DecoderLossComputer(
            DecoderLossConfig.from_training_config(
                config, max_epochs=int(config["max_epochs"])
            )
        )
        if self.decoder_loss.mode != "ce_dice":
            raise RuntimeError(
                "Legacy / fixed-ratio prompting requires training.loss_mode='ce_dice' "
                f"(got {self.decoder_loss.mode!r}; Kervadec is BAP-MOS method-only)"
            )
        self._boundary_dist_cache = None
        self.logger.info("Decoder loss mode: %s (non-method / fixed-ratio path)", self.decoder_loss.mode)
        
        # Adaptive learning state (block-based arm selection)
        self.batch_counter = 0
        self._current_block_arm = None  # Arm for current evaluation block (global)
        self._current_block_arms = None  # Arms per organ (for per-organ strategies)
        self._block_prompt_fallback_counts: Dict[Tuple[str, str, str], int] = {}
        self._batches_in_current_block = 0  # Batches trained with current arm
        self._resume_skip_train_batches = 0  # Resume: skip first N train loader batches this epoch
        self._train_dataloader_batches_completed_epoch = 0
        self._checkpoint_run_dir: Optional[Path] = None
        self._train_generator_ref = None  # torch.Generator for DataLoader shuffle resume
        self._resume_start_epoch = 0
        self._resume_mid_epoch_active = False
        self._resume_mid_epoch_start_epoch = -1
        
        # Fixed probe set for deterministic validation (reduces noise)
        # Will be initialized in train_loop() with access to val_loader
        self._validation_probe_indices = None
        self._val_dataset_ref = None  # Reference to validation dataset
        self.collapse_per_organ_arms_to_majority = False
        self.compute_train_boundary_metrics = bool(
            config.get("compute_train_boundary_metrics", True)
        )
        self.run_test_after_train = bool(config.get("run_test_after_train", False))
    
    @staticmethod
    def is_per_organ_prompt_strategy(name: str) -> bool:
        return name in PER_ORGAN_PROMPT_STRATEGIES

    def _is_per_organ_prompt_strategy(self) -> bool:
        return self.is_per_organ_prompt_strategy(self.prompt_strategy)

    def _uses_true_per_organ_prompt_execution(self) -> bool:
        return self._is_per_organ_prompt_strategy() and not self.collapse_per_organ_arms_to_majority

    def _init_prompt_sampler(self):
        """Initialize prompt sampler based on strategy."""
        if self.prompt_strategy == "box_point":
            # Phase 1: Mutually exclusive
            self.sampler = BoxPointSampler(
                box_ratio=self.cfg["box_ratio"],
                point_ratio=self.cfg["point_ratio"],
                seed=self.cfg["seed"]
            )
            self.logger.info(f"Prompt strategy: Box vs Point ({self.cfg['box_ratio']:.1%} box, {self.cfg['point_ratio']:.1%} point)")
        
        elif self.prompt_strategy == "three_way":
            # Phase 2: Three-mode categorical
            self.sampler = ThreeWaySampler(
                box_weight=self.cfg["box_weight"],
                point_weight=self.cfg["point_weight"],
                both_weight=self.cfg["both_weight"],
                seed=self.cfg["seed"]
            )
            self.logger.info(f"Prompt strategy: Three-way (weights: {self.cfg['box_weight']}:{self.cfg['point_weight']}:{self.cfg['both_weight']})")
        
        elif self.prompt_strategy == "adaptive":
            # Phase 3: UCB1 bandit
            bandit_cfg = self.cfg["bandit"]
            warmup_blocks_per_arm = bandit_cfg.get("warmup_blocks_per_arm", 10)
            self.block_size_batches = bandit_cfg.get("block_size_batches", 50)
            self.probe_size = bandit_cfg.get("probe_size", 10)
            self.probe_seed = bandit_cfg.get("probe_seed", self.cfg["seed"])  # Fallback to experiment seed
            self.reward_aggregation = bandit_cfg.get("reward_aggregation", "organ_balanced")
            
            self.sampler = UCB1Bandit(
                arms=bandit_cfg.get("arms", ["box", "point", "both"]),
                exploration_constant=bandit_cfg.get("exploration_constant", 2.0),
                warmup_blocks=warmup_blocks_per_arm,
                min_pulls_per_arm=bandit_cfg.get("min_pulls_per_arm", 5),
                reward_clip_max=bandit_cfg.get("reward_clip_max_msd_mm", 20.0),
                seed=self.cfg["seed"]
            )
            total_warmup_batches = warmup_blocks_per_arm * len(self.sampler.arms) * self.block_size_batches
            self.logger.info(f"Prompt strategy: Adaptive (UCB1, c={bandit_cfg.get('exploration_constant', 2.0)})")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); not 'number of outer batches')"
            )
            self.logger.info(
                f"  Warmup scale: {warmup_blocks_per_arm} blocks/arm × {len(self.sampler.arms)} arms "
                f"× threshold {self.block_size_batches} = {total_warmup_batches} (same units as threshold)"
            )
            self.logger.info(f"  Min pulls/arm: {bandit_cfg.get('min_pulls_per_arm', 5)}, Reward clip: {bandit_cfg.get('reward_clip_max_msd_mm', 20.0)} mm")
            self.logger.info(f"  Probe: {self.probe_size} images (seed={self.probe_seed}), Reward: {self.reward_aggregation}")
        
        elif self.prompt_strategy == "ucb1_per_organ":
            # Per-organ UCB1 bandit
            bandit_cfg = self.cfg["bandit"]
            warmup_blocks_per_arm = bandit_cfg.get("warmup_blocks_per_arm", 10)
            self.block_size_batches = bandit_cfg.get("block_size_batches", 50)
            self.probe_size = bandit_cfg.get("probe_size", 10)
            self.probe_seed = bandit_cfg.get("probe_seed", self.cfg["seed"])
            
            self.organs = bandit_cfg.get("organs", list(self._organ_keys))
            
            self.sampler = UCB1PerOrganBandit(
                organs=self.organs,
                arms=bandit_cfg.get("arms", ["box", "point", "both"]),
                exploration_constant=bandit_cfg.get("exploration_constant", 2.0),
                warmup_blocks=warmup_blocks_per_arm,
                min_pulls_per_arm=bandit_cfg.get("min_pulls_per_arm", 5),
                reward_clip_max=bandit_cfg.get("reward_clip_max_msd_mm", 20.0),
                seed=self.cfg["seed"]
            )
            total_warmup_batches = warmup_blocks_per_arm * len(self.sampler.arms) * self.block_size_batches
            self.logger.info(f"Prompt strategy: Per-Organ UCB1 (organs={self.organs})")
            self.logger.info(f"  Exploration constant: {bandit_cfg.get('exploration_constant', 2.0)}")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); not 'number of outer batches')"
            )
            self.logger.info(
                f"  Warmup scale: {warmup_blocks_per_arm} blocks/arm × {len(self.sampler.arms)} arms "
                f"× threshold {self.block_size_batches} = {total_warmup_batches} (same units as threshold)"
            )
            self.logger.info(f"  Probe: {self.probe_size} images (seed={self.probe_seed})")
        
        elif self.prompt_strategy == "epsilon_greedy_per_organ":
            # Per-organ epsilon-greedy bandit
            bandit_cfg = self.cfg["bandit"]
            warmup_blocks_per_arm = bandit_cfg.get("warmup_blocks_per_arm", 10)
            self.block_size_batches = bandit_cfg.get("block_size_batches", 50)
            self.probe_size = bandit_cfg.get("probe_size", 10)
            self.probe_seed = bandit_cfg.get("probe_seed", self.cfg["seed"])
            
            self.organs = bandit_cfg.get("organs", list(self._organ_keys))
            
            self.sampler = EpsilonGreedyPerOrganBandit(
                organs=self.organs,
                arms=bandit_cfg.get("arms", ["box", "point", "both"]),
                epsilon=bandit_cfg.get("epsilon", 0.1),
                epsilon_decay=bandit_cfg.get("epsilon_decay", 1.0),
                epsilon_min=bandit_cfg.get("epsilon_min", 0.01),
                warmup_blocks=warmup_blocks_per_arm,
                min_pulls_per_arm=bandit_cfg.get("min_pulls_per_arm", 5),
                reward_clip_max=bandit_cfg.get("reward_clip_max_msd_mm", 20.0),
                seed=self.cfg["seed"]
            )
            total_warmup_batches = warmup_blocks_per_arm * len(self.sampler.arms) * self.block_size_batches
            self.logger.info(f"Prompt strategy: Per-Organ Epsilon-Greedy (organs={self.organs})")
            self.logger.info(f"  Epsilon: {bandit_cfg.get('epsilon', 0.1)}, Decay: {bandit_cfg.get('epsilon_decay', 1.0)}, Min: {bandit_cfg.get('epsilon_min', 0.01)}")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); not 'number of outer batches')"
            )
            self.logger.info(
                f"  Warmup scale: {warmup_blocks_per_arm} blocks/arm × {len(self.sampler.arms)} arms "
                f"× threshold {self.block_size_batches} = {total_warmup_batches} (same units as threshold)"
            )
            self.logger.info(f"  Probe: {self.probe_size} images (seed={self.probe_seed})")
        
        elif self.prompt_strategy == "epsilon_decay_per_organ":
            # Per-organ epsilon-decay bandit
            bandit_cfg = self.cfg["bandit"]
            warmup_blocks_per_arm = bandit_cfg.get("warmup_blocks_per_arm", 10)
            self.block_size_batches = bandit_cfg.get("block_size_batches", 50)
            self.probe_size = bandit_cfg.get("probe_size", 10)
            self.probe_seed = bandit_cfg.get("probe_seed", self.cfg["seed"])
            
            self.organs = bandit_cfg.get("organs", list(self._organ_keys))
            
            self.sampler = EpsilonDecayPerOrganBandit(
                organs=self.organs,
                arms=bandit_cfg.get("arms", ["box", "point", "both"]),
                epsilon_start=bandit_cfg.get("epsilon_start", 0.3),
                epsilon_end=bandit_cfg.get("epsilon_end", 0.01),
                decay_schedule=bandit_cfg.get("decay_schedule", "exponential"),
                decay_steps=bandit_cfg.get("decay_steps", 1000),
                decay_rate=bandit_cfg.get("decay_rate"),
                step_size=bandit_cfg.get("step_size", 100),
                warmup_blocks=warmup_blocks_per_arm,
                min_pulls_per_arm=bandit_cfg.get("min_pulls_per_arm", 5),
                reward_clip_max=bandit_cfg.get("reward_clip_max_msd_mm", 20.0),
                seed=self.cfg["seed"]
            )
            total_warmup_batches = warmup_blocks_per_arm * len(self.sampler.arms) * self.block_size_batches
            self.logger.info(f"Prompt strategy: Per-Organ Epsilon-Decay (organs={self.organs})")
            self.logger.info(f"  Epsilon: {bandit_cfg.get('epsilon_start', 0.3)} → {bandit_cfg.get('epsilon_end', 0.01)}")
            self.logger.info(f"  Decay: {bandit_cfg.get('decay_schedule', 'exponential')}, Steps: {bandit_cfg.get('decay_steps', 1000)}")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); not 'number of outer batches')"
            )
            self.logger.info(
                f"  Warmup scale: {warmup_blocks_per_arm} blocks/arm × {len(self.sampler.arms)} arms "
                f"× threshold {self.block_size_batches} = {total_warmup_batches} (same units as threshold)"
            )
            self.logger.info(f"  Probe: {self.probe_size} images (seed={self.probe_seed})")
        
        elif self.prompt_strategy in ("ucb_tuned_per_organ", "bap_mos_tuned"):
            # Per-organ UCB-Tuned bandit (variance-aware); bap_mos_tuned is the paper-facing name.
            bandit_cfg = self.cfg["bandit"]
            warmup_blocks_per_arm = bandit_cfg.get("warmup_blocks_per_arm", 10)
            self.block_size_batches = bandit_cfg.get("block_size_batches", 50)
            self.probe_size = bandit_cfg.get("probe_size", 10)
            self.probe_seed = bandit_cfg.get("probe_seed", self.cfg["seed"])
            
            self.organs = bandit_cfg.get("organs", list(self._organ_keys))
            
            self.sampler = UCBTunedPerOrganBandit(
                organs=self.organs,
                arms=bandit_cfg.get("arms", ["box", "point", "both"]),
                warmup_blocks=warmup_blocks_per_arm,
                min_pulls_per_arm=bandit_cfg.get("min_pulls_per_arm", 5),
                reward_clip_max=bandit_cfg.get("reward_clip_max_msd_mm", 20.0),
                seed=self.cfg["seed"]
            )
            total_warmup_batches = warmup_blocks_per_arm * len(self.sampler.arms) * self.block_size_batches
            label = (
                "BAP-MOS-Tuned (true per-organ UCB-Tuned)"
                if self.prompt_strategy == "bap_mos_tuned"
                else "Per-Organ UCB-Tuned"
            )
            self.logger.info(f"Prompt strategy: {label} (organs={self.organs})")
            self.logger.info("  Variance-aware confidence bounds (automatic exploration adjustment)")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); not 'number of outer batches')"
            )
            self.logger.info(
                f"  Warmup scale: {warmup_blocks_per_arm} blocks/arm × {len(self.sampler.arms)} arms "
                f"× threshold {self.block_size_batches} = {total_warmup_batches} (same units as threshold)"
            )
            self.logger.info(f"  Probe: {self.probe_size} images (seed={self.probe_seed})")
        
        else:
            raise ValueError(f"Unknown prompt strategy: {self.prompt_strategy}")

        if self._is_per_organ_prompt_strategy():
            bandit_cfg = self.cfg.get("bandit", {})
            self.collapse_per_organ_arms_to_majority = bool(
                bandit_cfg.get("collapse_per_organ_arms_to_majority", False)
            )
            if self.collapse_per_organ_arms_to_majority:
                self.logger.info(
                    "  Per-organ training: majority-vote collapse (ablation; not organ-specific execution)"
                )
            else:
                self.logger.info(
                    "  Per-organ training: each organ uses its own selected prompt arm during forward"
                )
    
    def _ensure_per_organ_block_arms(self) -> Dict[str, str]:
        """Select and cache one prompt arm per organ for the current bandit block."""
        if self._current_block_arms is None:
            self._current_block_arms = {
                organ: self.sampler.select_arm(organ) for organ in self.organs
            }
            self._block_prompt_fallback_counts = {}
            msg = f"[Per-Organ Block] Arms selected: {self._current_block_arms}"
            if self.collapse_per_organ_arms_to_majority:
                majority = self._majority_prompt_mode(self._current_block_arms)
                msg += f" | Majority execution mode: '{majority}'"
            self.logger.info(msg)
        return dict(self._current_block_arms)

    def _get_prompt_modes_by_organ(self) -> Dict[str, str]:
        """Per-organ prompt arms for the current block (true per-organ execution)."""
        return self._ensure_per_organ_block_arms()

    def _majority_prompt_mode(self, organ_arms: Dict[str, str]) -> str:
        from collections import Counter
        return Counter(organ_arms.values()).most_common(1)[0][0]

    def _get_prompt_mode(self, sample_id: str) -> str:
        """Global prompt mode for one slice (box / point / both).

        Per-organ strategies with ``collapse_per_organ_arms_to_majority`` use the
        plurality of per-organ arms. True per-organ execution should call
        ``_forward_one_true_per_organ`` instead.
        """
        if self.prompt_strategy == "box_point":
            return self.sampler.sample_prompt_type(sample_id, self.current_epoch)
        
        elif self.prompt_strategy == "three_way":
            return self.sampler.sample_prompt_mode(sample_id, self.current_epoch)
        
        elif self.prompt_strategy == "adaptive":
            if self._current_block_arm is None:
                self._current_block_arm = self.sampler.select_arm()
            return self._current_block_arm
        
        elif self._is_per_organ_prompt_strategy():
            organ_arms = self._ensure_per_organ_block_arms()
            if self.collapse_per_organ_arms_to_majority:
                return self._majority_prompt_mode(organ_arms)
            raise RuntimeError(
                "_get_prompt_mode() must not be used for true per-organ execution "
                "(collapse_per_organ_arms_to_majority=False). Use _get_prompt_modes_by_organ() "
                "or _forward_one_true_per_organ() instead."
            )
        
        else:
            raise ValueError(f"Unknown prompt strategy: {self.prompt_strategy}")

    def _record_prompt_fallback(self, organ: str, selected: str, effective: str) -> None:
        """Count selected→effective prompt fallbacks within the current bandit block."""
        if selected == effective:
            return
        key = (organ, selected, effective)
        self._block_prompt_fallback_counts[key] = (
            self._block_prompt_fallback_counts.get(key, 0) + 1
        )

    def _log_block_prompt_fallbacks(self) -> Dict[str, float]:
        """Log and return WandB scalars for prompt fallbacks in the completed block."""
        if not self._block_prompt_fallback_counts:
            return {}
        total = sum(self._block_prompt_fallback_counts.values())
        self.logger.info(
            f"[Per-Organ Prompt Fallbacks] {total} organ-decodes used a different mode than selected"
        )
        log_dict: Dict[str, float] = {"bandit/prompt_fallback_total": float(total)}
        for (organ, selected, effective), count in sorted(
            self._block_prompt_fallback_counts.items()
        ):
            self.logger.info(
                f"  {organ}: selected={selected} → effective={effective} (n={count})"
            )
            organ_key = organ.lower()
            log_dict[f"bandit/{organ_key}_fallback_{selected}_to_{effective}"] = float(count)
        return log_dict

    @staticmethod
    def _point_set_is_valid(point_set: Optional[dict]) -> bool:
        if point_set is None:
            return False
        labels = point_set.get("labels", [])
        if len(point_set.get("points", [])) == 0:
            return False
        try:
            return int(np.sum(np.asarray(labels) == 1)) > 0
        except Exception:
            return False

    def _build_organ_prompt_lookups(
        self,
        point_result: Optional[Tuple],
        box_result: Optional[Tuple],
    ) -> Tuple[Dict[str, dict], Dict[str, torch.Tensor]]:
        points_by_organ: Dict[str, dict] = {}
        boxes_by_organ: Dict[str, torch.Tensor] = {}
        if point_result is not None:
            point_sets, point_organs = point_result
            for i, organ_name in enumerate(point_organs):
                points_by_organ[organ_name] = point_sets[i]
        if box_result is not None:
            boxes_t, box_organs = box_result
            for i, organ_name in enumerate(box_organs):
                boxes_by_organ[organ_name] = boxes_t[i : i + 1, :]
        return points_by_organ, boxes_by_organ

    def _decode_one_organ(
        self,
        img_emb: torch.Tensor,
        original_size: Tuple[int, int],
        organ_name: str,
        mode: str,
        points_by_organ: Dict[str, dict],
        boxes_by_organ: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Decode a single organ mask under ``mode`` (with per-organ fallbacks)."""
        box_t = boxes_by_organ.get(organ_name)
        point_set = points_by_organ.get(organ_name)
        effective_mode = mode

        if mode == "box":
            if box_t is None:
                return None, effective_mode
            sparse, dense = self._encode_box_prompt(box_t)
        elif mode == "point":
            if not self._point_set_is_valid(point_set):
                if box_t is None:
                    return None, effective_mode
                effective_mode = "box"
                sparse, dense = self._encode_box_prompt(box_t)
            else:
                sparse, dense = self._encode_point_prompt(point_set, original_size)
        elif mode == "both":
            if (
                box_t is not None
                and self._point_set_is_valid(point_set)
            ):
                sparse, dense = self._encode_both_prompts(point_set, box_t, original_size)
            elif box_t is not None:
                effective_mode = "box"
                sparse, dense = self._encode_box_prompt(box_t)
            elif self._point_set_is_valid(point_set):
                effective_mode = "point"
                sparse, dense = self._encode_point_prompt(point_set, original_size)
            else:
                return None, effective_mode
        else:
            raise ValueError(f"Unknown prompt mode: {mode}")

        low_res_mask, _ = self.model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return low_res_mask.squeeze(1), effective_mode

    def _encode_box_prompt(self, box_t: torch.Tensor):
        with torch.no_grad():
            return self.model.prompt_encoder(
                points=None,
                boxes=box_t.unsqueeze(1),
                masks=None,
            )

    def _encode_point_prompt(self, point_set: dict, original_size: Tuple[int, int]):
        pts_t = self.transform.apply_coords(point_set["points"], original_size)
        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
        lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            return self.model.prompt_encoder(
                points=(pts_t, lab_t),
                boxes=None,
                masks=None,
            )

    def _encode_both_prompts(self, point_set: dict, box_t: torch.Tensor, original_size: Tuple[int, int]):
        pts_t = self.transform.apply_coords(point_set["points"], original_size)
        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
        lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            return self.model.prompt_encoder(
                points=(pts_t, lab_t),
                boxes=box_t.unsqueeze(1),
                masks=None,
            )

    @staticmethod
    def _record_prompt_usage(
        prompt_mode_counts: Dict[str, int],
        prompt_meta: Union[str, Dict[str, str]],
    ) -> None:
        if isinstance(prompt_meta, dict):
            for mode in prompt_meta.values():
                if mode in prompt_mode_counts:
                    prompt_mode_counts[mode] += 1
        elif prompt_meta in prompt_mode_counts:
            prompt_mode_counts[prompt_meta] += 1

    def _forward_one_true_per_organ(
        self,
        image_rgb,
        mask_multi,
        filename,
        is_train: bool,
        embedding_path: Optional[str] = None,
        cached_embedding_pack: Optional[dict] = None,
    ):
        """Forward pass: each organ uses its own bandit-selected prompt mode."""
        from bapmos.multiorgan.dataset_multi_organ import has_any_organ

        if not has_any_organ(mask_multi):
            return None

        gt = self._prepare_mask(mask_multi)
        img_emb, original_size = self._prepare_image_embedding(
            image_rgb, filename, embedding_path, cached_embedding_pack
        )

        organ_modes = self._get_prompt_modes_by_organ()
        point_result = self._points_prompt(mask_multi, original_size, filename, is_train)
        box_result = self._box_prompt(mask_multi, original_size, filename)
        points_by_organ, boxes_by_organ = self._build_organ_prompt_lookups(point_result, box_result)

        present_organs = [
            o for o in self._organ_keys
            if o in points_by_organ or o in boxes_by_organ
        ]
        if len(present_organs) == 0:
            return None

        organ_masks = []
        executed_organs = []
        executed_modes: Dict[str, str] = {}

        for organ_name in present_organs:
            mode = organ_modes.get(organ_name, "box")
            mask, effective_mode = self._decode_one_organ(
                img_emb,
                original_size,
                organ_name,
                mode,
                points_by_organ,
                boxes_by_organ,
            )
            if mask is None:
                continue
            organ_masks.append(mask)
            executed_organs.append(organ_name)
            executed_modes[organ_name] = effective_mode
            if effective_mode != mode:
                self._record_prompt_fallback(organ_name, mode, effective_mode)

        if len(organ_masks) == 0:
            return None

        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        logits = torch.zeros(1, self.num_classes, 256, 256, device=self.device, dtype=torch.float32)
        for idx, organ_name in enumerate(executed_organs):
            class_id = self._organ_to_class[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]

        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)

        return logits, gt, executed_modes

    def _prepare_image_embedding(
        self,
        image_rgb,
        filename,
        embedding_path: Optional[str],
        cached_embedding_pack: Optional[dict],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.use_cached_image_embeddings:
            if cached_embedding_pack is not None:
                img_emb, original_size = self._tensor_from_cached_embedding_pack(
                    cached_embedding_pack,
                    str(filename),
                )
            elif embedding_path:
                img_emb, original_size = self._load_cached_sam_embedding(embedding_path)
            else:
                raise RuntimeError(
                    "use_cached_image_embeddings is True but neither cached_embedding_pack nor "
                    "embedding_path was provided."
                )
            img_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
            if original_size != img_hw:
                self.logger.warning(
                    "Cached original_size %s does not match loaded image shape %s for %s",
                    original_size,
                    img_hw,
                    filename,
                )
        else:
            x, original_size = self._prepare_image(image_rgb)
            with torch.no_grad():
                img_emb = self.model.image_encoder(x)
        return img_emb, original_size
    
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
    
    def _prepare_image(self, image_rgb):
        """Prepare image for SAM."""
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
        x = x.unsqueeze(0)
        return x, original_size
    
    def _prepare_mask(self, mask_multi):
        """Resize multi-class mask to 256x256."""
        mask_resized = cv2.resize(
            mask_multi.astype(np.float32),
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        gt = torch.as_tensor(mask_resized, device=self.device, dtype=torch.float32)
        gt = gt.unsqueeze(0).unsqueeze(0)
        return gt
    
    def _points_prompt(self, mask_multi, original_size, sample_id, is_train: bool):
        """Generate per-organ point prompts."""
        base = self.cfg["seed"]
        epoch_offset = (self.current_epoch * 1000) if is_train else 0
        rng = stable_rng_from_id(sample_id, base + epoch_offset)
        
        from bapmos.legacy.pfus1_advanced.scale_aware_prompts import (
            is_scale_aware_prompt_geometry,
            sample_per_organ_points_with_scale_aware_geometry,
        )

        if is_scale_aware_prompt_geometry(self.cfg):
            point_sets = sample_per_organ_points_with_scale_aware_geometry(
                mask_multi,
                self.cfg,
                num_pos_per_organ=self.cfg["num_pos_per_organ"],
                num_neg=self.cfg.get("num_neg_points", 3),
                rng=rng,
                organ_to_class=self._organ_to_class,
            )
        else:
            point_sets = sample_per_organ_points_with_negatives(
                mask_multi,
                num_pos_per_organ=self.cfg["num_pos_per_organ"],
                num_neg=self.cfg.get("num_neg_points", 3),
                ring_width=self.cfg.get("ring_width", 20),
                rng=rng,
                organ_to_class=self._organ_to_class,
            )
        
        valid_point_sets = []
        present_organs = []

        for organ_name in self._organ_keys:
            if point_sets[organ_name] is not None:
                valid_point_sets.append(point_sets[organ_name])
                present_organs.append(organ_name)
        
        if len(valid_point_sets) == 0:
            return None
        
        return valid_point_sets, present_organs
    
    def _box_prompt(self, mask_multi, original_size, sample_id):
        """Generate per-organ box prompts."""
        organ_boxes_dict = compute_per_organ_boxes(mask_multi, organ_to_class=self._organ_to_class)
        
        rng = stable_rng_from_id(str(sample_id), self.cfg["seed"] + self.current_epoch * 1000)
        
        boxes_list = []
        present_organs = []
        image_hw = (int(mask_multi.shape[0]), int(mask_multi.shape[1]))
        from bapmos.legacy.pfus1_advanced.scale_aware_prompts import apply_box_margin

        for organ_name in self._organ_keys:
            box = organ_boxes_dict[organ_name]
            if box is not None:
                box = apply_box_margin(box, self.cfg, image_hw=image_hw)
                box = jitter_box(box, self.cfg.get("box_noise_pixels", 0), rng)
                boxes_list.append(box)
                present_organs.append(organ_name)
        
        if len(boxes_list) == 0:
            return None
        
        boxes_array = np.array(boxes_list)
        boxes_t = self.transform.apply_boxes(boxes_array, original_size)
        boxes_t = torch.as_tensor(boxes_t, device=self.device, dtype=torch.float32)
        
        return boxes_t, present_organs

    def _load_cached_sam_embedding(self, embedding_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load ``image_embedding`` and ``original_size`` from a precompute script ``.pt`` file."""
        path = Path(embedding_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cached SAM embedding not found: {path}")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        return self._tensor_from_cached_embedding_pack(pack, str(path))

    def _tensor_from_cached_embedding_pack(
        self,
        pack: dict,
        source: str,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Validate cached pack and move embedding to ``self.device`` (float32)."""
        if "image_embedding" not in pack or "original_size" not in pack:
            raise KeyError(f"Invalid embedding pack (missing keys): {source}")
        cached_ckpt = pack.get("checkpoint")
        if cached_ckpt and cached_ckpt != self._sam_checkpoint_name:
            raise ValueError(
                f"Cached embedding checkpoint mismatch for {source}: "
                f"cache reports {cached_ckpt!r}, trainer uses {self._sam_checkpoint_name!r}. "
                "Recompute embeddings with the same SAM checkpoint as training (do not mix Meta SAM vs MedSAM caches)."
            )
        emb = pack["image_embedding"]
        if emb.dtype in (torch.float16, torch.bfloat16):
            emb = emb.float()
        emb = emb.to(self.device)
        orig = pack["original_size"]
        original_size = (int(orig[0]), int(orig[1]))
        return emb, original_size
    
    def _forward_one(
        self,
        image_rgb,
        mask_multi,
        filename,
        is_train: bool,
        embedding_path: Optional[str] = None,
        cached_embedding_pack: Optional[dict] = None,
    ):
        """Forward pass with adaptive prompt selection."""
        if self._uses_true_per_organ_prompt_execution():
            return self._forward_one_true_per_organ(
                image_rgb,
                mask_multi,
                filename,
                is_train,
                embedding_path=embedding_path,
                cached_embedding_pack=cached_embedding_pack,
            )

        from bapmos.multiorgan.dataset_multi_organ import has_any_organ
        
        # GUARD 1: Check if any organ present
        if not has_any_organ(mask_multi):
            return None
        
        gt = self._prepare_mask(mask_multi)
        img_emb, original_size = self._prepare_image_embedding(
            image_rgb, filename, embedding_path, cached_embedding_pack
        )
        
        # Determine prompt mode
        prompt_mode = self._get_prompt_mode(str(filename))
        
        # GUARD 2: Validate point-based modes have organs
        if prompt_mode in ["point", "both"] and not has_any_organ(mask_multi):
            # Fallback to box mode
            self.logger.warning(f"No organs in {filename}, falling back from {prompt_mode} to box")
            prompt_mode = "box"
        
        # Generate prompts based on mode with fallback logic
        if prompt_mode == "box":
            prompt_result = self._box_prompt(mask_multi, original_size, filename)
            use_points = False
            use_boxes = True
        
        elif prompt_mode == "point":
            prompt_result = self._points_prompt(mask_multi, original_size, filename, is_train)
            use_points = True
            use_boxes = False
            
            # GUARD 3: Validate point prompts are not empty
            if prompt_result is not None:
                point_sets, organs = prompt_result
                if len(point_sets) == 0 or any(
                    pts is None or 
                    len(pts.get('points', [])) == 0 or 
                    (pts.get('labels', []) == 1).sum() == 0 
                    for pts in point_sets
                ):
                    self.logger.debug(f"Invalid point prompts for {filename}, falling back to box")
                    prompt_result = self._box_prompt(mask_multi, original_size, filename)
                    use_points = False
                    use_boxes = True
                    prompt_mode = "box"  # Update mode for tracking
        
        elif prompt_mode == "both":
            # Use both box and point prompts simultaneously
            point_result = self._points_prompt(mask_multi, original_size, filename, is_train)
            box_result = self._box_prompt(mask_multi, original_size, filename)
            
            # GUARD 4: Fallback if either prompt type fails
            if point_result is None and box_result is not None:
                # Points failed, use box only
                self.logger.debug(f"Point prompts failed for {filename}, using box only")
                prompt_result = box_result
                use_points = False
                use_boxes = True
                prompt_mode = "box"
            elif box_result is None and point_result is not None:
                # Box failed, use points only
                self.logger.debug(f"Box prompts failed for {filename}, using points only")
                prompt_result = point_result
                use_points = True
                use_boxes = False
                prompt_mode = "point"
            elif point_result is None or box_result is None:
                # Both failed
                return None
            else:
                # Both succeeded
                # Validate point prompts
                point_sets, point_organs = point_result
                if len(point_sets) == 0 or any(
                    pts is None or 
                    len(pts.get('points', [])) == 0 or 
                    (pts.get('labels', []) == 1).sum() == 0 
                    for pts in point_sets
                ):
                    # Points invalid, fallback to box
                    self.logger.debug(f"Invalid point prompts in both mode for {filename}, using box only")
                    prompt_result = box_result
                    use_points = False
                    use_boxes = True
                    prompt_mode = "box"
                else:
                    prompt_result = (point_result, box_result)
                    use_points = True
                    use_boxes = True
        
        else:
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")
        
        if prompt_result is None:
            return None
        
        # Process each organ
        organ_masks = []
        
        if prompt_mode == "both":
            point_sets, point_organs = prompt_result[0]
            boxes_t, box_organs = prompt_result[1]
            
            # Ensure same organs present in both
            present_organs = [o for o in point_organs if o in box_organs]
            if len(present_organs) == 0:
                return None
            
            num_organs = len(present_organs)
            
            for i, organ_name in enumerate(present_organs):
                # Get point prompt
                point_idx = point_organs.index(organ_name)
                point_set = point_sets[point_idx]
                pts_t = self.transform.apply_coords(point_set['points'], original_size)
                pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                lab_t = torch.as_tensor(point_set['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
                
                # Get box prompt
                box_idx = box_organs.index(organ_name)
                box_i = boxes_t[box_idx:box_idx+1, :]
                
                # Encode both prompts
                with torch.no_grad():
                    sparse, dense = self.model.prompt_encoder(
                        points=(pts_t, lab_t),
                        boxes=box_i.unsqueeze(1),
                        masks=None
                    )
                
                # Decode
                low_res_mask, _ = self.model.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                organ_masks.append(low_res_mask.squeeze(1))
            
        else:
            # Single prompt type
            if use_points:
                point_sets, present_organs = prompt_result
                num_organs = len(point_sets)
                
                for i in range(num_organs):
                    point_set = point_sets[i]
                    pts_t = self.transform.apply_coords(point_set['points'], original_size)
                    pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                    lab_t = torch.as_tensor(point_set['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
                    
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=(pts_t, lab_t),
                            boxes=None,
                            masks=None
                        )
                    
                    low_res_mask, _ = self.model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                    )
                    organ_masks.append(low_res_mask.squeeze(1))
            
            elif use_boxes:
                boxes_t, present_organs = prompt_result
                num_organs = len(present_organs)
                
                for i in range(num_organs):
                    box_i = boxes_t[i:i+1, :]
                    
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=None,
                            boxes=box_i.unsqueeze(1),
                            masks=None
                        )
                    
                    low_res_mask, _ = self.model.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                    )
                    organ_masks.append(low_res_mask.squeeze(1))
        
        # Stack organ masks
        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        
        # Create 4-channel logits
        logits = torch.zeros(1, self.num_classes, 256, 256, device=self.device, dtype=torch.float32)
        
        for idx, organ_name in enumerate(present_organs):
            class_id = self._organ_to_class[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]
        
        # Background channel
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        
        return logits, gt, prompt_mode

    def _validation_checkpoint_scores(self):
        if not self.evaluator.per_slice_metrics:
            return None
        return checkpoint_scores_from_evaluator(
            self.evaluator,
            self._checkpoint_objective,
            self._evaluator_organ_labels,
        )

    def _summarize_evaluator_metrics(self) -> Optional[Dict[str, Optional[float]]]:
        """Organ-balanced + slice-pooled summaries from the current evaluator state."""
        if not self.evaluator.per_slice_metrics:
            return None
        organ_bal = organ_balanced_validation_summary(self.evaluator)
        overall = self.evaluator.aggregate_metrics(organ_name=None) or {}
        return {
            "msd_mm": overall.get("msd_mm_mean"),
            "hd95_mm": overall.get("hd95_mm_mean"),
            "msd_mm_organ_balanced": organ_bal.get("val_msd"),
            "hd95_mm_organ_balanced": organ_bal.get("val_hd95"),
            "dice_organ_balanced": organ_bal.get("val_dice"),
        }

    def _per_organ_wandb_dict(self, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for organ in self._evaluator_organ_labels:
            agg = self.evaluator.aggregate_metrics(organ_name=organ)
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

    @torch.no_grad()
    def _collect_boundary_metrics_from_loader(self, loader) -> Dict[str, Optional[float]]:
        """Full-split boundary metrics (no backward); forward in eval prompt mode."""
        was_training = self.model.training
        self.model.eval()
        self.evaluator.reset()
        n = 0

        for batch in loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]
            emb_paths = batch.get("embedding_path")
            cached_packs = batch.get("sam_cached_pack")

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                if mask_multi.max() == 0:
                    continue
                ep = emb_paths[i] if emb_paths is not None else None
                cp = cached_packs[i] if cached_packs is not None else None
                out = self._forward_one(
                    image_rgb,
                    mask_multi,
                    fn,
                    is_train=False,
                    embedding_path=ep,
                    cached_embedding_pack=cp,
                )
                if out is None:
                    continue
                logits, gt, _ = out
                pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
                gt_classes = gt.cpu().numpy()[0, 0]
                self.evaluator.evaluate_multiclass_slice(
                    pred_classes.astype(np.uint8),
                    gt_classes.astype(np.uint8),
                    slice_idx=n,
                    image_id=fn,
                    class_mapping=self._multiclass_eval_mapping,
                )
                n += 1

        if was_training:
            self.model.train()
        summary = self._summarize_evaluator_metrics()
        return summary if summary is not None else {}

    def export_test_split_metrics(
        self,
        test_loader,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate held-out test split with best checkpoint; write JSON + optional CSVs."""
        run_dir = self._checkpoint_run_dir or Path(self.cfg["run_dir"])
        best_path = run_dir / "best_checkpoint.pth"
        if not best_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {best_path}")

        try:
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=self.device)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.current_epoch = int(ckpt.get("epoch_index", 0))

        test_loss, test_dice, _, val_metrics, _ = self.run_epoch(test_loader, train=False)
        metrics = val_metrics or {}

        if output_dir is not None:
            output_dir = Path(output_dir)
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            self.evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
            self.evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
            self.evaluator.export_failure_analysis_csv(
                metrics_dir / "failure_analysis.csv", top_n=20
            )
            import pandas as pd

            rows = []
            for organ in self._evaluator_organ_labels:
                agg = self.evaluator.aggregate_metrics(organ_name=organ)
                if agg is not None:
                    rows.append(agg)
            if rows:
                pd.DataFrame(rows).to_csv(metrics_dir / "per_organ_metrics.csv", index=False)

        out: Dict[str, Any] = {
            "test_loss": float(test_loss),
            "test_dice": float(test_dice),
            "test_msd_mm": metrics.get("msd_mm_organ_balanced") or metrics.get("msd_mm"),
            "test_hd95_mm": metrics.get("hd95_mm_organ_balanced") or metrics.get("hd95_mm"),
            "test_dice_organ_balanced": metrics.get("dice_organ_balanced"),
            "best_val_msd": float(self.best_val_msd),
            "best_epoch": int(ckpt.get("epoch_index", 0)) + 1,
        }
        with open(run_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return out
    
    def run_epoch(self, loader, train: bool):
        """Run one epoch of training or validation."""
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss, total_dice, n = 0.0, 0.0, 0
        prompt_mode_counts = {"box": 0, "point": 0, "both": 0}
        
        # Reset evaluator for validation
        if not train:
            self.evaluator.reset()
        
        for batch in loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]

            _bs_samples = images.shape[0] if isinstance(images, torch.Tensor) else len(images)

            if train and self._resume_skip_train_batches > 0:
                self._resume_skip_train_batches -= 1
                continue
            
            # Block-based arm selection for adaptive learning
            if train and self.prompt_strategy == "adaptive":
                # Initialize block if needed (start of training or after evaluation)
                if self._current_block_arm is None:
                    self._current_block_arm = self.sampler.select_arm()
                    self._batches_in_current_block = 0
                    
                    # Explicit logging for paper Methods section
                    ucb_stats = self.sampler.get_statistics()
                    avg_rewards_str = {k: f"{v:.3f}" for k, v in ucb_stats['arm_avg_rewards'].items()}
                    self.logger.info(
                        f"[UCB1 Block {self.sampler.total_pulls}] Selected arm='{self._current_block_arm}' | "
                        f"Arm counts: {ucb_stats['arm_counts']} | "
                        f"Avg rewards: {avg_rewards_str}"
                    )
                
                # Increment within-block batch counter
                self._batches_in_current_block += _bs_samples
                self.batch_counter += _bs_samples
                
                # Check if current block is complete
                if self._batches_in_current_block >= self.block_size_batches:
                    # Evaluate model with current arm (using validation dataset)
                    val_msd = self._quick_validation()
                    
                    # Update bandit with reward for completed block
                    prev_arm = self._current_block_arm
                    self.sampler.update_reward(prev_arm, val_msd)
                    
                    # Explicit logging for paper Methods section
                    reward = self.sampler.reward_history[-1][2] if self.sampler.reward_history else 0
                    self.logger.info(
                        f"[UCB1 Reward] Arm='{prev_arm}' | "
                        f"Val MSD={val_msd:.3f}mm | "
                        f"Reward={reward:.3f} | "
                        f"Arm avg reward: {self.sampler.arm_avg_rewards[prev_arm]:.3f}"
                    )
                    
                    # Log to WandB
                    bandit_stats = self.sampler.get_statistics()
                    log_dict = {
                        "bandit/val_msd": val_msd,
                        "bandit/reward": self.sampler.reward_history[-1][2] if self.sampler.reward_history else 0,
                        "bandit/block_arm": self._current_block_arm,
                        **{f"bandit/arm_{arm}_count": bandit_stats['arm_counts'][arm] for arm in self.sampler.arms},
                        **{f"bandit/arm_{arm}_avg_reward": bandit_stats['arm_avg_rewards'][arm] for arm in self.sampler.arms},
                        "bandit/total_blocks": self.sampler.total_pulls,
                    }
                    wandb.log(log_dict)
                    
                    # Reset for next block (arm will be selected at start of next block)
                    self._current_block_arm = None
                    self._batches_in_current_block = 0
                    if self._checkpoint_run_dir is not None:
                        self.save_checkpoint(
                            self._checkpoint_run_dir / "last_checkpoint.pth",
                            float(val_msd),
                            completed_full_epoch=False,
                        )
            
            # Block-based arm selection for per-organ strategies
            elif train and self._is_per_organ_prompt_strategy():
                if self._current_block_arms is None:
                    if self._uses_true_per_organ_prompt_execution():
                        self._ensure_per_organ_block_arms()
                    else:
                        _ = self._get_prompt_mode("init")
                    self._batches_in_current_block = 0
                
                # Increment within-block batch counter
                self._batches_in_current_block += _bs_samples
                self.batch_counter += _bs_samples
                
                # Check if current block is complete
                if self._batches_in_current_block >= self.block_size_batches:
                    # Evaluate model and get per-organ MSDs
                    organ_msds = self._quick_validation_per_organ()
                    
                    # Update each organ's bandit with its own MSD
                    prev_arms = self._current_block_arms.copy()
                    for organ in self.organs:
                        organ_key = organ.lower()
                        if organ_key in organ_msds:
                            self.sampler.update_reward(organ, prev_arms[organ], organ_msds[organ_key])
                    
                    # Explicit logging
                    self.logger.info(f"[Per-Organ Reward] Arms used: {prev_arms}")
                    for organ in self.organs:
                        organ_key = organ.lower()
                        if organ_key in organ_msds:
                            self.logger.info(
                                f"  {organ}: arm='{prev_arms[organ]}', "
                                f"MSD={organ_msds[organ_key]:.3f}mm"
                            )
                    
                    fallback_log = self._log_block_prompt_fallbacks()

                    # Log to WandB
                    all_stats = self.sampler.get_all_statistics()
                    log_dict = {
                        "bandit/total_blocks": all_stats['aggregated']['total_blocks'],
                        **fallback_log,
                    }
                    
                    # Add per-organ metrics
                    for organ in self.organs:
                        organ_key = organ.lower()
                        if organ in all_stats['per_organ']:
                            organ_stats = all_stats['per_organ'][organ]
                            log_dict[f"bandit/{organ_key}_msd"] = organ_msds.get(organ_key, 0)
                            for arm in self.sampler.arms:
                                log_dict[f"bandit/{organ_key}_arm_{arm}_count"] = organ_stats['arm_counts'][arm]
                                log_dict[f"bandit/{organ_key}_arm_{arm}_avg_reward"] = organ_stats['arm_avg_rewards'][arm]
                    
                    # Add aggregated epsilon/exploration metrics if available
                    if 'avg_epsilon' in all_stats['aggregated']:
                        log_dict['bandit/avg_epsilon'] = all_stats['aggregated']['avg_epsilon']
                    if 'avg_exploration_rate' in all_stats['aggregated']:
                        log_dict['bandit/avg_exploration_rate'] = all_stats['aggregated']['avg_exploration_rate']
                    if 'avg_decay_progress' in all_stats['aggregated']:
                        log_dict['bandit/avg_decay_progress'] = all_stats['aggregated']['avg_decay_progress']
                    
                    wandb.log(log_dict)
                    
                    # Reset for next block
                    self._current_block_arms = None
                    self._batches_in_current_block = 0
                    if self._checkpoint_run_dir is not None:
                        # Use mean organ MSD as scalar reference only (same pattern as logging)
                        mean_ref = float(np.mean(list(organ_msds.values()))) if organ_msds else None
                        self.save_checkpoint(
                            self._checkpoint_run_dir / "last_checkpoint.pth",
                            mean_ref,
                            completed_full_epoch=False,
                        )
            
            # Track failures
            n_invalid_samples = 0
            n_empty_gt = 0
            
            emb_paths = batch.get("embedding_path")
            cached_packs = batch.get("sam_cached_pack")

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                ep = None
                if emb_paths is not None:
                    ep = emb_paths[i]
                cp = None
                if cached_packs is not None:
                    cp = cached_packs[i]
                
                if mask_multi.max() == 0:
                    n_empty_gt += 1
                    continue
                
                out = self._forward_one(
                    image_rgb,
                    mask_multi,
                    fn,
                    is_train=train,
                    embedding_path=ep,
                    cached_embedding_pack=cp,
                )
                if out is None:
                    n_invalid_samples += 1
                    continue
                
                logits, gt, prompt_mode = out
                self._record_prompt_usage(prompt_mode_counts, prompt_mode)
                
                loss = self.loss_fn(
                    logits,
                    gt,
                    dist_maps=self._prepare_boundary_dist_maps(gt, sample_id=fn),
                )
                d = multi_class_dice_from_logits(logits, gt, self.num_classes)
                
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.mask_decoder.parameters(), 1.0)
                    self.optimizer.step()
                else:
                    # Compute boundary metrics for validation
                    pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
                    gt_classes = gt.cpu().numpy()[0, 0]
                    
                    # Evaluate per organ
                    self.evaluator.evaluate_multiclass_slice(
                        pred_classes.astype(np.uint8),
                        gt_classes.astype(np.uint8),
                        slice_idx=n,
                        image_id=fn,
                        class_mapping=self._multiclass_eval_mapping,
                    )
                
                total_loss += float(loss.item())
                total_dice += float(d)
                n += 1

            if train:
                self._train_dataloader_batches_completed_epoch += 1
        
        # Log failure statistics
        if train and (n_invalid_samples > 0 or n_empty_gt > 0):
            self.logger.debug(
                f"Batch failures - Empty GT: {n_empty_gt}, "
                f"Invalid prompts: {n_invalid_samples}, "
                f"Successful: {n}"
            )
        
        if n == 0:
            return 0.0, 0.0, None, None, prompt_mode_counts

        val_metrics = None
        val_msd = None
        if not train and self.evaluator.per_slice_metrics:
            val_metrics = self._summarize_evaluator_metrics()
            if val_metrics:
                val_msd = val_metrics.get("msd_mm")

        return total_loss / n, total_dice / n, val_msd, val_metrics, prompt_mode_counts
    
    def _quick_validation(self):
        """Quick validation run to compute MSD for bandit reward.
        
        Uses a fixed deterministic probe set to reduce noise in reward signal.
        ``organ_balanced`` uses equal weight per organ in ``_evaluator_organ_labels``
        (simulation: Rectum, Bladder, PTV1; clinical: Bladder, PTV, Rectum, Urethra).
        """
        if self._val_dataset_ref is None:
            raise ValueError("Validation dataset not initialized. Call train_loop() first.")
        
        self.model.eval()
        self.evaluator.reset()
        
        val_dataset = self._val_dataset_ref
        indices = self._validation_probe_indices
        
        with torch.no_grad():
            for idx in indices:
                sample = val_dataset[idx]
                image_rgb = sample["image"].numpy() if torch.is_tensor(sample["image"]) else sample["image"]
                mask_multi = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else sample["mask"]
                fn = sample["filename"]
                ep = sample.get("embedding_path")
                cp = sample.get("sam_cached_pack")
                
                if mask_multi.max() == 0:
                    continue
                
                out = self._forward_one(
                    image_rgb,
                    mask_multi,
                    fn,
                    is_train=False,
                    embedding_path=ep,
                    cached_embedding_pack=cp,
                )
                if out is None:
                    continue
                
                logits, gt, _ = out
                pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
                gt_classes = gt.cpu().numpy()[0, 0]
                
                self.evaluator.evaluate_multiclass_slice(
                    pred_classes.astype(np.uint8),
                    gt_classes.astype(np.uint8),
                    slice_idx=idx,
                    image_id=fn,
                    class_mapping=self._multiclass_eval_mapping,
                )
        
        # Compute reward based on aggregation strategy
        if self.reward_aggregation == "organ_balanced":
            # Organ-balanced: equal weight per organ (avoid slice-count bias)
            organ_names = self._evaluator_organ_labels
            organ_msds = []
            
            for organ in organ_names:
                organ_summary = self.evaluator.aggregate_metrics(organ_name=organ)
                if organ_summary and organ_summary.get('msd_mm_mean') is not None:
                    organ_msds.append(organ_summary['msd_mm_mean'])
            
            if len(organ_msds) > 0:
                msd = np.mean(organ_msds)  # Equal weight per organ
            else:
                msd = 10.0  # Default
        else:
            # Slice-weighted (original approach)
            overall_summary = self.evaluator.aggregate_metrics(organ_name=None)
            if overall_summary and overall_summary.get('msd_mm_mean') is not None:
                msd = overall_summary['msd_mm_mean']
            else:
                msd = 10.0
        
        return msd
    
    def _quick_validation_per_organ(self):
        """Quick validation run to compute per-organ MSDs for per-organ bandits.
        
        Returns:
            dict: ``{organ_key: msd_mm}`` for each organ (taxonomy from ``data_root``).
        """
        if self._val_dataset_ref is None:
            raise ValueError("Validation dataset not initialized. Call train_loop() first.")
        
        self.model.eval()
        self.evaluator.reset()
        
        val_dataset = self._val_dataset_ref
        indices = self._validation_probe_indices
        
        with torch.no_grad():
            for idx in indices:
                sample = val_dataset[idx]
                image_rgb = sample["image"].numpy() if torch.is_tensor(sample["image"]) else sample["image"]
                mask_multi = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else sample["mask"]
                fn = sample["filename"]
                ep = sample.get("embedding_path")
                cp = sample.get("sam_cached_pack")
                
                if mask_multi.max() == 0:
                    continue
                
                out = self._forward_one(
                    image_rgb,
                    mask_multi,
                    fn,
                    is_train=False,
                    embedding_path=ep,
                    cached_embedding_pack=cp,
                )
                if out is None:
                    continue
                
                logits, gt, _ = out
                pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
                gt_classes = gt.cpu().numpy()[0, 0]
                
                self.evaluator.evaluate_multiclass_slice(
                    pred_classes.astype(np.uint8),
                    gt_classes.astype(np.uint8),
                    slice_idx=idx,
                    image_id=fn,
                    class_mapping=self._multiclass_eval_mapping,
                )
        
        # Compute per-organ MSDs
        organ_msds = {}
        for o in self._organs_for_per_organ_metrics:
            organ_summary = self.evaluator.aggregate_metrics(organ_name=o.evaluator_label)
            if organ_summary and organ_summary.get('msd_mm_mean') is not None:
                organ_msds[o.key] = organ_summary['msd_mm_mean']
            else:
                organ_msds[o.key] = 10.0  # Default penalty for missing data
        
        return organ_msds
    
    def save_checkpoint(
        self,
        path: Path,
        val_msd: Optional[float],
        *,
        completed_full_epoch: bool = True,
    ):
        """Persist decoder + optimizer + scheduler + sampler (+ RNG) for resume."""
        per_organ_policy = None
        if self._is_per_organ_prompt_strategy() and hasattr(self.sampler, "get_best_arms_per_organ"):
            per_organ_policy = self.sampler.get_best_arms_per_organ()

        ckpt = {
            "format_version": 2,
            "completed_full_epoch": completed_full_epoch,
            "epoch_index": int(self.current_epoch),
            "train_batches_completed_this_epoch": int(self._train_dataloader_batches_completed_epoch),
            "prompt_strategy": self.prompt_strategy,
            "collapse_per_organ_arms_to_majority": bool(self.collapse_per_organ_arms_to_majority),
            "per_organ_prompt_policy": per_organ_policy,
            "num_classes": self.num_classes,
            "best_val_msd_tracked": float(self.best_val_msd),
            "val_msd_at_save": float(val_msd) if val_msd is not None else None,
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "batch_counter": int(self.batch_counter),
            "block_state": {
                "current_block_arm": self._current_block_arm,
                "batches_in_current_block": int(self._batches_in_current_block),
                "current_block_arms": (
                    dict(self._current_block_arms) if self._current_block_arms is not None else None
                ),
            },
            "model_state": {
                "mask_decoder": self.model.mask_decoder.state_dict(),
            },
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.cfg,
            "sampler_state": pickle.dumps(self.sampler, protocol=pickle.HIGHEST_PROTOCOL),
        }
        rng_bundle = {
            "torch": torch.get_rng_state(),
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_bundle["cuda"] = torch.cuda.get_rng_state_all()
        ckpt["rng_state"] = rng_bundle
        gen = self._train_generator_ref
        if gen is not None:
            ckpt["train_generator_state"] = gen.get_state()

        torch.save(ckpt, path)
        kind = "epoch-end" if completed_full_epoch else "mid-epoch"
        self.logger.info(
            f"Checkpoint saved ({kind}): {path} | epoch_index={self.current_epoch} | "
            f"train_batches_completed_this_epoch={self._train_dataloader_batches_completed_epoch}"
        )

    def apply_checkpoint(self, ckpt: dict, train_generator: Optional[torch.Generator] = None):
        """Restore trainable state from ``save_checkpoint`` output."""
        if ckpt.get("format_version") != 2:
            raise ValueError(
                "Resume requires checkpoints written with format_version=2 "
                "(train at least once with this code version so last_checkpoint.pth is upgraded)."
            )
        if ckpt.get("sampler_state") is None:
            raise ValueError("Checkpoint missing sampler_state; cannot resume sampler-driven strategies.")

        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        best = ckpt.get("best_val_msd_tracked")
        if best is None:
            best = ckpt.get("best_val_msd")
        self.best_val_msd = float(best) if best is not None else float("inf")

        self.epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
        self.batch_counter = int(ckpt.get("batch_counter", 0))

        blk = ckpt.get("block_state") or {}
        self._current_block_arm = blk.get("current_block_arm")
        self._batches_in_current_block = int(blk.get("batches_in_current_block", 0))
        cba = blk.get("current_block_arms")
        self._current_block_arms = dict(cba) if cba is not None else None
        if "collapse_per_organ_arms_to_majority" in ckpt:
            self.collapse_per_organ_arms_to_majority = bool(ckpt["collapse_per_organ_arms_to_majority"])

        self.sampler = pickle.loads(ckpt["sampler_state"])

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

        completed = ckpt.get("completed_full_epoch", True)
        ei = int(ckpt.get("epoch_index", 0))
        if completed:
            self._resume_start_epoch = ei + 1
            self._resume_skip_train_batches = 0
            self._train_dataloader_batches_completed_epoch = 0
            self._resume_mid_epoch_active = False
            self._resume_mid_epoch_start_epoch = -1
        else:
            self._resume_start_epoch = ei
            skip = int(ckpt.get("train_batches_completed_this_epoch", 0))
            self._resume_skip_train_batches = skip
            self._train_dataloader_batches_completed_epoch = skip
            self._resume_mid_epoch_active = True
            self._resume_mid_epoch_start_epoch = ei
            self.logger.info(
                f"Resume mid-epoch: skip first {skip} train loader batches in epoch {ei}"
            )

        self.logger.info(
            f"Loaded checkpoint: resume from epoch index {self._resume_start_epoch} | "
            f"best_val_msd={self.best_val_msd:.4f} | "
            f"epochs_without_improvement={self.epochs_without_improvement}"
        )

    
    def train_loop(self, train_loader, val_loader):
        """Main training loop."""
        run_dir = Path(self.cfg["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_run_dir = run_dir
        
        # Initialize validation dataset reference and probe set for adaptive/per-organ learning
        if self._is_per_organ_prompt_strategy() or self.prompt_strategy == "adaptive":
            self._val_dataset_ref = val_loader.dataset
            probe_size = min(self.probe_size, len(self._val_dataset_ref))
            rng = np.random.default_rng(self.probe_seed)
            self._validation_probe_indices = rng.choice(
                len(self._val_dataset_ref), probe_size, replace=False
            )
            strategy_name = "Bandit" if self.prompt_strategy == "adaptive" else "Per-Organ Bandit"
            self.logger.info(f"{strategy_name} probe set initialized: {probe_size} images (seed={self.probe_seed})")
            self.logger.info(f"  Probe indices (first 5): {list(self._validation_probe_indices[:5])}")
            self.logger.info(
                f"  Bandit block threshold: {self.block_size_batches} "
                f"(each train DataLoader step adds len(batch); YAML key block_size_batches)"
            )
            self.logger.info(f"  CRITICAL: Bandit validates on VAL dataset only, never TRAIN")
        
        self.logger.info(f"Starting training: {self.cfg['run_name']}")
        self.logger.info(f"Output directory: {run_dir}")
        self.logger.info(f"Total epochs: {self.cfg['max_epochs']}")
        self.logger.info(f"Prompt strategy: {self.prompt_strategy}")
        
        # CSV logging
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=LEGACY_METRICS_CSV_FIELDS).writeheader()
        
        start_epoch = int(getattr(self, "_resume_start_epoch", 0))
        
        for epoch in range(start_epoch, self.cfg["max_epochs"]):
            self.current_epoch = epoch
            
            if getattr(self, "_resume_mid_epoch_active", False) and epoch > self._resume_mid_epoch_start_epoch:
                self._resume_mid_epoch_active = False
            if not (
                getattr(self, "_resume_mid_epoch_active", False)
                and epoch == getattr(self, "_resume_mid_epoch_start_epoch", -1)
            ):
                self._train_dataloader_batches_completed_epoch = 0
            
            t0 = time.time()
            
            # Reset sampler statistics
            if hasattr(self.sampler, 'reset_statistics'):
                self.sampler.reset_statistics()
            
            train_loss, train_dice, _, _, train_prompt_counts = self.run_epoch(
                train_loader, train=True
            )
            val_loss, val_dice, _, val_metrics, val_prompt_counts = self.run_epoch(
                val_loader, train=False
            )
            ckpt_scores = self._validation_checkpoint_scores()
            val_msd = ckpt_scores.primary_msd if ckpt_scores is not None else None

            val_per_organ_wandb = (
                self._per_organ_wandb_dict("val") if val_metrics else {}
            )

            train_metrics: Dict[str, Optional[float]] = {}
            train_per_organ_wandb: Dict[str, float] = {}
            if self.compute_train_boundary_metrics:
                train_metrics = self._collect_boundary_metrics_from_loader(train_loader)
                train_per_organ_wandb = self._per_organ_wandb_dict("train")

            lr = self.optimizer.param_groups[0]["lr"]
            
            # Save last checkpoint (epoch boundary — fine-grained bandit resumes use mid-epoch saves too)
            if val_msd is not None:
                self.save_checkpoint(
                    run_dir / "last_checkpoint.pth",
                    val_msd,
                    completed_full_epoch=True,
                )
            
            # Best checkpoint (primary validation MSD objective)
            if is_better_checkpoint(
                val_msd,
                self.best_val_msd,
                min_delta=self._checkpoint_objective.min_delta,
            ):
                prev_best = self.best_val_msd
                self.best_val_msd = float(val_msd)
                self.save_checkpoint(
                    run_dir / "best_checkpoint.pth",
                    val_msd,
                    completed_full_epoch=True,
                )
                self.epochs_without_improvement = 0
                label = self._checkpoint_objective.metric
                if prev_best < float('inf'):
                    self.logger.info(
                        f"✓ New best {label}: {val_msd:.4f} mm (improved from {prev_best:.4f} mm)"
                    )
                else:
                    self.logger.info(f"✓ New best {label}: {val_msd:.4f} mm")
            else:
                self.epochs_without_improvement += 1
                cur = f"{val_msd:.4f}" if val_msd is not None else "N/A"
                self.logger.info(
                    f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs "
                    f"(current: {cur} mm, best: {self.best_val_msd:.4f} mm)"
                )
                
                if self.epochs_without_improvement >= self.patience:
                    self.logger.warning(f"Early stopping triggered after {epoch+1} epochs")
                    self.logger.info(f"Final best val_msd: {self.best_val_msd:.4f} mm")
                    break
            
            def _m(d: Optional[Dict[str, Optional[float]]], key: str):
                if not d:
                    return ""
                v = d.get(key)
                return v if v is not None else ""

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_msd": _m(train_metrics, "msd_mm_organ_balanced") or _m(train_metrics, "msd_mm"),
                "train_hd95": _m(train_metrics, "hd95_mm_organ_balanced") or _m(train_metrics, "hd95_mm"),
                "train_dice_organ_balanced": _m(train_metrics, "dice_organ_balanced"),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_msd": val_msd if val_msd is not None else "",
                "val_hd95": (
                    ckpt_scores.organ_balanced_hd95
                    if ckpt_scores is not None
                    else _m(val_metrics, "hd95_mm_organ_balanced") or _m(val_metrics, "hd95_mm")
                ),
                "val_dice_organ_balanced": (
                    ckpt_scores.organ_balanced_dice
                    if ckpt_scores is not None
                    else _m(val_metrics, "dice_organ_balanced")
                ),
                "val_ptv_hd95": ckpt_scores.ptv_hd95 if ckpt_scores is not None else "",
                "val_ptv_dice": ckpt_scores.ptv_dice if ckpt_scores is not None else "",
                "best_val_msd": self.best_val_msd,
                "lr": lr,
                "train_box_count": train_prompt_counts.get("box", 0),
                "train_point_count": train_prompt_counts.get("point", 0),
                "train_both_count": train_prompt_counts.get("both", 0),
                "val_box_count": val_prompt_counts.get("box", 0),
                "val_point_count": val_prompt_counts.get("point", 0),
                "val_both_count": val_prompt_counts.get("both", 0),
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LEGACY_METRICS_CSV_FIELDS).writerow(row)
            
            # WandB logging
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/dice": train_dice,
                "val/loss": val_loss,
                "val/dice": val_dice,
                "lr": lr,
                "time/epoch_s": time.time() - t0,
                "prompts/train_box_count": train_prompt_counts.get("box", 0),
                "prompts/train_point_count": train_prompt_counts.get("point", 0),
                "prompts/train_both_count": train_prompt_counts.get("both", 0),
            }
            
            if val_msd is not None:
                log_dict["val/msd_mm"] = val_msd
            if val_metrics:
                ob_msd = val_metrics.get("msd_mm_organ_balanced")
                ob_hd = val_metrics.get("hd95_mm_organ_balanced")
                ob_dice = val_metrics.get("dice_organ_balanced")
                if ob_msd is not None:
                    log_dict["val/msd_mm_organ_balanced"] = ob_msd
                if ob_hd is not None:
                    log_dict["val/hd95_mm"] = ob_hd
                if val_metrics.get("hd95_mm") is not None:
                    log_dict["val/hd95_mm_slice_weighted"] = val_metrics["hd95_mm"]
                if ob_dice is not None:
                    log_dict["val/dice_organ_balanced"] = ob_dice
                log_dict.update(val_per_organ_wandb)
            if train_metrics:
                tr_msd = train_metrics.get("msd_mm_organ_balanced") or train_metrics.get("msd_mm")
                tr_hd = train_metrics.get("hd95_mm_organ_balanced") or train_metrics.get("hd95_mm")
                tr_dice = train_metrics.get("dice_organ_balanced")
                if tr_msd is not None:
                    log_dict["train/msd_mm"] = tr_msd
                if tr_hd is not None:
                    log_dict["train/hd95_mm"] = tr_hd
                if tr_dice is not None:
                    log_dict["train/dice_organ_balanced"] = tr_dice
                log_dict.update(train_per_organ_wandb)
            
            # Log sampler statistics
            if hasattr(self.sampler, 'get_statistics'):
                stats = self.sampler.get_statistics()
                if self.prompt_strategy == "box_point":
                    log_dict.update({
                        "prompts/box_ratio_actual": stats.get("box_ratio_actual", 0),
                        "prompts/point_ratio_actual": stats.get("point_ratio_actual", 0),
                    })
                elif self.prompt_strategy == "three_way":
                    log_dict.update({
                        "prompts/box_ratio_actual": stats.get("box_ratio_actual", 0),
                        "prompts/point_ratio_actual": stats.get("point_ratio_actual", 0),
                        "prompts/both_ratio_actual": stats.get("both_ratio_actual", 0),
                    })
                elif self.prompt_strategy == "adaptive":
                    log_dict.update({
                        "bandit/total_pulls": stats.get("total_pulls", 0),
                        **{f"bandit/arm_{arm}_count": stats['arm_counts'][arm] for arm in self.sampler.arms},
                        **{f"bandit/arm_{arm}_avg_reward": stats['arm_avg_rewards'][arm] for arm in self.sampler.arms},
                        **{f"bandit/arm_{arm}_selection_rate": stats['arm_selection_rates'][arm] for arm in self.sampler.arms},
                    })
            
            # Log per-organ bandit statistics
            if hasattr(self.sampler, 'get_all_statistics') and self._is_per_organ_prompt_strategy():
                all_stats = self.sampler.get_all_statistics()
                
                # Aggregated metrics
                log_dict.update({
                    "bandit/total_blocks": all_stats['aggregated']['total_blocks'],
                })
                
                # Add epsilon/exploration metrics if available
                if 'avg_epsilon' in all_stats['aggregated']:
                    log_dict['bandit/avg_epsilon'] = all_stats['aggregated']['avg_epsilon']
                if 'avg_exploration_rate' in all_stats['aggregated']:
                    log_dict['bandit/avg_exploration_rate'] = all_stats['aggregated']['avg_exploration_rate']
                if 'avg_decay_progress' in all_stats['aggregated']:
                    log_dict['bandit/avg_decay_progress'] = all_stats['aggregated']['avg_decay_progress']
                
                # Per-organ metrics
                for organ in self.organs:
                    organ_key = organ.lower()
                    if organ in all_stats['per_organ']:
                        organ_stats = all_stats['per_organ'][organ]
                        
                        # Best arm for this organ
                        best_arm = max(organ_stats['arm_avg_rewards'].items(), key=lambda x: x[1])[0]
                        log_dict[f"bandit/{organ_key}_best_arm"] = {"box": 0, "point": 1, "both": 2}[best_arm]
                        
                        # Per-arm metrics for each organ
                        for arm in self.sampler.arms:
                            log_dict[f"bandit/{organ_key}_arm_{arm}_count"] = organ_stats['arm_counts'][arm]
                            log_dict[f"bandit/{organ_key}_arm_{arm}_avg_reward"] = organ_stats['arm_avg_rewards'][arm]
                            log_dict[f"bandit/{organ_key}_arm_{arm}_selection_rate"] = organ_stats['arm_selection_rates'][arm]
            
            wandb.log(log_dict)
            
            ob_val_msd = (val_metrics or {}).get("msd_mm_organ_balanced")
            ob_val_hd = (val_metrics or {}).get("hd95_mm_organ_balanced")
            msd_str = f"{ob_val_msd:.4f}" if ob_val_msd is not None else (f"{val_msd:.4f}" if val_msd else "N/A")
            hd_str = f"{ob_val_hd:.4f}" if ob_val_hd is not None else "N/A"
            epoch_summary = (
                f"[{epoch+1}/{self.cfg['max_epochs']}] "
                f"Train: loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}, msd={msd_str} mm, hd95={hd_str} mm | "
                f"LR={lr:.2e} | {time.time()-t0:.1f}s"
            )
            self.logger.info(epoch_summary)
            
            # Log prompt distribution
            if train_prompt_counts:
                self.logger.debug(f"Train prompts - Box: {train_prompt_counts.get('box', 0)}, "
                                 f"Point: {train_prompt_counts.get('point', 0)}, "
                                 f"Both: {train_prompt_counts.get('both', 0)}")
            
            self.scheduler.step()
        
        self.logger.info(f"Training completed! Best val_msd: {self.best_val_msd:.4f} mm")
        self.logger.info(f"Checkpoints saved to: {run_dir}")

    def run_test_export_if_configured(self, test_loader) -> Optional[Dict[str, Any]]:
        if not self.run_test_after_train or test_loader is None:
            return None
        if not (self._checkpoint_run_dir / "best_checkpoint.pth").is_file():
            self.logger.warning("Skipping test export: no best_checkpoint.pth")
            return None
        from bapmos.paths import (
            method_slug_from_checkpoint,
            method_test_output_dir_from_checkpoint,
            write_method_evaluation_meta,
        )

        best_path = self._checkpoint_run_dir / "best_checkpoint.pth"
        out_dir = method_test_output_dir_from_checkpoint(
            best_path, self.cfg["data_root"]
        )
        write_method_evaluation_meta(
            out_dir,
            checkpoint=best_path,
            data_root=self.cfg["data_root"],
            method_slug=method_slug_from_checkpoint(best_path),
            split="test",
            extra={
                "prompt_strategy": self.prompt_strategy,
                "run_name": self.cfg.get("run_name"),
            },
        )
        self.logger.info("Running test split evaluation → %s", out_dir)
        result = self.export_test_split_metrics(test_loader, output_dir=out_dir)
        test_log = {
            "test/loss": result.get("test_loss"),
            "test/dice": result.get("test_dice"),
            "test/msd_mm": result.get("test_msd_mm"),
            "test/hd95_mm": result.get("test_hd95_mm"),
        }
        test_log = {k: v for k, v in test_log.items() if v is not None}
        if test_log:
            wandb.log(test_log)
            wandb.summary.update({k.replace("/", "_"): v for k, v in test_log.items()})
        self.logger.info("Test results: %s", result)
        return result


def load_config(config_path: Path, experiment_name: str):
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Find experiment
    exp_cfg = None
    for exp in cfg["experiments"]:
        if exp["name"] == experiment_name:
            exp_cfg = exp
            break
    
    if exp_cfg is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")
    
    # Merge common config with experiment config
    merged = {**cfg["common"], **exp_cfg}
    merged["wandb_project"] = cfg["wandb_project"]
    
    # Merge bandit config if it exists (for adaptive learning)
    if "bandit" in cfg:
        merged["bandit"] = cfg["bandit"]

    if "checkpoint_objective" in cfg:
        merged["checkpoint_objective"] = cfg["checkpoint_objective"]
    
    return merged


def _config_path_key(config_path: Path) -> str:
    """
    Path key for strategy inference.

    Standalone BAPMOS layouts use ``…/<strategy>/config.yaml``, so the basename alone
    is not enough (unlike historical ``configs/case_1/ucb1_global.yaml``).
    """
    parts = config_path.resolve().parts
    # Keep enough trailing components to disambiguate policies vs ratios.
    return "/".join(parts[-5:] if len(parts) >= 5 else parts)


def _infer_prompt_strategy(config_path: Path, config: dict) -> None:
    """Set ``config['prompt_strategy']`` from the YAML path (filename and/or parents)."""
    key = _config_path_key(config_path).replace("\\", "/")
    if "boxpoint_box_point" in key:
        config["prompt_strategy"] = "three_way"
    elif "box_point" in key:
        config["prompt_strategy"] = "box_point"
    elif "ucb1_per_organ" in key:
        config["prompt_strategy"] = "ucb1_per_organ"
    elif "epsilon_greedy_per_organ" in key:
        config["prompt_strategy"] = "epsilon_greedy_per_organ"
    elif "epsilon_decay_per_organ" in key:
        config["prompt_strategy"] = "epsilon_decay_per_organ"
    elif "bap_mos_tuned" in key:
        config["prompt_strategy"] = "bap_mos_tuned"
    elif "ucb_tuned_per_organ_majority" in key:
        config["prompt_strategy"] = "ucb_tuned_per_organ"
    elif "ucb_tuned_per_organ" in key:
        config["prompt_strategy"] = "ucb_tuned_per_organ"
    elif "adaptive" in key or "ucb1" in key:
        config["prompt_strategy"] = "adaptive"
    else:
        raise ValueError(
            f"Cannot determine prompt strategy from config path: {config_path} (key={key!r})"
        )


def optimization_strategy_folder(config_path: Path, prompt_strategy: str) -> str:
    """Run/log subdirectory under ``Optimization/`` (may differ from ``prompt_strategy``)."""
    key = _config_path_key(config_path).replace("\\", "/")
    if "ucb_tuned_per_organ_majority" in key:
        return "ucb_tuned_per_organ_majority"
    if prompt_strategy == "three_way":
        return "boxpoint_box_point"
    if prompt_strategy == "adaptive":
        return "ucb1_global"
    return prompt_strategy


def apply_training_cli_overrides(
    config: dict,
    *,
    max_epochs: Optional[int],
    patience: Optional[int],
) -> None:
    """Mutate ``config`` with optional CLI overrides (used for fresh runs and resume)."""
    if max_epochs is not None:
        config["max_epochs"] = int(max_epochs)
    if patience is not None:
        config["patience"] = int(patience)


def apply_replicate_seed_overrides(config: dict) -> None:
    """Honor PI_TRAIN_SEED / PI_CHECKPOINT_KFOLD_SEED for independent replicate runs."""
    train_seed = os.environ.get("PI_TRAIN_SEED", "").strip()
    if train_seed:
        seed = int(train_seed)
        config["seed"] = seed
        bandit = config.get("bandit")
        if isinstance(bandit, dict):
            bandit = dict(bandit)
            bandit["probe_seed"] = int(
                os.environ.get("PI_PROBE_SEED", train_seed)
            )
            config["bandit"] = bandit
    kfold_seed = os.environ.get("PI_CHECKPOINT_KFOLD_SEED", "").strip()
    if kfold_seed:
        ks = int(kfold_seed)
        obj = dict(config.get("checkpoint_objective") or {})
        obj["kfold_seed"] = ks
        config["checkpoint_objective"] = obj
        config["checkpoint_kfold_seed"] = ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name from config")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name")
    parser.add_argument(
        "--run_root",
        type=str,
        default=None,
        help=(
            "Checkpoint parent before strategy subfolder. Default: runs/<bundle>/Optimization "
            "with <bundle> inferred from YAML data_root (case_1, case_2, simulation, pfus1). "
            "For ablation sweeps use runs/Ablation (see bapmos.legacy.ablations)."
        ),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (default: WANDB_ENTITY env if set; omit entity when unset).",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help=(
            "Log parent before strategy subfolder. Default: logs/Optimization, "
            "or logs/<name> when --run_root is runs/<name>."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from last_checkpoint.pth. Pass a path, or 'auto' to use "
            "<run_root>/<strategy>/<run_name>/last_checkpoint.pth (requires --run_name "
            "to match the prior run directory, including finished runs)."
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help=(
            "If set, overrides config max_epochs after loading YAML or checkpoint. "
            "Typical with --resume when extending a run that already reached the original max_epochs."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="If set, overrides config patience after loading YAML or checkpoint.",
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    resume_arg = (args.resume or "").strip()
    
    ckpt_obj = None
    if resume_arg:
        if resume_arg.lower() == "auto":
            if not args.run_name:
                raise ValueError("--resume auto requires --run_name matching the prior run directory.")
            yaml_cfg = load_config(config_path, args.experiment)
            sam_ckpt_path = resolve_model_checkpoint(yaml_cfg["sam_checkpoint"])
            if not sam_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {yaml_cfg['sam_checkpoint']!r} "
                    f"(expected under {project_root()}/models)"
                )
            yaml_cfg["sam_checkpoint"] = str(sam_ckpt_path)
            _infer_prompt_strategy(config_path, yaml_cfg)
            sf = optimization_strategy_folder(config_path, yaml_cfg["prompt_strategy"])
            bundle = dataset_bundle_tag(yaml_cfg["data_root"])
            default_rr = f"runs/{bundle}/Optimization"
            rr = (args.run_root or default_rr).strip()
            ckpt_fp = Path(rr) / sf / args.run_name / "last_checkpoint.pth"
            if not ckpt_fp.is_file():
                raise FileNotFoundError(f"--resume auto: missing checkpoint {ckpt_fp}")
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
            config = ckpt_obj["config"]
        else:
            ckpt_fp = resolve_under_project(resume_arg)
            if not ckpt_fp.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
            ckpt_obj = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
            config = ckpt_obj["config"]
    else:
        config = load_config(config_path, args.experiment)

    ckpt_path = resolve_model_checkpoint(config["sam_checkpoint"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {config['sam_checkpoint']!r} "
            f"(expected under {project_root()}/models)"
        )
    config["sam_checkpoint"] = str(ckpt_path)

    apply_training_cli_overrides(
        config, max_epochs=args.max_epochs, patience=args.patience
    )
    apply_replicate_seed_overrides(config)
    
    if not resume_arg:
        _infer_prompt_strategy(config_path, config)
    
    # Seed (checkpoint resume overwrites RNG inside apply_checkpoint)
    seed_everything(config["seed"], deterministic=True)
    
    # Run directory layout
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.experiment}_{timestamp}"
    
    strategy_folder = optimization_strategy_folder(config_path, config["prompt_strategy"])

    bundle = dataset_bundle_tag(config["data_root"])
    default_run_root = f"runs/{bundle}/Optimization"
    default_log_root = "logs/Optimization"
    run_root_base = (args.run_root or default_run_root).strip()
    if args.log_root is not None:
        log_root_base = args.log_root.strip()
    else:
        rp = Path(run_root_base)
        if len(rp.parts) >= 2 and rp.parts[0] == "runs":
            log_root_base = str(Path("logs") / rp.parts[1])
        else:
            log_root_base = default_log_root

    if not resume_arg:
        run_dir = Path(run_root_base) / strategy_folder / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config["run_dir"] = str(run_dir)
        config["run_name"] = run_name
        config["run_root"] = run_root_base
        config["log_root"] = log_root_base
    else:
        run_dir = Path(config["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = config["run_name"]
        run_root_base = config["run_root"]
        log_root_base = config["log_root"]
        rr = Path(run_root_base)
        if not rr.is_absolute():
            rr = project_root() / rr
        rd = run_dir if run_dir.is_absolute() else project_root() / run_dir
        strategy_folder = rd.resolve().relative_to(rr.resolve()).parts[0]

    # Setup logging
    log_dir = Path(log_root_base) / strategy_folder
    logger = setup_logging(log_dir, run_name)
    
    # Save config (refresh on resume so edits to YAML are visible if user merges manually)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("="*80)
    logger.info(f"Multi-Organ Optimization Training: {config['prompt_strategy']}")
    if ckpt_obj is not None:
        logger.info("RESUME from checkpoint (format_version=%s)", ckpt_obj.get("format_version"))
    logger.info("="*80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Dataset bundle slug: {bundle} (default layout: runs/<slug>/Optimization, logs/<slug>/...)")
    logger.info(f"Log directory: {log_dir} (run_root={run_root_base}, log_root={log_root_base})")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Seed: {config['seed']}")
    logger.info(f"Max epochs: {config['max_epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr']}")
    logger.info(f"Data root: {config['data_root']}")
    logger.info(
        f"Splits subdirectory: {config.get('splits_subdir', 'splits_stratified')}"
    )
    dr = Path(config["data_root"])
    if not dr.is_absolute():
        dr = project_root() / dr
    ss_fp = dr / config.get("splits_subdir", "splits_stratified") / "split_summary.json"
    if ss_fp.is_file():
        try:
            ss = json.loads(ss_fp.read_text())
            logger.info(f"split_summary.json random_seed: {ss.get('random_seed')}")
        except (json.JSONDecodeError, OSError):
            pass
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("="*80 + "\n")
    
    # WandB (entity only from CLI / WANDB_ENTITY — never hardcode a private org)
    if args.wandb_entity:
        config["wandb_entity"] = args.wandb_entity
    wandb_kw = {
        "project": config["wandb_project"],
        "name": run_name,
        "config": config,
    }
    entity = (
        config.get("wandb_entity") or os.environ.get("WANDB_ENTITY") or ""
    ).strip()
    if entity:
        wandb_kw["entity"] = entity
    wandb.init(**wandb_kw)
    wandb.define_metric("epoch")
    logger.info(f"WandB project: {config['wandb_project']}")
    logger.info(f"WandB run: {run_name}\n")
    
    # Datasets
    ds_kwargs = dict(
        splits_subdir=config.get("splits_subdir", "splits_stratified"),
    )
    if config.get("use_cached_image_embeddings"):
        emb_dir = config.get("image_embedding_dir")
        if not emb_dir:
            raise ValueError(
                "use_cached_image_embeddings requires image_embedding_dir in the merged config."
            )
        ds_kwargs["image_embedding_dir"] = str(resolve_under_project(emb_dir))

    train_dataset = MultiOrganDataset(
        config["data_root"],
        split="train",
        **ds_kwargs,
    )
    val_dataset = MultiOrganDataset(
        config["data_root"],
        split="val",
        **ds_kwargs,
    )

    if config.get("use_cached_image_embeddings"):
        emb_root = Path(ds_kwargs["image_embedding_dir"])
        s0 = train_dataset[0]
        ex_path = s0.get("embedding_path", "")
        logger.info(
            "Cached SAM embeddings: resolved_dir=%s | splits_subdir=%s | "
            "train_samples=%d val_samples=%d | worker_preload=sam_cached_pack | example_path=%s",
            emb_root,
            ds_kwargs["splits_subdir"],
            len(train_dataset),
            len(val_dataset),
            ex_path,
        )
    
    g = torch.Generator()
    g.manual_seed(config["seed"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=multi_organ_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=multi_organ_collate_fn,
    )
    
    # Train
    logger.info("Initializing trainer...")
    trainer = OptimizationTrainer(config)
    if ckpt_obj is not None:
        trainer.apply_checkpoint(ckpt_obj, g)
    
    test_loader = None
    if config.get("run_test_after_train"):
        from bapmos.paths import should_skip_in_train_global_test

        splits_subdir = ds_kwargs.get("splits_subdir", "splits_stratified")
        if should_skip_in_train_global_test(config["data_root"], splits_subdir):
            logger.info(
                "Skipping in-train test export (no global %s/test.txt; "
                "pooled uses site_tests/; run stratified inference_output separately).",
                splits_subdir,
            )
            config["run_test_after_train"] = False
        else:
            test_dataset = MultiOrganDataset(
                config["data_root"],
                split="test",
                **ds_kwargs,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
                collate_fn=multi_organ_collate_fn,
            )
            logger.info("Test split enabled: %d samples", len(test_dataset))

    logger.info("Starting training loop...")
    trainer.train_loop(train_loader, val_loader)
    trainer.run_test_export_if_configured(test_loader)
    
    wandb.finish()
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best validation MSD: {trainer.best_val_msd:.4f} mm")
    logger.info(f"Checkpoints: {run_dir}")
    logger.info(f"Logs: {log_dir / run_name}.log")
    logger.info("="*80)


if __name__ == "__main__":
    main()
