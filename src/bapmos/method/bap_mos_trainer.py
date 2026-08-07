"""
BAP-MOS training loop integrating prompt geometry, bandit policy, and curriculum reward.

Training flow per decision block (K batches):
  1. ``BlockScheduler.begin_block`` — select frozen arms per organ (no t/N_a increment).
  2. Train K batches with ``prompt_geometry`` + single-forward SAM per organ.
  3. Validation MSD per organ (val probe) → reward → ``BlockScheduler.end_block``.

After each epoch: val metrics (MSD, Dice, HD95) for monitoring; optional full
train metrics when ``evaluation.eval_train_metrics`` is true.
Best-checkpoint selection uses the configured validation **objective**
(``evaluation.objective_metric``, typically ``ptv_kfold_msd``).
After training: test-split evaluation with best checkpoint (when enabled).

Optional ``common.use_cached_image_embeddings`` skips the frozen ViT encoder when
precomputed ``.pt`` packs are provided via ``common.image_embedding_dir``.

Checkpoint resume restores model/optimizer/bandit/scheduler state and core RNGs
(Python / NumPy / Torch / CUDA). Exact DataLoader shuffle order may still drift if
workers or sampler epoch state are not fully controlled by those RNGs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from bapmos.method.bandit_policy import BlockScheduler, PerOrganBanditPolicy
from bapmos.method.config_utils import run_directory_layout
from bapmos.method.paths import bapmos_method_test_output_dir_from_config, method_slug_from_config
from bapmos.method.curriculum_reward import compute_gamma
from bapmos.method.data_adapter import build_dataloaders, load_yaml_config, resolve_path
from bapmos.method.evaluation_metrics import (
    build_bandit_epoch_wandb_dict,
    build_block_wandb_dict,
    empty_prompt_counts,
    export_test_split_metrics,
    merge_prompt_counts,
    resolve_taxonomy,
    resolve_ptv_evaluator_organ,
    run_split_metrics_pass,
    validation_objective_msd,
    validate_block_per_organ_probe_metrics,
)
from bapmos.method.reward import (
    DEFAULT_HD95_LAMBDA,
    RewardMode,
    reward_threshold_label,
    rewards_from_validation_metrics,
    rewards_from_validation_msd,
)
from bapmos.method.prompt_geometry import build_prompts_for_slice
from bapmos.method.sam_model import FrozenSAMMultiOrgan
from bapmos.losses.kervadec_style import DecoderLossComputer, DecoderLossConfig
from bapmos.losses.loss_policy import force_bapmos_method_kervadec_loss
from bapmos.evaluation.baseline_epoch_monitoring import (
    build_epoch_wandb_log,
    build_test_wandb_log,
    per_organ_wandb_dict,
    summarize_evaluator_metrics,
)

logger = logging.getLogger(__name__)


def _skip_global_test_split(data_root: str | Path) -> bool:
    """Pooled prostate has no global ``test.txt``; per-site tests are exported separately."""
    from bapmos.paths import is_pooled_prostate_training_root

    return is_pooled_prostate_training_root(data_root)

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment,misc]
    _WANDB_AVAILABLE = False


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# SAM ViT-B mask-decoder low-res logits (typically image_encoder.img_size // 4).
DECODER_MASK_SIZE = 256

METRICS_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "train_dice",
    "val_loss",
    "val_dice",
    "val_msd_mm",
    "val_msd_mm_organ_balanced",
    "val_objective_msd_mm",
    "val_hd95_mm",
    "val_dice_organ_balanced",
    "best_val_msd_mm",
    "reward_threshold_mm",
    "reward_mode",
    "ring_mode",
    "ablation_id",
    "lr",
]


class BAPMOSTrainer:
    """End-to-end BAP-MOS trainer (decoder-only SAM + per-organ UCB-Tuned bandit)."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        common = config.get("common", config)
        bandit_cfg = config.get("bandit", {})
        prompt_cfg = config.get("prompt_geometry", {})
        curriculum_cfg = config.get("curriculum", {})
        reward_cfg = config.get("reward", {})
        ablation_cfg = config.get("ablation", {})
        eval_cfg = config.get("evaluation", {})
        wandb_cfg = config.get("wandb", {})

        # Required protocol fields — validate before loading SAM weights.
        if not common.get("data_root"):
            raise ValueError(
                "config common.data_root is required "
                "(use compose --version/--dataset or a complete YAML)."
            )
        if common.get("pixel_spacing") is None:
            raise ValueError(
                "config common.pixel_spacing is required "
                "(e.g. [0.159072, 0.159072] pooled or [1.0, 1.0] PFUS1)."
            )
        if common.get("max_epochs") is None:
            raise ValueError("config common.max_epochs is required")
        if common.get("patience") is None:
            raise ValueError("config common.patience is required")
        if not eval_cfg.get("objective_metric"):
            raise ValueError(
                "config evaluation.objective_metric is required "
                "(protocol default: ptv_kfold_msd)."
            )
        if not common.get("sam_checkpoint"):
            raise ValueError("config common.sam_checkpoint is required")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from bapmos.paths import resolve_model_checkpoint

        sam_ckpt = resolve_model_checkpoint(common["sam_checkpoint"])
        if not sam_ckpt.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {common['sam_checkpoint']!r} "
                "(place under BAPMOS/models/ — see docs/WEIGHTS.md)"
            )
        sam_init = str(common.get("sam_init", "meta")).lower()
        self.model = FrozenSAMMultiOrgan(str(sam_ckpt), init=sam_init).to(self.device)
        self.model.train()

        self.ablation_id: str = str(ablation_cfg.get("id", "bapmos"))
        self.ablation_description: str = str(ablation_cfg.get("description", ""))

        organs = bandit_cfg.get("organs") or common.get("organs")
        if not organs:
            data_root = common.get("data_root")
            if data_root:
                try:
                    from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

                    organs = list(get_baseline_taxonomy_profile(str(data_root)).organ_keys)
                except ValueError:
                    organs = ["rectum", "bladder", "ptv1"]
            else:
                organs = ["rectum", "bladder", "ptv1"]
        self.organ_keys: List[str] = list(organs)
        self.arms = list(bandit_cfg.get("arms", ["box", "point", "both"]))
        self.block_size = int(bandit_cfg.get("block_size_batches", 50))
        self.base_seed = int(config.get("experiment_seed", 42))
        self.experiment_name: str = str(config.get("experiment_name", "default_seed42"))

        self.max_epochs = int(common["max_epochs"])
        self.lr = float(common.get("lr", 1e-4))
        self.patience = int(common["patience"])
        self.val_msd_min_delta = float(common.get("val_msd_min_delta", 0.0))
        self.pixel_spacing = tuple(common["pixel_spacing"])

        self.probe_size = int(bandit_cfg.get("probe_size", 20))
        self.probe_seed = int(bandit_cfg.get("probe_seed", 42))

        self.eval_train_metrics = bool(eval_cfg.get("eval_train_metrics", True))
        # Prefer explicit export CLI over train-time test; default off.
        self.run_test_after_train = bool(eval_cfg.get("run_test_after_train", False))
        self.objective_metric = str(eval_cfg["objective_metric"])
        self.objective_organ_override = eval_cfg.get("objective_organ")
        self.kfold_n = int(eval_cfg.get("kfold_n", 5))
        self.kfold_seed = int(eval_cfg.get("kfold_seed", 42))
        self._objective_organ_label: Optional[str] = None

        self.use_cached_image_embeddings = bool(
            common.get("use_cached_image_embeddings", False)
        )
        self.image_embedding_dir: Optional[str] = common.get("image_embedding_dir")
        self._sam_checkpoint_name = Path(str(common.get("sam_checkpoint", ""))).name
        if self.use_cached_image_embeddings:
            if not self.image_embedding_dir:
                raise ValueError(
                    "common.use_cached_image_embeddings=true requires "
                    "common.image_embedding_dir (precomputed SAM .pt cache)."
                )
            logger.info(
                "use_cached_image_embeddings=True: skipping SAM image_encoder; "
                "cache dir=%s | trainer checkpoint basename=%s "
                "(Meta SAM vs MedSAM caches are not interchangeable)",
                self.image_embedding_dir,
                self._sam_checkpoint_name,
            )

        self.wandb_enabled = bool(wandb_cfg.get("enabled", True)) and _WANDB_AVAILABLE
        # Default project name; configs may override.
        self.wandb_project = str(wandb_cfg.get("project", "bapmos"))
        self.wandb_entity = wandb_cfg.get("entity")
        if bool(wandb_cfg.get("enabled", True)) and not _WANDB_AVAILABLE:
            raise RuntimeError(
                "wandb.enabled is true but the wandb package is not installed. "
                "Install with: pip install -r requirements.txt "
                "(or pass --no-wandb / set wandb.enabled: false)."
            )

        self.num_pos = int(prompt_cfg.get("num_pos", 1))
        self.num_neg = int(prompt_cfg.get("num_neg", 3))
        self.r_min = int(prompt_cfg.get("r_min", 6))
        self.alpha = float(prompt_cfg.get("alpha", 0.1))
        self.box_jitter_px = int(prompt_cfg.get("box_jitter_px", 2))
        self.ring_mode: str = str(prompt_cfg.get("ring_mode", "scale_organ"))
        self.ring_width_fixed: int = int(prompt_cfg.get("ring_width", 20))

        self.reward_mode: RewardMode = str(
            reward_cfg.get("mode", curriculum_cfg.get("mode", "composite_fixed_clip"))
        )  # type: ignore[assignment]
        if self.reward_mode not in ("fixed_clip", "curriculum", "composite_fixed_clip"):
            raise ValueError(
                "reward.mode must be 'fixed_clip', 'curriculum', or 'composite_fixed_clip', "
                f"got {self.reward_mode!r}"
            )
        self.clip_max_mm = float(
            reward_cfg.get("clip_max_mm", curriculum_cfg.get("clip_max_mm", 20.0))
        )
        self.hd95_lambda = float(reward_cfg.get("hd95_lambda", DEFAULT_HD95_LAMBDA))

        self.tau_max = float(curriculum_cfg.get("tau_max_mm", 15.0))
        self.tau_min = float(curriculum_cfg.get("tau_min_mm", 0.1))
        self.decay_fraction = float(curriculum_cfg.get("decay_fraction", 0.8))
        self.gamma = compute_gamma(
            self.tau_max, self.tau_min, self.max_epochs, self.decay_fraction
        )

        bandit_memory = str(bandit_cfg.get("memory", "sliding"))
        if bandit_memory not in ("sliding", "cumulative"):
            raise ValueError(
                f"bandit.memory must be 'sliding' or 'cumulative', got {bandit_memory!r}"
            )
        self.policy = PerOrganBanditPolicy(
            organs=self.organ_keys,
            arms=self.arms,
            window_size=int(bandit_cfg.get("window_size", 50)),
            warmup_blocks_per_arm=int(bandit_cfg.get("warmup_blocks_per_arm", 10)),
            min_pulls_per_arm=int(bandit_cfg.get("min_pulls_per_arm", 5)),
            seed=self.base_seed,
            memory=bandit_memory,  # type: ignore[arg-type]
        )
        self.scheduler = BlockScheduler(self.policy, block_size_batches=self.block_size)

        self.optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=self.lr)
        self.organ_to_class: Dict[str, int] = {}
        self.num_classes = 4
        self.multiclass_eval_mapping: Dict[int, str] = {}
        self.evaluator_organ_labels: Tuple[str, ...] = ()
        self.epoch = 0
        # BAP-MOS method (SAM or MedSAM init): always Kervadec-style decoder loss.
        force_bapmos_method_kervadec_loss(config)
        self.decoder_loss = DecoderLossComputer(
            DecoderLossConfig.from_training_config(config, max_epochs=self.max_epochs)
        )
        if self.decoder_loss.mode != "kervadec":
            raise RuntimeError(
                "BAP-MOS method requires training.loss_mode='kervadec' "
                f"(got {self.decoder_loss.mode!r})"
            )
        self.best_val_msd = float("inf")
        self._epochs_without_improve = 0
        self.run_dir: Optional[Path] = None
        self._val_dataset_ref = None
        self._validation_probe_indices: Optional[np.ndarray] = None
        self._block_count = 0

        logger.info(
            "Ablation %s | reward=%s (clip=%.2f mm%s) | ring_mode=%s (width=%s) | "
            "decoder_loss=%s | sam_init=%s",
            self.ablation_id,
            self.reward_mode,
            self.clip_max_mm,
            (
                f", hd95_lambda={self.hd95_lambda:.1f}"
                if self.reward_mode == "composite_fixed_clip"
                else ""
            ),
            self.ring_mode,
            self.ring_width_fixed if self.ring_mode == "fixed" else f"scale r_min={self.r_min} alpha={self.alpha}",
            self.decoder_loss.mode,
            sam_init,
        )
        if self.decoder_loss.mode == "kervadec":
            kcfg = self.decoder_loss.config.kervadec
            logger.info(
                "Kervadec schedule: alpha %.3f → %.3f (%s over %d epochs)",
                kcfg.alpha_start,
                kcfg.alpha_end,
                kcfg.alpha_schedule,
                self.max_epochs,
            )

    def set_data_taxonomy(
        self,
        organ_to_class: Dict[str, int],
        num_classes: int,
        *,
        data_root: Optional[str] = None,
    ) -> None:
        self.organ_to_class = organ_to_class
        self.num_classes = num_classes
        if data_root is not None:
            tax = resolve_taxonomy(data_root)
            self.multiclass_eval_mapping = dict(tax.multiclass_eval_mapping)
            self.evaluator_organ_labels = tuple(tax.evaluator_organ_labels)
            self.pixel_spacing = tuple(tax.pixel_spacing_mm)
            logger.info(
                "Taxonomy %s | evaluator organs=%s | spacing=%s",
                tax.taxonomy_name,
                list(self.evaluator_organ_labels),
                self.pixel_spacing,
            )
        if self.objective_metric == "ptv_kfold_msd":
            self._objective_organ_label = resolve_ptv_evaluator_organ(
                self.evaluator_organ_labels,
                override=(
                    str(self.objective_organ_override)
                    if self.objective_organ_override
                    else None
                ),
            )
            logger.info(
                "Validation objective: %s on %s (k=%d, seed=%d)",
                self.objective_metric,
                self._objective_organ_label,
                self.kfold_n,
                self.kfold_seed,
            )

    def init_val_probe(self, val_dataset) -> None:
        """Fixed deterministic val indices for block-end bandit rewards."""
        self._val_dataset_ref = val_dataset
        n = len(val_dataset)
        if n <= 0:
            raise ValueError(
                "Validation dataset is empty; cannot build a bandit probe or train."
            )
        probe_size = min(self.probe_size, n)
        rng = np.random.default_rng(self.probe_seed)
        self._validation_probe_indices = rng.choice(n, probe_size, replace=False)
        logger.info(
            "Bandit val probe: %d / %d slices (seed=%d)",
            probe_size,
            n,
            self.probe_seed,
        )

    def _decoder_mask_size(self) -> int:
        """Low-res decoder spatial size (SAM ViT-B: img_size // 4)."""
        img_size = int(getattr(self.model, "img_size", DECODER_MASK_SIZE * 4))
        return max(1, img_size // 4)

    @staticmethod
    def _as_numpy_hwc(image: Any) -> np.ndarray:
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            # CHW → HWC
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _as_numpy_mask(mask: Any) -> np.ndarray:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        return np.asarray(mask)

    def _arms_for_eval(self, split: str) -> Dict[str, str]:
        """Epoch/test eval uses best arms per organ; block probe uses scheduler arms."""
        if split == "block":
            return self.scheduler.current_arms or self.policy.get_best_arms_per_organ()
        return self.policy.get_best_arms_per_organ()

    def loss_fn(
        self,
        logits: torch.Tensor,
        gt_multi: torch.Tensor,
        dist_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decoder loss: CE+Dice (default) or scheduled Kervadec CE+Dice + boundary."""
        return self.decoder_loss(
            logits,
            gt_multi,
            self.num_classes,
            epoch=self.epoch,
            dist_maps=dist_maps,
        )

    def _prepare_gt_tensor(self, mask: np.ndarray) -> torch.Tensor:
        import cv2

        size = self._decoder_mask_size()
        gt = cv2.resize(
            mask.astype(np.uint8),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        return torch.as_tensor(gt, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0)

    def _prepare_boundary_dist_maps(self, gt: torch.Tensor) -> Optional[torch.Tensor]:
        """Precompute signed φ_G on resized GT when using Kervadec boundary loss."""
        if self.decoder_loss.mode != "kervadec":
            return None
        from bapmos.losses.kervadec_style import multiclass_distance_maps

        label_hw = gt.detach().squeeze().long().cpu().numpy().astype(np.int64, copy=False)
        dist_np = multiclass_distance_maps(label_hw, self.num_classes)
        return torch.as_tensor(dist_np, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _compose_multiclass_logits_from_binary_organs(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Fill background logit from 1 − max(foreground sigmoid); organs stay independent.

        Overlapping high organ logits compete via softmax downstream.
        """
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        if organ_probs.numel() > 0:
            organ_union = organ_probs.max(dim=1, keepdim=True)[0]
            bg = 1.0 - organ_union
            logits[:, 0:1, :, :] = torch.logit(bg.clamp(1e-7, 1 - 1e-7))
        return logits

    def _tensor_from_cached_embedding_pack(
        self,
        pack: dict,
        source: str,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Validate cached pack and move embedding to ``self.device`` (float32)."""
        if "image_embedding" not in pack or "original_size" not in pack:
            raise KeyError(f"Invalid embedding pack (missing keys): {source}")
        cached_ckpt = pack.get("checkpoint")
        if cached_ckpt and self._sam_checkpoint_name and cached_ckpt != self._sam_checkpoint_name:
            raise ValueError(
                f"Cached embedding checkpoint mismatch for {source}: "
                f"cache reports {cached_ckpt!r}, trainer uses {self._sam_checkpoint_name!r}. "
                "Recompute embeddings with the same SAM checkpoint (do not mix Meta SAM vs MedSAM)."
            )
        emb = pack["image_embedding"]
        if emb.dtype in (torch.float16, torch.bfloat16):
            emb = emb.float()
        emb = emb.to(self.device)
        orig = pack["original_size"]
        original_size = (int(orig[0]), int(orig[1]))
        return emb, original_size

    def _load_cached_sam_embedding(self, embedding_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        path = Path(embedding_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cached SAM embedding not found: {path}")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        return self._tensor_from_cached_embedding_pack(pack, str(path))

    def _prepare_image_embedding(
        self,
        image_rgb: np.ndarray,
        sample_id: str,
        embedding_path: Optional[str] = None,
        cached_embedding_pack: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.use_cached_image_embeddings:
            if cached_embedding_pack is not None:
                img_emb, original_size = self._tensor_from_cached_embedding_pack(
                    cached_embedding_pack, str(sample_id)
                )
            elif embedding_path and Path(embedding_path).is_file():
                img_emb, original_size = self._load_cached_sam_embedding(embedding_path)
            else:
                # Test export / probe paths may not pass packs; encode on the fly.
                if embedding_path:
                    logger.warning(
                        "Cached embedding missing at %s for %s; encoding on the fly",
                        embedding_path,
                        sample_id,
                    )
                else:
                    logger.debug(
                        "use_cached_image_embeddings=True but no pack/path for %s; "
                        "encoding on the fly",
                        sample_id,
                    )
                return self.model.encode_image(image_rgb, self.device)
            img_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
            if original_size != img_hw:
                logger.warning(
                    "Cached original_size %s does not match loaded image shape %s for %s",
                    original_size,
                    img_hw,
                    sample_id,
                )
            return img_emb, original_size
        return self.model.encode_image(image_rgb, self.device)

    def _forward_sample(
        self,
        image_rgb: Union[np.ndarray, torch.Tensor],
        mask_multi: Union[np.ndarray, torch.Tensor],
        sample_id: str,
        arms: Dict[str, str],
        train: bool,
        *,
        embedding_path: Optional[str] = None,
        cached_embedding_pack: Optional[dict] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-organ decode → stacked multi-class logits (1, C, H, W) at decoder resolution."""
        image_rgb = self._as_numpy_hwc(image_rgb)
        mask_multi = self._as_numpy_mask(mask_multi)
        prompts = build_prompts_for_slice(
            mask_multi,
            self.organ_to_class,
            arms,
            sample_id,
            self.epoch,
            self.base_seed,
            train=train,
            box_jitter_px=self.box_jitter_px if train else 0,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
            r_min=self.r_min,
            alpha=self.alpha,
            ring_mode=self.ring_mode,
            ring_width_fixed=self.ring_width_fixed,
        )
        if not prompts:
            return None

        img_emb, original_size = self._prepare_image_embedding(
            image_rgb,
            sample_id,
            embedding_path=embedding_path,
            cached_embedding_pack=cached_embedding_pack,
        )
        gt = self._prepare_gt_tensor(mask_multi)

        size = self._decoder_mask_size()
        logits = torch.zeros(
            1, self.num_classes, size, size, device=self.device, dtype=torch.float32
        )
        for organ_key, prompt in prompts.items():
            class_id = self.organ_to_class[organ_key]
            low_res = self.model.forward_organ(img_emb, prompt, original_size, self.device)
            logits[0, class_id] = low_res.squeeze(0).squeeze(0)

        logits = self._compose_multiclass_logits_from_binary_organs(logits)
        return logits, gt

    def _bandit_rewards(self, organ_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if self.reward_mode == "composite_fixed_clip":
            return rewards_from_validation_metrics(
                organ_metrics,  # type: ignore[arg-type]
                clip_max_mm=self.clip_max_mm,
                hd95_lambda=self.hd95_lambda,
            )
        organ_msds = {
            k: float(v["msd_mm"]) for k, v in organ_metrics.items() if np.isfinite(v["msd_mm"])
        }
        return rewards_from_validation_msd(
            organ_msds,
            self.epoch,
            reward_mode=self.reward_mode,
            clip_max_mm=self.clip_max_mm,
            tau_max=self.tau_max,
            tau_min=self.tau_min,
            max_epochs=self.max_epochs,
            decay_fraction=self.decay_fraction,
            gamma=self.gamma,
        )

    def _current_reward_threshold(self) -> float:
        return reward_threshold_label(
            self.reward_mode,
            self.epoch,
            clip_max_mm=self.clip_max_mm,
            tau_max=self.tau_max,
            tau_min=self.tau_min,
            max_epochs=self.max_epochs,
            decay_fraction=self.decay_fraction,
            gamma=self.gamma,
        )

    @staticmethod
    def _setup_wandb_metric_definitions() -> None:
        """Epoch charts share ``epoch``; block charts use ``block_step``.

        Block metrics live under ``bandit_block/*`` (not ``bandit/block/*``) so they do
        not overlap the epoch-level ``bandit/*`` glob / step metric.
        """
        if wandb is None:
            return
        wandb.define_metric("epoch")
        wandb.define_metric("block_step")
        wandb.define_metric("bandit_block/*", step_metric="block_step")
        for key in (
            "train/*",
            "val/*",
            "lr",
            "time/*",
            "reward/*",
            "prompts/*",
            "test/*",
            "bandit/*",
        ):
            wandb.define_metric(key, step_metric="epoch")

    def _wandb_log_epoch(self, epoch: int, log_dict: Dict[str, Any]) -> None:
        if not self.wandb_enabled or wandb is None:
            return
        payload = dict(log_dict)
        payload["epoch"] = int(epoch)
        wandb.log(payload)

    def _wandb_log_block(self, log_dict: Dict[str, Any]) -> None:
        if not self.wandb_enabled or wandb is None:
            return
        payload = dict(log_dict)
        payload["block_step"] = int(self._block_count)
        wandb.log(payload)

    def train_epoch(self, train_loader, val_loader) -> Tuple[float, Dict[str, int]]:
        """One training epoch with in-epoch bandit block transitions.

        Block lifecycle invariant (``BlockScheduler``):
        - ``needs_new_block()`` iff ``current_arms`` is empty (no open block).
        - ``begin_block`` freezes arms and resets the in-block counter.
        - ``register_batch()`` advances by one dataloader step (not sample count);
          returns True when ``batches_in_block >= block_size_batches``.
        - ``end_block`` commits rewards and clears arms → ``needs_new_block()`` again.
        Mid-epoch ``begin_block`` after ``end_block`` is intentional so the next
        batches immediately open a new block without waiting for the next epoch.
        ``val_loader`` is unused here; full-split val runs in ``train()`` after the epoch.
        """
        _ = val_loader
        if self.scheduler.needs_new_block():
            present = self.organ_keys
            arms = self.scheduler.begin_block(present)
            logger.info("New block arms: %s", arms)

        total_loss = 0.0
        n = 0
        train_prompt_counts = empty_prompt_counts()
        for batch in train_loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]
            emb_paths = batch.get("embedding_path")
            cached_packs = batch.get("sam_cached_pack")
            arms = self.scheduler.current_arms
            if not arms:
                raise RuntimeError(
                    "scheduler.current_arms is empty during train_epoch; "
                    "begin_block must run before the first batch."
                )
            bs = len(images)

            for i in range(bs):
                image_i = self._as_numpy_hwc(images[i])
                mask_i = self._as_numpy_mask(masks[i])
                ep = emb_paths[i] if emb_paths is not None else None
                cp = cached_packs[i] if cached_packs is not None else None
                out = self._forward_sample(
                    image_i,
                    mask_i,
                    str(filenames[i]),
                    arms,
                    train=True,
                    embedding_path=ep,
                    cached_embedding_pack=cp,
                )
                if out is None:
                    continue
                # Count arms only for organs present on this slice (realized prompts).
                realized = {
                    k: arms[k]
                    for k, cid in self.organ_to_class.items()
                    if k in arms and int((mask_i == cid).sum()) > 0
                }
                merge_prompt_counts(train_prompt_counts, realized)
                logits, gt = out
                loss = self.loss_fn(
                    logits, gt, dist_maps=self._prepare_boundary_dist_maps(gt)
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.mask_decoder.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                n += 1

            if self.scheduler.register_batch():
                if self._val_dataset_ref is None or self._validation_probe_indices is None:
                    raise RuntimeError("Call init_val_probe(val_dataset) before training.")
                organ_metrics = validate_block_per_organ_probe_metrics(
                    self,
                    self._val_dataset_ref,
                    self._validation_probe_indices,
                )
                organ_msds = {
                    k: (
                        organ_metrics[k]["msd_mm"]
                        if k in organ_metrics
                        else float("inf")
                    )
                    for k in self.organ_keys
                }
                rewards = self._bandit_rewards(organ_metrics)
                self.scheduler.end_block(rewards)
                self._block_count += 1
                logger.info(
                    "Block end epoch=%d metrics=%s rewards=%s",
                    self.epoch,
                    {
                        k: {
                            "dice": round(v["dice"], 4),
                            "msd": round(v["msd_mm"], 4),
                            "hd95": round(v["hd95_mm"], 4),
                        }
                        for k, v in organ_metrics.items()
                    },
                    {k: round(v, 4) for k, v in rewards.items()},
                )
                self._wandb_log_block(
                    build_block_wandb_dict(
                        organ_msds,
                        rewards,
                        self._block_count,
                        organ_metrics=organ_metrics,
                    )
                )
            if self.scheduler.needs_new_block():
                self.scheduler.begin_block(self.organ_keys)

        return total_loss / max(n, 1), train_prompt_counts

    def save_checkpoint(self, path: Path, val_msd: float, *, tag: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "format_version": 2,
            "epoch_index": int(self.epoch),
            "mask_decoder": self.model.mask_decoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "bandit_state": self.policy.export_state(),
            "scheduler_state": self.scheduler.export_state(),
            "block_count": int(self._block_count),
            "bandit_stats": self.policy.get_all_statistics(),
            "val_msd": val_msd,
            "best_val_msd_mm": self.best_val_msd,
            "epochs_without_improvement": int(self._epochs_without_improve),
            "config": self.cfg,
            "tag": tag,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def apply_checkpoint(self, ckpt: dict, *, restore_rng: bool = True) -> int:
        """Restore trainable state. Returns 0-based epoch index to run next."""
        fv = ckpt.get("format_version")
        if fv is not None and int(fv) != 2:
            raise ValueError(
                f"Resume requires format_version=2 checkpoints; got {fv!r}. "
                "Re-run training once to upgrade last_checkpoint.pth."
            )
        self.model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("bandit_state"):
            self.policy.load_state(ckpt["bandit_state"])
        if ckpt.get("scheduler_state"):
            self.scheduler.load_state(ckpt["scheduler_state"])
        self._block_count = int(ckpt.get("block_count", 0))
        best = ckpt.get("best_val_msd_mm")
        if best is None:
            best = ckpt.get("best_val_msd")
        self.best_val_msd = float(best) if best is not None else float("inf")
        self._epochs_without_improve = int(ckpt.get("epochs_without_improvement", 0))
        rs = ckpt.get("rng_state")
        if restore_rng and rs:
            torch_state = rs["torch"]
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            else:
                torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
            torch.set_rng_state(torch_state)
            np.random.set_state(rs["numpy"])
            random.setstate(rs["random"])
            cuda_states = rs.get("cuda")
            if cuda_states is not None and torch.cuda.is_available():
                restored = []
                for i, state in enumerate(cuda_states):
                    if i >= torch.cuda.device_count():
                        break
                    if not isinstance(state, torch.Tensor):
                        state = torch.as_tensor(state, dtype=torch.uint8)
                    else:
                        state = state.detach().cpu().to(dtype=torch.uint8)
                    # torch.cuda.set_rng_state_all expects CPU ByteTensors.
                    restored.append(state.contiguous())
                if restored:
                    torch.cuda.set_rng_state_all(restored)
        ei = int(ckpt.get("epoch_index", ckpt.get("epoch", -1)))
        if ei < 0:
            raise ValueError("Checkpoint missing epoch_index.")
        start = ei + 1
        logger.info(
            "[resume] Loaded checkpoint after epoch index %d; continuing from %d | best_val_msd_mm=%.4f",
            ei,
            start,
            self.best_val_msd,
        )
        return start

    def train(
        self,
        train_loader,
        val_loader,
        run_dir: Path,
        *,
        test_loader=None,
        skip_test: bool = False,
        start_epoch: int = 0,
    ) -> None:
        self.run_dir = run_dir
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS).writeheader()

        if start_epoch >= self.max_epochs:
            logger.info(
                "[resume] start_epoch=%d >= max_epochs=%d; skipping training loop.",
                start_epoch,
                self.max_epochs,
            )
        for epoch in range(start_epoch, self.max_epochs):
            self.epoch = epoch
            t0 = time.time()
            train_loss, train_prompt_counts = self.train_epoch(train_loader, val_loader)

            eval_arms = self._arms_for_eval("val")

            train_dice_val: Optional[float] = None
            train_metrics: Optional[Dict[str, Optional[float]]] = None
            val_metrics: Optional[Dict[str, Optional[float]]] = None
            tr_ev = None
            val_prompt_counts = empty_prompt_counts()
            if self.eval_train_metrics:
                tr_loss, tr_dice, tr_summ, tr_ev, tr_prompt_counts = run_split_metrics_pass(
                    self, train_loader, split="train", arms=eval_arms
                )
                train_loss = tr_loss
                train_dice_val = tr_dice
                train_metrics = summarize_evaluator_metrics(tr_ev)
                train_prompt_counts = tr_prompt_counts

            val_loss, val_dice, val_summ, val_ev, val_prompt_counts = run_split_metrics_pass(
                self, val_loader, split="val", arms=eval_arms
            )
            val_metrics = summarize_evaluator_metrics(val_ev)
            val_msd_organ_balanced = val_summ.get("val_msd")
            val_msd = validation_objective_msd(
                val_ev,
                metric=self.objective_metric,
                organ_name=self._objective_organ_label,
                kfold_n=self.kfold_n,
                kfold_seed=self.kfold_seed,
            )
            if val_msd is None:
                val_msd = val_msd_organ_balanced
            val_hd95 = val_summ.get("val_hd95")
            val_rep_dice = val_summ.get("val_dice")
            thresh = self._current_reward_threshold()
            lr = self.optimizer.param_groups[0]["lr"]
            epoch_no = epoch + 1

            self.save_checkpoint(run_dir / "last_checkpoint.pth", float(val_msd or self.best_val_msd))

            if val_msd is not None and val_msd < self.best_val_msd - self.val_msd_min_delta:
                self.best_val_msd = float(val_msd)
                self.save_checkpoint(run_dir / "best_checkpoint.pth", self.best_val_msd, tag="best")
                self._epochs_without_improve = 0
                logger.info("New best val_msd_mm: %.4f", self.best_val_msd)
            else:
                self._epochs_without_improve += 1
                logger.info(
                    "No val MSD improvement (%d/%d)",
                    self._epochs_without_improve,
                    self.patience,
                )

            row = {
                "epoch": epoch_no,
                "train_loss": train_loss,
                "train_dice": train_dice_val if train_dice_val is not None else "",
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_msd_mm": val_msd if val_msd is not None else "",
                "val_msd_mm_organ_balanced": (
                    val_msd_organ_balanced if val_msd_organ_balanced is not None else ""
                ),
                "val_objective_msd_mm": val_msd if val_msd is not None else "",
                "val_hd95_mm": val_hd95 if val_hd95 is not None else "",
                "val_dice_organ_balanced": val_rep_dice if val_rep_dice is not None else "",
                "best_val_msd_mm": self.best_val_msd,
                "reward_threshold_mm": thresh,
                "reward_mode": self.reward_mode,
                "ring_mode": self.ring_mode,
                "ablation_id": self.ablation_id,
                "lr": lr,
            }
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDS).writerow(row)

            extra: Dict[str, Any] = {
                "reward/threshold_mm": thresh,
                "prompts/train_box_count": train_prompt_counts.get("box", 0),
                "prompts/train_point_count": train_prompt_counts.get("point", 0),
                "prompts/train_both_count": train_prompt_counts.get("both", 0),
                "prompts/val_box_count": val_prompt_counts.get("box", 0),
                "prompts/val_point_count": val_prompt_counts.get("point", 0),
                "prompts/val_both_count": val_prompt_counts.get("both", 0),
                "best_val_msd_mm": self.best_val_msd,
            }
            if val_msd is not None:
                extra["val/objective_msd_mm"] = float(val_msd)
            extra.update(
                build_bandit_epoch_wandb_dict(self.policy, total_blocks=self._block_count)
            )

            log_dict = build_epoch_wandb_log(
                epoch=epoch_no,
                train_loss=train_loss,
                train_dice=float(train_dice_val if train_dice_val is not None else 0.0),
                val_loss=val_loss,
                val_dice=val_dice,
                lr=lr,
                time_epoch_s=time.time() - t0,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_per_organ_wandb=per_organ_wandb_dict(
                    val_ev, list(self.evaluator_organ_labels), "val"
                ),
                train_per_organ_wandb=(
                    per_organ_wandb_dict(tr_ev, list(self.evaluator_organ_labels), "train")
                    if self.eval_train_metrics
                    else None
                ),
                extra=extra,
            )
            self._wandb_log_epoch(epoch_no, log_dict)

            logger.info(
                "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_dice=%.4f val_msd=%s val_hd95=%s",
                epoch + 1,
                self.max_epochs,
                train_loss,
                val_loss,
                val_dice,
                f"{val_msd:.4f}" if val_msd is not None else "N/A",
                f"{val_hd95:.4f}" if val_hd95 is not None else "N/A",
            )

            if self._epochs_without_improve >= self.patience:
                logger.info("Early stopping.")
                break

        if (
            not skip_test
            and self.run_test_after_train
            and test_loader is not None
            and (run_dir / "best_checkpoint.pth").is_file()
        ):
            from bapmos.inference_output.protocol import is_primary_inference_seed_run

            test_out_dir = run_dir / "test_metrics"
            method_slug = method_slug_from_config(self.cfg)
            # Canonical shared inference/output mirrors: primary seed-42 only.
            # Replicate runs still write isolated metrics under this run_dir.
            canon_dir = None
            if is_primary_inference_seed_run(cfg=self.cfg, run_name=run_dir.name):
                canon_dir = bapmos_method_test_output_dir_from_config(self.cfg)
            # Metrics/JSON only — stratified PNG/PDF protocol is via
            # ``python -m bapmos.method.run_test_inference`` (seed-42 primary).
            logger.info(
                "Running test split metrics → %s (canonical: %s). "
                "Stratified inference_output panels require run_test_inference.",
                test_out_dir,
                canon_dir if canon_dir is not None else "(skipped; non-primary seed)",
            )
            test_out = export_test_split_metrics(
                self,
                test_loader,
                test_out_dir,
                canonical_method_dir=canon_dir,
                method_slug=method_slug,
                cfg=self.cfg,
            )
            test_evaluator = test_out.pop("_evaluator", None)
            test_log = build_test_wandb_log(
                test_out,
                evaluator=test_evaluator,
                organ_labels=list(self.evaluator_organ_labels),
            )
            if wandb is not None and self.wandb_enabled and test_log:
                wandb.log(test_log)
                wandb.summary.update(
                    {k.replace("/", "_"): v for k, v in test_log.items()}
                )
            logger.info("Test results: %s", test_out)


def _overlay_runtime_speed_settings(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Apply val-only eval / embedding-cache knobs from current YAML onto a resumed ckpt config.

    Resume replaces the whole config from the checkpoint; without this overlay, speedups
    in version.yaml would be ignored until a fresh run.
    """
    src_eval = src.get("evaluation") or {}
    dst_eval = dst.setdefault("evaluation", {})
    if "eval_train_metrics" in src_eval:
        dst_eval["eval_train_metrics"] = bool(src_eval["eval_train_metrics"])

    src_common = src.get("common") or {}
    dst_common = dst.setdefault("common", {})
    for key in ("use_cached_image_embeddings", "image_embedding_dir", "num_workers"):
        if key in src_common:
            dst_common[key] = src_common[key]


def run_training_job(
    cfg: Dict[str, Any],
    run_root: Optional[str] = None,
    *,
    skip_test: bool = False,
    run_name: Optional[str] = None,
    resume_ckpt: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Train one resolved config (experiment overrides already applied).

    On resume, the checkpoint's ``config`` (including ``run_dir``) is restored so
    training continues in the original run folder. Pass ``cli_overrides`` to
    re-apply flags such as ``no_wandb`` / ``skip_test`` after that replacement.
    Runtime speed settings (``eval_train_metrics``, cached embeddings) are overlaid
    from the freshly composed YAML so resume picks up operator speedups.
    """
    fresh_cfg = copy.deepcopy(cfg)
    ckpt_obj: Optional[dict] = None
    if resume_ckpt is not None:
        resume_ckpt = Path(resume_ckpt)
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        ckpt_obj = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if "config" not in ckpt_obj:
            raise KeyError(
                f"Resume checkpoint missing 'config' key: {resume_ckpt}. "
                "Re-train once to produce a format_version=2 checkpoint."
            )
        cfg = copy.deepcopy(ckpt_obj["config"])
        _overlay_runtime_speed_settings(cfg, fresh_cfg)
        if cli_overrides:
            if cli_overrides.get("no_wandb"):
                cfg.setdefault("wandb", {})["enabled"] = False
            if cli_overrides.get("skip_test"):
                cfg.setdefault("evaluation", {})["run_test_after_train"] = False

    common = cfg.get("common", {})
    rel_root = run_root or cfg.get("run_root") or run_directory_layout(cfg)
    stable = run_name or cfg.get("run_name")
    if ckpt_obj is not None:
        if "run_dir" not in cfg:
            raise KeyError(
                f"Resumed checkpoint config missing run_dir: {resume_ckpt}. "
                "Cannot continue in the original run folder."
            )
        run_dir = Path(cfg["run_dir"])
        run_name_final = str(cfg.get("run_name", stable or "run"))
        logger.info(
            "[resume] Continuing original run_dir=%s from checkpoint=%s",
            run_dir,
            resume_ckpt,
        )
    elif stable:
        run_name_final = str(stable)
        run_dir = resolve_path(rel_root) / run_name_final
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["run_name"] = run_name_final
        cfg["run_root"] = str(rel_root)
    else:
        run_name_final = (
            f"{cfg.get('experiment_name', 'run')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir = resolve_path(rel_root) / run_name_final
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["run_name"] = run_name_final
        cfg["run_root"] = str(rel_root)

    setup_logging(run_dir / "logs")

    train_seed = int(cfg.get("experiment_seed", common.get("seed", 42)))
    from bapmos.method.bapmos_seeds import seed_everything

    # Fresh runs: seed before model/dataloader construction. Resume restores RNG
    # from the checkpoint after load (apply_checkpoint).
    if ckpt_obj is None:
        seed_everything(train_seed, deterministic=False)
        logger.info("seed_everything(%d) for fresh run", train_seed)

    trainer = BAPMOSTrainer(cfg)

    emb_dir = None
    if bool(common.get("use_cached_image_embeddings", False)):
        emb_dir = common.get("image_embedding_dir")
        if not emb_dir:
            raise ValueError(
                "use_cached_image_embeddings requires common.image_embedding_dir"
            )
        emb_dir = str(resolve_path(str(emb_dir)))

    train_loader, val_loader, test_loader, val_ds, organ_to_class, num_classes = (
        build_dataloaders(
            common["data_root"],
            splits_subdir=common.get("splits_subdir", "splits_stratified"),
            batch_size=int(common.get("batch_size", 1)),
            num_workers=int(common.get("num_workers", 0)),
            organ_keys=list(cfg.get("bandit", {}).get("organs", [])),
            include_test=bool(cfg.get("evaluation", {}).get("run_test_after_train", False))
            and not _skip_global_test_split(common["data_root"]),
            seed=int(cfg.get("experiment_seed", common.get("seed", 42))),
            image_embedding_dir=emb_dir,
        )
    )
    if emb_dir:
        s0 = val_ds[0]
        logger.info(
            "Cached SAM embeddings: dir=%s | example_path=%s | has_pack=%s",
            emb_dir,
            s0.get("embedding_path"),
            "sam_cached_pack" in s0,
        )
    trainer.set_data_taxonomy(
        organ_to_class, num_classes, data_root=common["data_root"]
    )
    trainer.init_val_probe(val_ds)

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    start_epoch = 0
    if ckpt_obj is not None:
        start_epoch = trainer.apply_checkpoint(ckpt_obj)
        logger.info(
            "[resume] Applied checkpoint %s → start_epoch=%d | best_val_msd_mm=%.4f | run_dir=%s",
            resume_ckpt,
            start_epoch,
            trainer.best_val_msd,
            run_dir,
        )

    if trainer.wandb_enabled and wandb is not None:
        wandb_cfg = cfg.get("wandb", {})
        init_kw: Dict[str, Any] = {
            "project": trainer.wandb_project,
            "name": str(wandb_cfg.get("name") or run_name_final),
            "config": cfg,
            "dir": str(run_dir),
        }
        if trainer.wandb_entity:
            init_kw["entity"] = trainer.wandb_entity
        tags = wandb_cfg.get("tags")
        if tags:
            init_kw["tags"] = list(tags)
        group = wandb_cfg.get("group")
        if group:
            init_kw["group"] = str(group)
        if ckpt_obj is not None:
            init_kw["resume"] = "allow"
        wandb.init(**init_kw)
        trainer._setup_wandb_metric_definitions()
        logger.info("WandB project=%s run=%s", trainer.wandb_project, run_name_final)

    logger.info(
        "Run directory: %s | version=%s dataset=%s experiment=%s seed=%d",
        run_dir,
        cfg.get("ablation", {}).get("id"),
        cfg.get("meta", {}).get("dataset", cfg.get("dataset", {}).get("name")),
        cfg.get("experiment_name"),
        cfg.get("experiment_seed"),
    )
    try:
        trainer.train(
            train_loader,
            val_loader,
            run_dir,
            test_loader=test_loader,
            skip_test=skip_test,
            start_epoch=start_epoch,
        )
        if bool(cfg.get("evaluation", {}).get("cleanup_checkpoints_after_train", False)):
            from bapmos.hpo.objective import delete_trial_checkpoints

            n = delete_trial_checkpoints(run_dir)
            if n:
                logger.info(
                    "cleanup_checkpoints_after_train: deleted %d checkpoint(s) under %s",
                    n,
                    run_dir,
                )
    finally:
        if trainer.wandb_enabled and wandb is not None:
            wandb.finish()
    return run_dir


def main(argv: Optional[List[str]] = None) -> None:
    from bapmos.method.bapmos_seeds import apply_training_seed_env
    from bapmos.method.config_utils import (
        compose_config,
        list_experiments,
        load_composed_config,
        resolve_experiment,
    )

    parser = argparse.ArgumentParser(
        description="BAP-MOS trainer (compose version × dataset × experiment)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Advanced: load a fully specified YAML as-is (skips site common + selected/). "
            "Prefer --version + --dataset for production."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Suite version key (e.g. bapmos, bapmos_sam). Prefer with --dataset so selected/ merges.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset: pooled, pfus1, case1, case2, simulation (with --version)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name from the version YAML experiments list",
    )
    parser.add_argument(
        "--run-all-experiments",
        action="store_true",
        help="Run every experiment entry in the version YAML for this dataset",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Print experiment names for --version (+ --dataset) and exit",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Override checkpoint root (default: runs/<bundle>/Optimization/bapmos_<ablation_id>/)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip held-out test evaluation after training",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Stable run directory name (no timestamp). Required for Slurm auto-resume; "
            "typically the experiment name, e.g. clip10_seed42."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from last_checkpoint.pth: path, or 'auto' with --run-name under "
            "<run_root>/<run-name>/last_checkpoint.pth. Resume continues in the "
            "checkpoint's original run_dir (CLI --run-root/--run-name are ignored for "
            "the destination folder)."
        ),
    )
    args = parser.parse_args(argv)

    if args.run_all_experiments and args.experiment:
        parser.error("Do not combine --run-all-experiments with --experiment.")
    if (args.resume or "").strip().lower() == "auto" and not (
        (args.run_name or "").strip() or args.experiment or args.run_all_experiments
    ):
        parser.error(
            "--resume auto requires --run-name and/or --experiment "
            "(or --run-all-experiments, which sets run-name per experiment)."
        )

    if args.list_experiments:
        if args.config:
            base = load_yaml_config(args.config)
        else:
            if not args.version or not args.dataset:
                parser.error("--list-experiments requires --version and --dataset (or --config)")
            base = compose_config(args.version, args.dataset)
        for exp in list_experiments(base):
            print(exp.get("name"))
        return

    def _apply_cli_flags(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
        cfg_out = copy.deepcopy(cfg_in)
        if args.no_wandb:
            cfg_out.setdefault("wandb", {})["enabled"] = False
        if args.skip_test:
            cfg_out.setdefault("evaluation", {})["run_test_after_train"] = False
        return cfg_out

    def _resolve_resume_ckpt(cfg_in: Dict[str, Any]) -> Optional[Path]:
        resume_arg = (args.resume or "").strip()
        if not resume_arg:
            return None
        if resume_arg.lower() == "auto":
            run_nm = (args.run_name or cfg_in.get("experiment_name") or "").strip()
            if not run_nm:
                raise ValueError("--resume auto requires --run-name or a named experiment.")
            # Prefer explicit CLI --run-root, then version.yaml run_root, then legacy layout.
            rel_root = (
                args.run_root
                or cfg_in.get("run_root")
                or run_directory_layout(cfg_in)
            )
            ckpt_fp = resolve_path(str(rel_root)) / run_nm / "last_checkpoint.pth"
            if not ckpt_fp.is_file():
                raise FileNotFoundError(f"--resume auto: missing checkpoint {ckpt_fp}")
            return ckpt_fp
        ckpt_fp = resolve_path(resume_arg)
        if not ckpt_fp.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_fp}")
        return ckpt_fp

    def _run_one(cfg_in: Dict[str, Any]) -> Path:
        cfg_r = apply_training_seed_env(_apply_cli_flags(cfg_in))
        ckpt = _resolve_resume_ckpt(cfg_r)
        run_nm = (args.run_name or cfg_r.get("experiment_name") or "").strip() or None
        return run_training_job(
            cfg_r,
            run_root=args.run_root,
            skip_test=args.skip_test,
            run_name=run_nm,
            resume_ckpt=ckpt,
            cli_overrides={"no_wandb": bool(args.no_wandb), "skip_test": bool(args.skip_test)},
        )

    if args.run_all_experiments:
        if args.config:
            base = load_yaml_config(args.config)
        else:
            if not args.version or not args.dataset:
                parser.error("--run-all-experiments requires --version and --dataset (or --config)")
            base = compose_config(args.version, args.dataset)
        names = [str(e.get("name")) for e in list_experiments(base)]
        for name in names:
            if args.config:
                cfg = resolve_experiment(load_yaml_config(args.config), name)
            else:
                cfg = load_composed_config(
                    version=args.version,
                    dataset=args.dataset,
                    experiment_name=name,
                )
            logger.info("=== Starting experiment %s ===", name)
            prev_run_name = args.run_name
            if not prev_run_name:
                args.run_name = name
            _run_one(cfg)
            args.run_name = prev_run_name
        return

    if not args.config and (not args.version or not args.dataset):
        parser.error("Provide --version and --dataset, or --config.")
    cfg = load_composed_config(
        version=args.version,
        dataset=args.dataset,
        config_path=args.config,
        experiment_name=args.experiment,
    )
    _run_one(cfg)


if __name__ == "__main__":
    main()
