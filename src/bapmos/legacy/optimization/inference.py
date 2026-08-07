"""
Shared Inference Script for Optimization Experiments

Evaluates trained models on test set using boundary-aware metrics.
Generates:
- per_slice_metrics.csv
- summary_metrics.csv
- failure_analysis.csv
- Prediction masks (--save_predictions)
- Optional PNG+PDF panels + ``visualization_index.csv`` (--save_visualizations)
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Path setup not needed - using package-relative imports

import numpy as np
import math
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from bapmos.multiorgan.dataset_multi_organ import (
    MultiOrganDataset,
    compute_per_organ_boxes,
    multi_organ_collate_fn,
    sample_per_organ_points_with_negatives,
)
from bapmos.paths import (
    inference_output_dir_for_checkpoint,
    method_slug_from_checkpoint,
    project_root,
    resolve_model_checkpoint,
    resolve_under_project,
    simulation_cleaned_dataset_dir,
    write_method_evaluation_meta,
)
from bapmos.training_taxonomy import get_baseline_taxonomy_profile, log_baseline_taxonomy_startup

from bapmos.evaluation.test_panels import (
    distance_unit_short,
    multiclass_mask_to_bgr,
    save_pred_gt_difference_map,
    save_multiclass_test_panel_png_pdf,
    save_overlay_pred_pair_png_pdf,
)
from bapmos.evaluation.viz_index import write_visualization_index_csv
from bapmos.evaluation.viz_selection import (
    SliceVizRecord,
    aggregate_slice_metrics_for_image,
    select_slices_for_visualization,
)

from .metrics import MetricsEvaluator
from .trainer import PER_ORGAN_PROMPT_STRATEGIES


def _csv_metric_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.6f}"


def stable_rng_from_id(sample_id: str, base_seed: int):
    """Create stable RNG from sample ID."""
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def jitter_box(box, noise_pixels, rng):
    """Add jitter to bounding box (set to 0 for inference)."""
    return box  # No jitter during inference


class OptimizationInference:
    """
    Inference engine for optimization experiments.
    
    Args:
        checkpoint_path (Path): Path to best_checkpoint.pth
        device (str): Device to run inference on
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cuda",
        *,
        prompt_config_path: Optional[Path] = None,
        data_root_override: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = dict(ckpt["config"])
        if prompt_config_path is not None:
            self.apply_prompt_config(prompt_config_path)
        if data_root_override:
            self.config["data_root"] = data_root_override
        self.prompt_strategy = ckpt.get("prompt_strategy", "box_point")
        self.collapse_per_organ_arms_to_majority = bool(
            ckpt.get("collapse_per_organ_arms_to_majority", False)
        )
        self.per_organ_prompt_policy: Optional[Dict[str, str]] = ckpt.get("per_organ_prompt_policy")
        if (
            self.per_organ_prompt_policy is None
            and self.prompt_strategy in PER_ORGAN_PROMPT_STRATEGIES
            and ckpt.get("sampler_state") is not None
        ):
            try:
                sampler = pickle.loads(ckpt["sampler_state"])
                if hasattr(sampler, "get_best_arms_per_organ"):
                    self.per_organ_prompt_policy = sampler.get_best_arms_per_organ()
            except Exception:
                pass
        
        print(f"Prompt strategy: {self.prompt_strategy}")
        print(f"Best val MSD: {ckpt.get('best_val_msd', 'N/A')}")
        if prompt_config_path is not None:
            print(f"Prompt geometry config: {prompt_config_path}")
        if data_root_override:
            print(f"Data root override: {data_root_override}")

        data_root = self.config.get("data_root")
        if not data_root:
            raise ValueError("checkpoint config must include 'data_root' for taxonomy detection")
        self._taxonomy = get_baseline_taxonomy_profile(data_root)
        log_baseline_taxonomy_startup(self._taxonomy, prefix="[inference]")
        self._organ_keys = list(self._taxonomy.organ_keys)
        self._organ_to_class = dict(self._taxonomy.organ_to_class)
        self.num_classes = self._taxonomy.num_classes
        self._multiclass_eval_mapping = dict(self._taxonomy.multiclass_eval_mapping)
        
        # Load model
        sam_checkpoint = self.config.get("sam_checkpoint", "models/sam_base/sam_vit_b_01ec64.pth")
        ckpt_path = resolve_model_checkpoint(sam_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_checkpoint!r} "
                f"(tried project root {project_root()})"
            )
        self.model = sam_model_registry["vit_b"](checkpoint=str(ckpt_path)).to(self.device)
        from bapmos.external_baselines.medsam_init.weight_loader import (
            apply_medsam_encoder_init,
            is_medsam_init_config,
        )

        if is_medsam_init_config(self.config):
            apply_medsam_encoder_init(self.model, self.config)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.model.eval()
        
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        
        pixel_spacing = self.config.get("pixel_spacing", self._taxonomy.pixel_spacing_mm)
        pixel_spacing = tuple(pixel_spacing)
        unit_note = (
            "pixel units (spacing=1.0)"
            if self._taxonomy.taxonomy_name == "pfus1_eight_organ"
            else "mm"
        )
        print(f"Using pixel spacing: {pixel_spacing} ({unit_note})")
        self.evaluator = MetricsEvaluator(
            pixel_spacing=pixel_spacing,
            organs=list(self._taxonomy.evaluator_organ_labels),
        )

        if self.prompt_strategy == "box_point":
            if self.config.get("box_ratio", 0.5) >= 0.5:
                self.inference_mode = "box"
            else:
                self.inference_mode = "point"
        elif self.prompt_strategy == "three_way":
            weights = {
                "box": self.config.get("box_weight", 1),
                "point": self.config.get("point_weight", 1),
                "both": self.config.get("both_weight", 1),
            }
            self.inference_mode = max(weights, key=weights.get)
        elif self.prompt_strategy == "adaptive":
            bandit_state = ckpt.get("bandit_state", {})
            arm_avg_rewards = bandit_state.get("arm_avg_rewards", {"box": 0, "point": 0, "both": 0})
            self.inference_mode = max(arm_avg_rewards, key=arm_avg_rewards.get)
            print(f"Bandit best arm: {self.inference_mode} (avg rewards: {arm_avg_rewards})")
        elif (
            self.prompt_strategy in PER_ORGAN_PROMPT_STRATEGIES
            and self.per_organ_prompt_policy
            and not self.collapse_per_organ_arms_to_majority
        ):
            self.inference_mode = None
            print(f"Per-organ inference policy: {self.per_organ_prompt_policy}")
        else:
            self.inference_mode = "box"

        if self.inference_mode is not None:
            print(f"Inference mode: {self.inference_mode}")

    def apply_prompt_config(self, prompt_config_path: Path) -> None:
        """Merge ``common:`` from a training YAML into checkpoint config (infer-only geometry)."""
        path = resolve_under_project(str(prompt_config_path))
        if not path.is_file():
            raise FileNotFoundError(f"Prompt config not found: {prompt_config_path}")
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        common = doc.get("common") or {}
        if not common:
            raise ValueError(f"No 'common' section in prompt config: {path}")
        self.config.update(common)
        print(f"Merged prompt config common keys: {sorted(common.keys())}")

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
    
    def _points_prompt(self, mask_multi, original_size, sample_id):
        """Generate per-organ point prompts."""
        rng = stable_rng_from_id(sample_id, self.config["seed"])
        
        from bapmos.legacy.pfus1_advanced.scale_aware_prompts import (
            is_scale_aware_prompt_geometry,
            sample_per_organ_points_with_scale_aware_geometry,
        )

        if is_scale_aware_prompt_geometry(self.config):
            point_sets = sample_per_organ_points_with_scale_aware_geometry(
                mask_multi,
                self.config,
                num_pos_per_organ=self.config.get("num_pos_per_organ", 1),
                num_neg=self.config.get("num_neg_points", 3),
                rng=rng,
                organ_to_class=self._organ_to_class,
            )
        else:
            point_sets = sample_per_organ_points_with_negatives(
                mask_multi,
                num_pos_per_organ=self.config.get("num_pos_per_organ", 1),
                num_neg=self.config.get("num_neg_points", 3),
                ring_width=self.config.get("ring_width", 20),
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
        organ_boxes_dict = compute_per_organ_boxes(
            mask_multi, organ_to_class=self._organ_to_class
        )
        
        rng = stable_rng_from_id(str(sample_id), self.config["seed"])
        image_hw = (int(mask_multi.shape[0]), int(mask_multi.shape[1]))
        from bapmos.legacy.pfus1_advanced.scale_aware_prompts import apply_box_margin

        boxes_list = []
        present_organs = []

        for organ_name in self._organ_keys:
            box = organ_boxes_dict[organ_name]
            if box is not None:
                box = apply_box_margin(box, self.config, image_hw=image_hw)
                box = jitter_box(box, 0, rng)  # No jitter for inference
                boxes_list.append(box)
                present_organs.append(organ_name)
        
        if len(boxes_list) == 0:
            return None
        
        boxes_array = np.array(boxes_list)
        boxes_t = self.transform.apply_boxes(boxes_array, original_size)
        boxes_t = torch.as_tensor(boxes_t, device=self.device, dtype=torch.float32)
        
        return boxes_t, present_organs
    
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

    def _predict_one_per_organ_policy(
        self, image_rgb, mask_multi, filename, img_emb, original_size
    ):
        """Decode each present organ with its learned prompt arm."""
        point_result = self._points_prompt(mask_multi, original_size, filename)
        box_result = self._box_prompt(mask_multi, original_size, filename)

        points_by: Dict[str, dict] = {}
        boxes_by: Dict[str, torch.Tensor] = {}
        if point_result is not None:
            point_sets, point_organs = point_result
            for i, organ_name in enumerate(point_organs):
                points_by[organ_name] = point_sets[i]
        if box_result is not None:
            boxes_t, box_organs = box_result
            for i, organ_name in enumerate(box_organs):
                boxes_by[organ_name] = boxes_t[i : i + 1, :]

        present_organs = [
            o for o in self._organ_keys if o in points_by or o in boxes_by
        ]
        if not present_organs:
            return None

        policy = self.per_organ_prompt_policy or {}
        organ_masks = []
        executed_organs = []

        for organ_name in present_organs:
            mode = policy.get(organ_name, "box")
            box_t = boxes_by.get(organ_name)
            point_set = points_by.get(organ_name)

            if mode == "box":
                if box_t is None:
                    continue
                with torch.no_grad():
                    sparse, dense = self.model.prompt_encoder(
                        points=None, boxes=box_t.unsqueeze(1), masks=None
                    )
            elif mode == "point":
                if not self._point_set_is_valid(point_set):
                    if box_t is None:
                        continue
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=None, boxes=box_t.unsqueeze(1), masks=None
                        )
                else:
                    pts_t = self.transform.apply_coords(point_set["points"], original_size)
                    pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                    lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=(pts_t, lab_t), boxes=None, masks=None
                        )
            elif mode == "both":
                if box_t is not None and self._point_set_is_valid(point_set):
                    pts_t = self.transform.apply_coords(point_set["points"], original_size)
                    pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                    lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=(pts_t, lab_t), boxes=box_t.unsqueeze(1), masks=None
                        )
                elif box_t is not None:
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=None, boxes=box_t.unsqueeze(1), masks=None
                        )
                elif self._point_set_is_valid(point_set):
                    pts_t = self.transform.apply_coords(point_set["points"], original_size)
                    pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                    lab_t = torch.as_tensor(point_set["labels"], device=self.device, dtype=torch.int32).unsqueeze(0)
                    with torch.no_grad():
                        sparse, dense = self.model.prompt_encoder(
                            points=(pts_t, lab_t), boxes=None, masks=None
                        )
                else:
                    continue
            else:
                continue

            with torch.no_grad():
                low_res_mask, _ = self.model.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
            organ_masks.append(low_res_mask.squeeze(1))
            executed_organs.append(organ_name)

        if not organ_masks:
            return None

        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        logits = torch.zeros(
            1, self.num_classes, 256, 256, device=self.device, dtype=torch.float32
        )
        for idx, organ_name in enumerate(executed_organs):
            class_id = self._organ_to_class[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]

        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)

        pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
        pred_resized = cv2.resize(
            pred_classes.astype(np.float32),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return pred_resized.astype(np.uint8)

    def predict_one(self, image_rgb, mask_multi, filename):
        """Predict segmentation for one sample with robust fallback logic."""
        from bapmos.multiorgan.dataset_multi_organ import has_any_organ
        
        # Guard: Check if any organ present
        if not has_any_organ(mask_multi):
            return None
        
        x, original_size = self._prepare_image(image_rgb)
        
        # Image encoder
        with torch.no_grad():
            img_emb = self.model.image_encoder(x)

        if self.inference_mode is None and self.per_organ_prompt_policy:
            return self._predict_one_per_organ_policy(
                image_rgb, mask_multi, filename, img_emb, original_size
            )
        
        # Generate prompts with fallback logic
        if self.inference_mode == "box":
            prompt_result = self._box_prompt(mask_multi, original_size, filename)
            use_points = False
            use_boxes = True
        
        elif self.inference_mode == "point":
            prompt_result = self._points_prompt(mask_multi, original_size, filename)
            use_points = True
            use_boxes = False
            
            # Fallback to box if point prompts fail
            if prompt_result is None:
                prompt_result = self._box_prompt(mask_multi, original_size, filename)
                if prompt_result is not None:
                    use_points = False
                    use_boxes = True
        
        elif self.inference_mode == "both":
            point_result = self._points_prompt(mask_multi, original_size, filename)
            box_result = self._box_prompt(mask_multi, original_size, filename)
            
            # Fallback logic: use whichever is available
            if point_result is None and box_result is not None:
                # Points failed, use box only
                prompt_result = box_result
                use_points = False
                use_boxes = True
            elif box_result is None and point_result is not None:
                # Box failed, use points only
                prompt_result = point_result
                use_points = True
                use_boxes = False
            elif point_result is None or box_result is None:
                # Both failed
                return None
            else:
                # Both succeeded
                prompt_result = (point_result, box_result)
                use_points = True
                use_boxes = True
        
        else:
            raise ValueError(f"Unknown inference mode: {self.inference_mode}")
        
        if prompt_result is None:
            return None
        
        # Process each organ
        organ_masks = []
        
        if self.inference_mode == "both":
            point_sets, point_organs = prompt_result[0]
            boxes_t, box_organs = prompt_result[1]
            
            present_organs = [o for o in point_organs if o in box_organs]
            if len(present_organs) == 0:
                return None
            
            for organ_name in present_organs:
                point_idx = point_organs.index(organ_name)
                point_set = point_sets[point_idx]
                pts_t = self.transform.apply_coords(point_set['points'], original_size)
                pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                lab_t = torch.as_tensor(point_set['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
                
                box_idx = box_organs.index(organ_name)
                box_i = boxes_t[box_idx:box_idx+1, :]
                
                with torch.no_grad():
                    sparse, dense = self.model.prompt_encoder(
                        points=(pts_t, lab_t),
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
        
        else:
            if use_points:
                point_sets, present_organs = prompt_result
                
                for point_set in point_sets:
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
                
                for i in range(len(present_organs)):
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
        
        # Create multi-class prediction
        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        logits = torch.zeros(
            1, self.num_classes, 256, 256, device=self.device, dtype=torch.float32
        )
        
        for idx, organ_name in enumerate(present_organs):
            class_id = self._organ_to_class[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]
        
        # Background channel
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        
        # Get final prediction
        pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
        
        # Resize to original size
        pred_resized = cv2.resize(
            pred_classes.astype(np.float32),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_resized.astype(np.uint8)
    
    def run_inference(
        self,
        test_loader,
        output_dir: Path,
        save_predictions: bool = True,
        save_visualizations: bool = False,
        viz_selection: str = "all",
        viz_max: Optional[int] = None,
        viz_per_patient_max: Optional[int] = None,
        viz_seed: int = 42,
        save_viz_pdf: bool = True,
    ):
        """Run inference on test set."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_predictions:
            pred_dir = output_dir / "predictions"
            pred_ids_dir = pred_dir / "multiclass"
            pred_dir.mkdir(exist_ok=True)
            pred_ids_dir.mkdir(exist_ok=True)

        viz_dir = output_dir / "visualizations"
        if save_visualizations:
            viz_dir.mkdir(parents=True, exist_ok=True)

        unit = distance_unit_short(self._taxonomy.taxonomy_name)
        simple_viz = (
            save_visualizations
            and viz_selection == "all"
            and viz_max is None
            and viz_per_patient_max is None
        )

        viz_records: list[SliceVizRecord] = []
        idx_rows: list[dict] = []

        print(f"\nRunning inference on {len(test_loader.dataset)} test images...")

        self.evaluator.reset()

        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]

                pred = self.predict_one(image_rgb, mask_multi, fn)

                if pred is None:
                    print(f"Warning: Failed to predict for {fn}")
                    continue

                if save_predictions:
                    cv2.imwrite(str(pred_ids_dir / f"{fn}_pred_ids.png"), pred)
                    cv2.imwrite(
                        str(pred_dir / f"{fn}_pred.png"),
                        multiclass_mask_to_bgr(pred, self._taxonomy.organ_definitions),
                    )

                self.evaluator.evaluate_multiclass_slice(
                    pred,
                    mask_multi.astype(np.uint8),
                    slice_idx=i,
                    image_id=fn,
                    class_mapping=self._multiclass_eval_mapping,
                )

                rec = aggregate_slice_metrics_for_image(self.evaluator, str(fn))

                if save_visualizations and rec is not None:
                    if simple_viz:
                        stem = Path(str(fn)).stem
                        try:
                            png_path, pdf_path = save_overlay_pred_pair_png_pdf(
                                image_rgb=image_rgb,
                                pred_mask=pred.astype(np.uint8),
                                sample_stem=stem,
                                output_dir=viz_dir,
                                organ_definitions=self._taxonomy.organ_definitions,
                                taxonomy_name=self._taxonomy.taxonomy_name,
                                save_pdf=save_viz_pdf,
                            )
                            d_png, d_pdf = save_pred_gt_difference_map(
                                image_rgb=image_rgb,
                                gt_mask=mask_multi.astype(np.uint8),
                                pred_mask=pred.astype(np.uint8),
                                sample_stem=stem,
                                output_dir=viz_dir,
                                organ_definitions=self._taxonomy.organ_definitions,
                                taxonomy_name=self._taxonomy.taxonomy_name,
                                save_pdf=save_viz_pdf,
                            )
                            row = {
                                "sample_id": rec.sample_id,
                                "split": "test",
                                "mean_dice": _csv_metric_float(rec.mean_dice),
                                "mean_msd": _csv_metric_float(rec.mean_msd),
                                "mean_hd95": _csv_metric_float(rec.mean_hd95),
                                "distance_unit": unit,
                                "viz_png_relative": str(png_path.relative_to(output_dir)),
                                "diff_png_relative": str(d_png.relative_to(output_dir)),
                                "visualization_selection_mode": viz_selection,
                            }
                            if pdf_path is not None:
                                row["viz_pdf_relative"] = str(pdf_path.relative_to(output_dir))
                            if d_pdf is not None:
                                row["diff_pdf_relative"] = str(d_pdf.relative_to(output_dir))
                            idx_rows.append(row)
                        except Exception as exc:
                            print(f"Warning: visualization failed for {fn}: {exc}")
                    else:
                        viz_records.append(rec)

        if save_visualizations and not simple_viz:
            picked = select_slices_for_visualization(
                viz_records,
                viz_selection,
                viz_max,
                viz_seed,
                per_patient_max=viz_per_patient_max,
            )
            selected_stems = {r.sample_stem for r in picked}
            by_stem = {r.sample_stem: r for r in picked}

            for batch in tqdm(test_loader, desc="Visualizations (2nd pass)"):
                images = batch["image"]
                masks = batch["mask"]
                filenames = batch["filename"]

                for i in range(len(images)):
                    image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                    mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                    fn = filenames[i]
                    stem = Path(str(fn)).stem

                    if stem not in selected_stems:
                        continue

                    pred = self.predict_one(image_rgb, mask_multi, fn)
                    if pred is None:
                        continue

                    rec = by_stem.get(stem)
                    if rec is None:
                        continue

                    try:
                        png_path, pdf_path = save_multiclass_test_panel_png_pdf(
                            image_rgb=image_rgb,
                            gt_mask=mask_multi.astype(np.uint8),
                            pred_mask=pred.astype(np.uint8),
                            sample_stem=stem,
                            output_dir=viz_dir,
                            organ_definitions=self._taxonomy.organ_definitions,
                            taxonomy_name=self._taxonomy.taxonomy_name,
                        )
                        row = {
                            "sample_id": rec.sample_id,
                            "split": "test",
                            "mean_dice": _csv_metric_float(rec.mean_dice),
                            "mean_msd": _csv_metric_float(rec.mean_msd),
                            "mean_hd95": _csv_metric_float(rec.mean_hd95),
                            "distance_unit": unit,
                            "panel_png_relative": str(png_path.relative_to(output_dir)),
                            "visualization_selection_mode": viz_selection,
                        }
                        if pdf_path is not None:
                            row["panel_pdf_relative"] = str(pdf_path.relative_to(output_dir))
                        idx_rows.append(row)
                    except Exception as exc:
                        print(f"Warning: visualization failed for {fn}: {exc}")

        if save_visualizations and idx_rows:
            write_visualization_index_csv(output_dir, idx_rows)

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        print("\nExporting metrics...")
        self.evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
        self.evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
        self.evaluator.export_failure_analysis_csv(
            metrics_dir / "failure_analysis.csv", top_n=20
        )
        rows = []
        for organ in self._taxonomy.evaluator_organ_labels:
            agg = self.evaluator.aggregate_metrics(organ_name=organ)
            if agg is not None:
                rows.append(agg)
        if rows:
            import pandas as pd

            pd.DataFrame(rows).to_csv(metrics_dir / "per_organ_metrics.csv", index=False)

        print("\n" + "=" * 80)
        print("INFERENCE SUMMARY")
        print("=" * 80)

        for organ in self._taxonomy.evaluator_organ_labels:
            summary = self.evaluator.aggregate_metrics(organ)
            if summary:
                print(f"\n{organ}:")
                print(f"  Slices: {summary['n_slices']} ({summary['n_valid_boundary']} valid boundary)")
                if summary["msd_mm_mean"] is not None:
                    dist_u = "px" if self._taxonomy.taxonomy_name == "pfus1_eight_organ" else "mm"
                    print(
                        f"  MSD: {summary['msd_mm_mean']:.3f} ± {summary['msd_mm_std']:.3f} {dist_u}"
                    )
                    print(
                        f"  HD95: {summary['hd95_mm_mean']:.3f} ± {summary['hd95_mm_std']:.3f} {dist_u}"
                    )
                print(f"  Dice: {summary['dice_mean']:.3f} ± {summary['dice_std']:.3f}")
                print(f"  Empty pred rate: {summary['empty_pred_rate']:.1%}")

        print("\n" + "=" * 80)
        print(f"Results saved to: {output_dir}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_checkpoint.pth")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root with slice PNGs, masks/, splits/ (default: data/prostate/simulation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: output/<dataset>/<method_slug>/ e.g. "
            "output/real_data/case1/ucb1_global/). Use --legacy-output-layout for the old "
            "runs-mirror path .../Optimization/<strategy>/<run>/test/."
        ),
    )
    parser.add_argument(
        "--method-slug",
        dest="method_slug",
        type=str,
        default=None,
        help="Flat folder name under output/<dataset>/ (default: inferred from checkpoint).",
    )
    parser.add_argument(
        "--legacy-output-layout",
        action="store_true",
        help="Use pre-2026 layout output/<dataset>/Optimization/<strategy>/<run>/test/.",
    )
    parser.add_argument("--save_predictions", action="store_true", help="Save prediction masks")
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help=(
            "Save figure panels under <output_dir>/visualizations/ and visualization_index.csv. "
            "Use --max-visualizations for large test sets (e.g. PFUS1)."
        ),
    )
    parser.add_argument(
        "--save-viz-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When visualizing, also write PDF panels (default: on). Use --no-save-viz-pdf for PFUS1-scale runs.",
    )
    parser.add_argument(
        "--visualization-selection",
        choices=["all", "random", "worst_msd", "best_msd", "per_patient_even"],
        default="all",
        help="Subset policy when used with --max-visualizations or non-all modes.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=None,
        dest="max_visualizations",
        help="Maximum panels to write after selection (global cap).",
    )
    parser.add_argument(
        "--viz-per-patient-max",
        type=int,
        default=None,
        dest="viz_per_patient_max",
        help="With per_patient_even: evenly spaced PDF/PNG panels per patient.",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=42,
        help="Seed for random / capped-all shuffling.",
    )
    parser.add_argument(
        "--splits-subdir",
        dest="splits_subdir",
        type=str,
        default="splits_stratified",
        help="Subdirectory of data_root with train/val/test lists (default: splits_stratified)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--prompt-config",
        dest="prompt_config",
        type=str,
        default=None,
        help=(
            "YAML with a common: section (e.g. pfus1_advanced geometry). "
            "Merged into checkpoint config for infer-only prompt tests; does not reload weights."
        ),
    )
    args = parser.parse_args()

    default_data = simulation_cleaned_dataset_dir()
    data_root = args.data_root or str(default_data)
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if args.output_dir:
        output_dir = resolve_under_project(args.output_dir)
        method_slug = args.method_slug or method_slug_from_checkpoint(checkpoint_path)
    else:
        layout_style = "legacy" if args.legacy_output_layout else "method"
        output_dir = inference_output_dir_for_checkpoint(
            checkpoint_path,
            data_root,
            split="test",
            layout_style=layout_style,
            method_slug=args.method_slug,
        )
        method_slug = args.method_slug or method_slug_from_checkpoint(checkpoint_path)
    print(f"Inference output directory: {output_dir}")

    meta_extra: dict = {}
    try:
        try:
            _ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            _ckpt_meta = torch.load(checkpoint_path, map_location="cpu")
        _cfg_meta = _ckpt_meta.get("config") or {}
        if _cfg_meta.get("prompt_strategy"):
            meta_extra["prompt_strategy"] = _cfg_meta["prompt_strategy"]
        if _cfg_meta.get("run_name"):
            meta_extra["run_name"] = _cfg_meta["run_name"]
    except Exception:
        pass
    write_method_evaluation_meta(
        output_dir,
        checkpoint=checkpoint_path,
        data_root=data_root,
        method_slug=method_slug,
        split="test",
        extra=meta_extra or None,
    )

    splits_subdir = args.splits_subdir
    if splits_subdir == "splits_stratified":
        from bapmos.training_taxonomy import default_splits_subdir

        splits_subdir = default_splits_subdir(data_root)

    test_dataset = MultiOrganDataset(
        data_root,
        split="test",
        splits_subdir=splits_subdir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=multi_organ_collate_fn,
    )
    
    prompt_config_path = (
        resolve_under_project(args.prompt_config) if args.prompt_config else None
    )
    inference = OptimizationInference(
        checkpoint_path,
        device=args.device,
        prompt_config_path=prompt_config_path,
        data_root_override=data_root,
    )
    inference.run_inference(
        test_loader,
        output_dir,
        save_predictions=args.save_predictions,
        save_visualizations=args.save_visualizations,
        viz_selection=args.visualization_selection,
        viz_max=args.max_visualizations,
        viz_per_patient_max=args.viz_per_patient_max,
        viz_seed=args.visualization_seed,
        save_viz_pdf=args.save_viz_pdf,
    )


if __name__ == "__main__":
    main()
