"""
Comprehensive visualization generator for **simulation** optimization experiments.

Uses simulation-only organ/color definitions. For Case 1/2/PFUS1, extend taxonomy
support or use dataset-specific viz scripts before relying on this module.

Output structure for each experiment:
Optimization_Visualization/<phase>/<experiment>/
├── train/
│   ├── predictions/           # Binary prediction masks
│   ├── overlays/              # Annotated with Dice scores
│   ├── simple_overlays/       # Semi-transparent colored overlays
│   ├── difference/            # Error maps (TP=black, FP=red, FN=blue)
│   ├── difference_v2/         # Same as difference/
│   ├── difference_v3/         # Superimposed on original image
│   ├── difference_v4/         # Publication-quality PDF difference maps
│   ├── multiplane/            # 3-plane comparison views
│   ├── rotation_3d/           # 3D rotation GIFs
│   ├── slices/                # Slice animation GIFs
│   ├── results.csv            # Per-image metrics
│   └── summary.json           # Aggregate statistics
├── val/                       # Same structure
└── test/                      # Same structure

Usage:
    # Single experiment
    python -m bapmos.legacy.optimization.generate_visualizations \\
        --checkpoint runs/Optimization/box_point/exp_box_0.90/best_checkpoint.pth
    
    # All experiments
    python -m bapmos.legacy.optimization.generate_visualizations --batch
"""

import os
import sys
from pathlib import Path

import argparse
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Import from multiorgan module
from bapmos.multiorgan.dataset_multi_organ import (
    MultiOrganDataset,
    sample_per_organ_points_with_negatives,
    compute_per_organ_boxes,
)
from bapmos.organ_registry import (
    SIMULATION_CLASS_ID_TO_DISPLAY,
    SIMULATION_COLORS_BGR,
    SIMULATION_ORGAN_KEYS,
    SIMULATION_ORGAN_TO_CLASS,
    SIMULATION_THREE_ORGANS,
)
from bapmos.paths import project_root, resolve_model_checkpoint, resolve_under_project
from bapmos.pdf_export import PDF_EXPORT_DPI

_SIM_FG_CLASS_IDS = [o.class_id for o in SIMULATION_THREE_ORGANS]


def stable_rng_from_id(sample_id: str, base_seed: int):
    """Create deterministic RNG from sample ID."""
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def dice_coefficient(pred_mask, gt_mask, class_id):
    """Compute Dice coefficient for a specific class."""
    pred_binary = (pred_mask == class_id).astype(np.float32)
    gt_binary = (gt_mask == class_id).astype(np.float32)
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union


def hausdorff_distance(pred_mask, gt_mask, class_id, percentile=100):
    """
    Compute Hausdorff Distance (HD) or HD95 for a specific class.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        class_id: Class to evaluate
        percentile: Use 100 for HD, 95 for HD95
    
    Returns:
        Hausdorff distance in pixels (or np.inf if masks don't have proper surfaces)
    """
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)
    
    # If both empty, distance is 0
    if pred_binary.sum() == 0 and gt_binary.sum() == 0:
        return 0.0
    
    # If one is empty and other isn't, return inf
    if pred_binary.sum() == 0 or gt_binary.sum() == 0:
        return np.inf
    
    # Get surface points (edges)
    from scipy.ndimage import binary_erosion
    
    # Surface = mask XOR eroded_mask
    pred_surface = pred_binary ^ binary_erosion(pred_binary)
    gt_surface = gt_binary ^ binary_erosion(gt_binary)
    
    # If no surface (solid masks), use distance transform
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        # Compute distance transform
        dist_pred = distance_transform_edt(~pred_binary.astype(bool))
        dist_gt = distance_transform_edt(~gt_binary.astype(bool))
        
        # Get surface distances
        surface_dist_pred_to_gt = dist_gt[pred_binary == 1]
        surface_dist_gt_to_pred = dist_pred[gt_binary == 1]
    else:
        # Distance from pred surface to nearest gt surface
        dist_pred_to_gt = distance_transform_edt(~gt_surface.astype(bool))
        surface_dist_pred_to_gt = dist_pred_to_gt[pred_surface == 1]
        
        # Distance from gt surface to nearest pred surface
        dist_gt_to_pred = distance_transform_edt(~pred_surface.astype(bool))
        surface_dist_gt_to_pred = dist_gt_to_pred[gt_surface == 1]
    
    # Hausdorff distance
    if percentile == 100:
        hd_pred_to_gt = surface_dist_pred_to_gt.max() if len(surface_dist_pred_to_gt) > 0 else 0
        hd_gt_to_pred = surface_dist_gt_to_pred.max() if len(surface_dist_gt_to_pred) > 0 else 0
        return max(hd_pred_to_gt, hd_gt_to_pred)
    else:
        # HD95 - 95th percentile
        hd_pred_to_gt = np.percentile(surface_dist_pred_to_gt, percentile) if len(surface_dist_pred_to_gt) > 0 else 0
        hd_gt_to_pred = np.percentile(surface_dist_gt_to_pred, percentile) if len(surface_dist_gt_to_pred) > 0 else 0
        return max(hd_pred_to_gt, hd_gt_to_pred)


def mean_surface_distance(pred_mask, gt_mask, class_id):
    """
    Compute Mean Surface Distance (MSD), also known as Mean Surface Distance (MSD).
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        class_id: Class to evaluate
    
    Returns:
        Average surface distance in pixels
    """
    pred_binary = (pred_mask == class_id).astype(np.uint8)
    gt_binary = (gt_mask == class_id).astype(np.uint8)
    
    # If both empty, distance is 0
    if pred_binary.sum() == 0 and gt_binary.sum() == 0:
        return 0.0
    
    # If one is empty and other isn't, return inf
    if pred_binary.sum() == 0 or gt_binary.sum() == 0:
        return np.inf
    
    # Get surface points
    from scipy.ndimage import binary_erosion
    pred_surface = pred_binary ^ binary_erosion(pred_binary)
    gt_surface = gt_binary ^ binary_erosion(gt_binary)
    
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0
    
    # Distance from pred surface to nearest gt point
    dist_pred_to_gt = distance_transform_edt(~gt_binary.astype(bool))
    surface_dist_pred_to_gt = dist_pred_to_gt[pred_surface == 1]
    
    # Distance from gt surface to nearest pred point
    dist_gt_to_pred = distance_transform_edt(~pred_binary.astype(bool))
    surface_dist_gt_to_pred = dist_gt_to_pred[gt_surface == 1]
    
    # Average of both directions
    all_distances = np.concatenate([surface_dist_pred_to_gt, surface_dist_gt_to_pred])
    return np.mean(all_distances)


def add_text_annotation(image, text, position, font_scale=0.6, thickness=2,
                        text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=5):
    """Add text annotation with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    cv2.rectangle(image, (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return text_height + baseline + 2 * padding


def create_simple_overlay(image, pred_mask):
    """Create semi-transparent colored overlay showing only predictions."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    overlay = image.copy()
    
    for class_id in _SIM_FG_CLASS_IDS:
        organ_mask = (pred_mask == class_id).astype(np.uint8)
        if organ_mask.sum() > 0:
            color_rgb = SIMULATION_COLORS_BGR[class_id][::-1]  # BGR to RGB
            colored_mask = np.zeros_like(overlay)
            colored_mask[organ_mask == 1] = color_rgb
            overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
    
    return overlay


def create_overlay(image, pred_mask, dice_scores):
    """Create annotated overlay with Dice scores."""
    simple = create_simple_overlay(image, pred_mask)
    
    y_offset = 30
    for class_id in _SIM_FG_CLASS_IDS:
        organ_name = SIMULATION_CLASS_ID_TO_DISPLAY[class_id]
        dice = dice_scores[organ_name]
        color_rgb = SIMULATION_COLORS_BGR[class_id][::-1]  # BGR to RGB
        text = f"{organ_name}: {dice:.3f}"
        y_offset += add_text_annotation(simple, text, (10, y_offset),
                                       text_color=color_rgb, bg_color=(0, 0, 0))
    
    mean_dice = np.mean(list(dice_scores.values()))
    add_text_annotation(simple, f"Mean: {mean_dice:.3f}", (10, y_offset + 10),
                       text_color=(255, 255, 255), bg_color=(0, 0, 0))
    
    return simple


def create_difference_map(gt_mask, pred_mask):
    """Create difference map: Green=Correct, Blue=Missed, Red=False, White=Background."""
    H, W = gt_mask.shape
    diff_map = np.ones((H, W, 3), dtype=np.uint8) * 255  # White background
    
    for class_id in _SIM_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        # Correct (True positive): green (BGR)
        tp = gt_binary & pred_binary
        diff_map[tp] = [0, 255, 0]
        
        # False positive: red (BGR)
        fp = (~gt_binary) & pred_binary
        diff_map[fp] = [0, 0, 255]
        
        # Missed (False negative): blue (BGR)
        fn = gt_binary & (~pred_binary)
        diff_map[fn] = [255, 0, 0]
    
    return diff_map


def create_difference_v3(image, gt_mask, pred_mask):
    """Create difference map superimposed on original image."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    H, W = gt_mask.shape
    diff_map = image.copy()
    overlay = image.copy()
    
    all_tp = np.zeros((H, W), dtype=bool)
    all_fp = np.zeros((H, W), dtype=bool)
    all_fn = np.zeros((H, W), dtype=bool)
    
    for class_id in _SIM_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        all_tp |= (gt_binary & pred_binary)
        all_fp |= ((~gt_binary) & pred_binary)
        all_fn |= (gt_binary & (~pred_binary))
    
    # Apply colors (BGR)
    overlay[all_tp] = [0, 255, 0]   # Green
    overlay[all_fp] = [0, 0, 255]   # Red
    overlay[all_fn] = [255, 0, 0]   # Blue
    
    # Blend 50/50
    mask_any = all_tp | all_fp | all_fn
    diff_map[mask_any] = cv2.addWeighted(image[mask_any], 0.5, overlay[mask_any], 0.5, 0)
    
    return diff_map


def save_difference_map_as_pdf(gt_mask, pred_mask, output_path):
    """
    Save difference map as PDF for publication quality.
    Uses matplotlib vector output (not raster-embedded).
    
    Args:
        gt_mask: (H, W) ground truth multi-class mask
        pred_mask: (H, W) predicted multi-class mask
        output_path: Path to save PDF file
    """
    H, W = gt_mask.shape
    
    # Collect all pixels for each category across all organs
    all_tp = np.zeros((H, W), dtype=bool)
    all_fp = np.zeros((H, W), dtype=bool)
    all_fn = np.zeros((H, W), dtype=bool)
    
    for class_id in _SIM_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        # True positive
        tp = gt_binary & pred_binary
        all_tp |= tp
        
        # False positive
        fp = (~gt_binary) & pred_binary
        all_fp |= fp
        
        # False negative
        fn = gt_binary & (~pred_binary)
        all_fn |= fn
    
    # Create RGB image (matplotlib uses RGB)
    diff_map_rgb = np.ones((H, W, 3), dtype=np.uint8) * 200  # Grey background
    diff_map_rgb[all_tp] = [0, 255, 0]   # Green for correct
    diff_map_rgb[all_fp] = [255, 0, 0]   # Red for false
    diff_map_rgb[all_fn] = [0, 0, 255]   # Blue for missed
    
    # Figure size in inches so savefig at PDF_EXPORT_DPI yields W×H pixels
    dpi = PDF_EXPORT_DPI
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(diff_map_rgb)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Rectangle
    legend_x = 10
    legend_y = 10
    legend_width = 250
    legend_height = 120
    
    # White background for legend
    rect = Rectangle((legend_x, legend_y), legend_width, legend_height, 
                     facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Legend title
    ax.text(legend_x + 10, legend_y + 25, "Legend:", 
            fontsize=10, fontweight='bold', va='top')
    
    # Legend items
    item_height = 30
    box_size = 20
    
    # Green: Correct
    rect_green = Rectangle((legend_x + 10, legend_y + 35), box_size, box_size, 
                           facecolor='green', edgecolor='none')
    ax.add_patch(rect_green)
    ax.text(legend_x + 40, legend_y + 47, "Correct", fontsize=8, va='center')
    
    # Red: False
    rect_red = Rectangle((legend_x + 10, legend_y + 35 + item_height), box_size, box_size, 
                         facecolor='red', edgecolor='none')
    ax.add_patch(rect_red)
    ax.text(legend_x + 40, legend_y + 47 + item_height, "False", fontsize=8, va='center')
    
    # Blue: Missed
    rect_blue = Rectangle((legend_x + 10, legend_y + 35 + 2*item_height), box_size, box_size, 
                          facecolor='blue', edgecolor='none')
    ax.add_patch(rect_blue)
    ax.text(legend_x + 40, legend_y + 47 + 2*item_height, "Missed", fontsize=8, va='center')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# Backward-compatible alias (deprecated)
save_difference_map_as_eps = save_difference_map_as_pdf


def create_prediction_mask(pred_mask):
    """Create colored prediction mask for visualization."""
    H, W = pred_mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id in _SIM_FG_CLASS_IDS:
        mask = (pred_mask == class_id)
        colored[mask] = SIMULATION_COLORS_BGR[class_id]
    
    return colored


class OptimizationVisualizer:
    """Generate comprehensive visualizations for optimization experiments."""
    
    def __init__(self, checkpoint_path: Path, output_base: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_base = output_base
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.prompt_strategy = ckpt.get("prompt_strategy", "box_point")
        
        # Determine inference mode
        if self.prompt_strategy == "box_point":
            self.inference_mode = "box" if self.config.get("box_ratio", 0.5) >= 0.5 else "point"
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
        else:
            self.inference_mode = "box"
        
        print(f"Inference mode: {self.inference_mode}")
        
        # Load model
        sam_checkpoint = self.config.get("sam_checkpoint", "models/sam_base/sam_vit_b_01ec64.pth")
        ckpt_path = resolve_model_checkpoint(sam_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_checkpoint!r} "
                f"(tried project root {project_root()})"
            )
        self.model = sam_model_registry["vit_b"](checkpoint=str(ckpt_path)).to(self.device)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.model.eval()
        
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
    
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
        
        return x.unsqueeze(0), original_size
    
    def _get_prompts(self, mask_multi, original_size, sample_id):
        """Generate prompts based on inference mode."""
        # Box prompts
        if self.inference_mode in ["box", "both"]:
            organ_boxes_dict = compute_per_organ_boxes(
                mask_multi, organ_to_class=SIMULATION_ORGAN_TO_CLASS
            )
            boxes_list = []
            box_organs = []
            
            for organ_name in SIMULATION_ORGAN_KEYS:
                box = organ_boxes_dict[organ_name]
                if box is not None:
                    boxes_list.append(box)
                    box_organs.append(organ_name)
            
            if boxes_list:
                boxes_array = np.array(boxes_list)
                boxes_t = self.transform.apply_boxes(boxes_array, original_size)
                boxes_t = torch.as_tensor(boxes_t, device=self.device, dtype=torch.float32)
            else:
                boxes_t, box_organs = None, []
        else:
            boxes_t, box_organs = None, []
        
        # Point prompts
        if self.inference_mode in ["point", "both"]:
            rng = stable_rng_from_id(sample_id, self.config["seed"])
            point_sets = sample_per_organ_points_with_negatives(
                mask_multi,
                num_pos_per_organ=self.config.get("num_pos_per_organ", 1),
                num_neg=self.config.get("num_neg_points", 3),
                ring_width=self.config.get("ring_width", 20),
                rng=rng,
                organ_to_class=SIMULATION_ORGAN_TO_CLASS,
            )
            
            valid_point_sets = []
            point_organs = []
            for organ_name in SIMULATION_ORGAN_KEYS:
                if point_sets[organ_name] is not None:
                    valid_point_sets.append(point_sets[organ_name])
                    point_organs.append(organ_name)
        else:
            valid_point_sets, point_organs = [], []
        
        return boxes_t, box_organs, valid_point_sets, point_organs
    
    def predict(self, image_rgb, mask_multi, sample_id):
        """Run inference on one sample."""
        x, original_size = self._prepare_image(image_rgb)
        
        with torch.no_grad():
            img_emb = self.model.image_encoder(x)
        
        boxes_t, box_organs, point_sets, point_organs = self._get_prompts(mask_multi, original_size, sample_id)
        
        # Determine common organs
        if self.inference_mode == "both":
            present_organs = [o for o in box_organs if o in point_organs]
        elif self.inference_mode == "box":
            present_organs = box_organs
        else:  # point
            present_organs = point_organs
        
        if not present_organs:
            return None
        
        # Process each organ
        organ_masks = []
        for organ_name in present_organs:
            if self.inference_mode == "both":
                # Use both prompts
                point_idx = point_organs.index(organ_name)
                box_idx = box_organs.index(organ_name)
                
                point_set = point_sets[point_idx]
                pts_t = self.transform.apply_coords(point_set['points'], original_size)
                pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                lab_t = torch.as_tensor(point_set['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
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
            
            elif self.inference_mode == "box":
                box_idx = box_organs.index(organ_name)
                box_i = boxes_t[box_idx:box_idx+1, :]
                
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
            
            else:  # point
                point_idx = point_organs.index(organ_name)
                point_set = point_sets[point_idx]
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
        
        # Create multi-class prediction
        organ_masks_stacked = torch.stack(organ_masks, dim=0).squeeze(1)
        logits = torch.zeros(1, 4, 256, 256, device=self.device, dtype=torch.float32)
        
        for idx, organ_name in enumerate(present_organs):
            class_id = SIMULATION_ORGAN_TO_CLASS[organ_name]
            logits[0, class_id, :, :] = organ_masks_stacked[idx, :, :]
        
        # Background
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        organ_union_prob = organ_probs.max(dim=1, keepdim=True)[0]
        background_prob = 1.0 - organ_union_prob
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        
        # Final prediction
        pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]
        
        # Resize to original size
        pred_resized = cv2.resize(
            pred_classes.astype(np.float32),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_resized.astype(np.uint8)
    
    def process_split(self, split: str):
        """Process one split and generate all 2D visualizations."""
        print(f"\n{'='*80}")
        print(f"Processing {split} split")
        print(f"{'='*80}\n")
        
        # Create output directories
        split_dir = self.output_base / split
        (split_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (split_dir / "overlays").mkdir(parents=True, exist_ok=True)
        (split_dir / "simple_overlays").mkdir(parents=True, exist_ok=True)
        (split_dir / "difference").mkdir(parents=True, exist_ok=True)
        (split_dir / "difference_v2").mkdir(parents=True, exist_ok=True)
        (split_dir / "difference_v3").mkdir(parents=True, exist_ok=True)
        (split_dir / "difference_v4").mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset = MultiOrganDataset(
            self.config["data_root"],
            split=split,
            splits_subdir=self.config.get("splits_subdir", "splits_stratified"),
        )
        
        results = []
        
        for idx in tqdm(range(len(dataset)), desc=f"Generating {split} visualizations"):
            sample = dataset[idx]
            image = sample["image"].numpy() if torch.is_tensor(sample["image"]) else sample["image"]
            mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else sample["mask"]
            filename = sample["filename"]
            
            if (mask > 0).sum() == 0:
                continue
            
            # Run inference
            pred_mask = self.predict(image, mask, filename)
            if pred_mask is None:
                continue
            
            # Compute metrics
            dice_scores = {}
            hd_scores = {}
            hd95_scores = {}
            msd_scores = {}
            
            for class_id in _SIM_FG_CLASS_IDS:
                organ_name = SIMULATION_CLASS_ID_TO_DISPLAY[class_id]
                dice = dice_coefficient(pred_mask, mask, class_id)
                hd = hausdorff_distance(pred_mask, mask, class_id, percentile=100)
                hd95 = hausdorff_distance(pred_mask, mask, class_id, percentile=95)
                asd = mean_surface_distance(pred_mask, mask, class_id)
                
                dice_scores[organ_name] = dice
                hd_scores[organ_name] = hd
                hd95_scores[organ_name] = hd95
                msd_scores[organ_name] = asd
            
            mean_dice = np.mean(list(dice_scores.values()))
            mean_hd = np.mean([v for v in hd_scores.values() if v != np.inf])
            mean_hd95 = np.mean([v for v in hd95_scores.values() if v != np.inf])
            mean_msd = np.mean([v for v in msd_scores.values() if v != np.inf])
            
            # 1. Save binary prediction
            pred_binary = (pred_mask > 0).astype(np.uint8) * 255
            Image.fromarray(pred_binary).save(split_dir / "predictions" / f"{filename}.png")
            
            # 2. Save simple overlay
            simple = create_simple_overlay(image, pred_mask)
            cv2.imwrite(str(split_dir / "simple_overlays" / f"{filename}.png"),
                       cv2.cvtColor(simple, cv2.COLOR_RGB2BGR))
            
            # 3. Save annotated overlay
            annotated = create_overlay(image, pred_mask, dice_scores)
            cv2.imwrite(str(split_dir / "overlays" / f"{filename}.png"),
                       cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            
            # 4. Save difference maps
            diff = create_difference_map(mask, pred_mask)
            cv2.imwrite(str(split_dir / "difference" / f"{filename}.png"),
                       cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(split_dir / "difference_v2" / f"{filename}.png"),
                       cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
            
            diff_v3 = create_difference_v3(image, mask, pred_mask)
            cv2.imwrite(str(split_dir / "difference_v3" / f"{filename}.png"),
                       cv2.cvtColor(diff_v3, cv2.COLOR_RGB2BGR))
            
            # v4: PDF difference map (publication quality)
            base_filename = filename.replace('.png', '')
            save_difference_map_as_pdf(
                mask,
                pred_mask,
                str(split_dir / "difference_v4" / f"{base_filename}.pdf"),
            )
            
            # Store results
            results.append({
                "filename": filename,
                "mean_dice": mean_dice,
                "dice_rectum": dice_scores['Rectum'],
                "dice_bladder": dice_scores['Bladder'],
                "dice_ptv1": dice_scores['PTV1'],
                "mean_hd": mean_hd,
                "hd_rectum": hd_scores['Rectum'],
                "hd_bladder": hd_scores['Bladder'],
                "hd_ptv1": hd_scores['PTV1'],
                "mean_hd95": mean_hd95,
                "hd95_rectum": hd95_scores['Rectum'],
                "hd95_bladder": hd95_scores['Bladder'],
                "hd95_ptv1": hd95_scores['PTV1'],
                "mean_msd": mean_msd,
                "msd_rectum": msd_scores['Rectum'],
                "msd_bladder": msd_scores['Bladder'],
                "msd_ptv1": msd_scores['PTV1']
            })
        
        # Save CSV
        if results:
            csv_path = split_dir / "results.csv"
            fieldnames = ["filename", "mean_dice", "dice_rectum", "dice_bladder", "dice_ptv1",
                         "mean_hd", "hd_rectum", "hd_bladder", "hd_ptv1",
                         "mean_hd95", "hd95_rectum", "hd95_bladder", "hd95_ptv1",
                         "mean_msd", "msd_rectum", "msd_bladder", "msd_ptv1"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            # Save summary JSON
            mean_dices = [r["mean_dice"] for r in results]
            mean_hds = [r["mean_hd"] for r in results if r["mean_hd"] != np.inf]
            mean_hd95s = [r["mean_hd95"] for r in results if r["mean_hd95"] != np.inf]
            mean_msds = [r["mean_msd"] for r in results if r["mean_msd"] != np.inf]
            
            summary = {
                "split": split,
                "num_images": len(results),
                "mean_dice": {
                    "mean": float(np.mean(mean_dices)),
                    "std": float(np.std(mean_dices)),
                    "median": float(np.median(mean_dices)),
                    "min": float(np.min(mean_dices)),
                    "max": float(np.max(mean_dices))
                },
                "mean_hd": {
                    "mean": float(np.mean(mean_hds)) if len(mean_hds) > 0 else np.inf,
                    "std": float(np.std(mean_hds)) if len(mean_hds) > 0 else 0.0,
                    "median": float(np.median(mean_hds)) if len(mean_hds) > 0 else np.inf,
                    "min": float(np.min(mean_hds)) if len(mean_hds) > 0 else np.inf,
                    "max": float(np.max(mean_hds)) if len(mean_hds) > 0 else np.inf
                },
                "mean_hd95": {
                    "mean": float(np.mean(mean_hd95s)) if len(mean_hd95s) > 0 else np.inf,
                    "std": float(np.std(mean_hd95s)) if len(mean_hd95s) > 0 else 0.0,
                    "median": float(np.median(mean_hd95s)) if len(mean_hd95s) > 0 else np.inf,
                    "min": float(np.min(mean_hd95s)) if len(mean_hd95s) > 0 else np.inf,
                    "max": float(np.max(mean_hd95s)) if len(mean_hd95s) > 0 else np.inf
                },
                "mean_msd": {
                    "mean": float(np.mean(mean_msds)) if len(mean_msds) > 0 else np.inf,
                    "std": float(np.std(mean_msds)) if len(mean_msds) > 0 else 0.0,
                    "median": float(np.median(mean_msds)) if len(mean_msds) > 0 else np.inf,
                    "min": float(np.min(mean_msds)) if len(mean_msds) > 0 else np.inf,
                    "max": float(np.max(mean_msds)) if len(mean_msds) > 0 else np.inf
                },
                "per_organ": {}
            }
            
            for organ in SIMULATION_ORGAN_KEYS:
                dice_scores = [r[f"dice_{organ}"] for r in results]
                hd_scores = [r[f"hd_{organ}"] for r in results if r[f"hd_{organ}"] != np.inf]
                hd95_scores = [r[f"hd95_{organ}"] for r in results if r[f"hd95_{organ}"] != np.inf]
                msd_scores = [r[f"msd_{organ}"] for r in results if r[f"msd_{organ}"] != np.inf]
                
                summary["per_organ"][organ] = {
                    "dice": {
                        "mean": float(np.mean(dice_scores)),
                        "std": float(np.std(dice_scores)),
                        "median": float(np.median(dice_scores))
                    },
                    "hd": {
                        "mean": float(np.mean(hd_scores)) if len(hd_scores) > 0 else np.inf,
                        "std": float(np.std(hd_scores)) if len(hd_scores) > 0 else 0.0,
                        "median": float(np.median(hd_scores)) if len(hd_scores) > 0 else np.inf
                    },
                    "hd95": {
                        "mean": float(np.mean(hd95_scores)) if len(hd95_scores) > 0 else np.inf,
                        "std": float(np.std(hd95_scores)) if len(hd95_scores) > 0 else 0.0,
                        "median": float(np.median(hd95_scores)) if len(hd95_scores) > 0 else np.inf
                    },
                    "asd": {
                        "mean": float(np.mean(msd_scores)) if len(msd_scores) > 0 else np.inf,
                        "std": float(np.std(msd_scores)) if len(msd_scores) > 0 else 0.0,
                        "median": float(np.median(msd_scores)) if len(msd_scores) > 0 else np.inf
                    }
                }
            
            with open(split_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        print(f"✓ Processed {len(results)} images")
        return results
    
    def generate_3d_visualizations(self):
        """Generate 3D visualizations from predictions."""
        print(f"\n{'='*80}")
        print(f"Generating 3D visualizations")
        print(f"{'='*80}\n")
        
        # Import 3D visualization modules
        try:
            ext_3d = project_root() / "3D_Project_Inference"
            sys.path.insert(0, str(ext_3d))
            from volume_overlay_renderer import VolumeOverlayRenderer
            from slice_comparison_renderer import SliceComparisonRenderer
            from comparison_calculator import ComparisonCalculator
        except ImportError as e:
            print(f"⚠️  Skipping 3D visualizations: {e}")
            return
        
        calculator = ComparisonCalculator()
        volume_renderer = VolumeOverlayRenderer()
        slice_renderer = SliceComparisonRenderer()
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_base / split
            pred_dir = split_dir / "predictions"
            
            if not pred_dir.exists():
                continue
            
            # Load predictions
            predictions = {}
            for pred_file in sorted(pred_dir.glob("*.png")):
                img_id = pred_file.stem
                mask = np.array(Image.open(pred_file))
                mask_resized = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
                predictions[img_id] = (mask_resized > 0).astype(np.uint8)
            
            if len(predictions) < 3:
                continue
            
            # Load ground truths
            gt_base = Path(self.config["data_root"]).parent / "combined_masks"
            ground_truths = {}
            for img_id in predictions.keys():
                gt_file = gt_base / f"{img_id}_combined_mask.png"
                if gt_file.exists():
                    gt_mask = np.array(Image.open(gt_file))
                    gt_resized = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST))
                    ground_truths[img_id] = (gt_resized > 0).astype(np.uint8)
            
            # Filter to matched pairs
            matched_ids = sorted(set(predictions.keys()) & set(ground_truths.keys()))
            if len(matched_ids) < 3:
                continue
            
            # Stack to volume
            pred_volume = calculator.stack_slices_to_volume(
                {k: predictions[k] for k in matched_ids}, matched_ids
            )
            gt_volume = calculator.stack_slices_to_volume(
                {k: ground_truths[k] for k in matched_ids}, matched_ids
            )
            
            correct, false, missed = calculator.calculate_volume_comparison(pred_volume, gt_volume)
            
            # Create 3D output dirs
            (split_dir / "rotation_3d").mkdir(exist_ok=True)
            (split_dir / "multiplane").mkdir(exist_ok=True)
            (split_dir / "slices").mkdir(exist_ok=True)
            
            # Generate visualizations
            try:
                volume_renderer.render_comparison_rotation(
                    gt_volume, correct, false, missed,
                    output_path=split_dir / "rotation_3d" / f"{split}_rotation_3d.gif",
                    title=f"{split.capitalize()} Set"
                )
                
                slice_renderer.render_multiplane_comparison(
                    gt_volume, correct, false, missed,
                    output_path=split_dir / "multiplane" / f"{split}_multiplane.png",
                    title=f"{split.capitalize()} Set"
                )
                
                slice_renderer.render_slice_animation(
                    gt_volume, correct, false, missed,
                    output_path=split_dir / "slices" / f"{split}_slices.gif",
                    title=f"{split.capitalize()} Set"
                )
                
                print(f"✓ Generated 3D visualizations for {split}")
            except Exception as e:
                print(f"⚠️  Error generating 3D for {split}: {e}")
        
        print(f"✓ 3D visualization complete")
    
    def generate_all(self, splits=['train', 'val', 'test']):
        """Generate all visualizations."""
        print(f"\n{'='*80}")
        print(f"Optimization Visualization Generator")
        print(f"Output: {self.output_base}")
        print(f"{'='*80}\n")
        
        # 2D visualizations
        for split in splits:
            self.process_split(split)
        
        # 3D visualizations
        self.generate_3d_visualizations()
        
        print(f"\n{'='*80}")
        print(f"✅ All visualizations complete!")
        print(f"{'='*80}\n")


def find_checkpoints(runs_dir: Path):
    """Find all checkpoints for batch processing."""
    checkpoints = []
    
    for phase_dir in sorted(runs_dir.iterdir()):
        if not phase_dir.is_dir():
            continue
        
        for exp_dir in sorted(phase_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            checkpoint_path = exp_dir / "best_checkpoint.pth"
            if checkpoint_path.exists():
                exp_name = exp_dir.name.split('_2026')[0].split('_202')[0]
                checkpoints.append({
                    'checkpoint': checkpoint_path,
                    'experiment': exp_name,
                    'phase': phase_dir.name
                })
    
    return checkpoints


def process_batch(runs_dir: Path, output_base: Path, device: str, splits: list):
    """Process all checkpoints in batch mode."""
    print("="*80)
    print("Optimization Experiments - Batch Visualization Generator")
    print("="*80)
    
    all_checkpoints = find_checkpoints(runs_dir)
    
    if not all_checkpoints:
        print(f"\n❌ No checkpoints found in {runs_dir}")
        return 1
    
    print(f"\nFound {len(all_checkpoints)} experiments")
    
    with tqdm(total=len(all_checkpoints), desc="Overall Progress") as pbar:
        for checkpoint_info in all_checkpoints:
            exp_name = checkpoint_info['experiment']
            phase = checkpoint_info['phase']
            checkpoint_path = checkpoint_info['checkpoint']
            output_dir = output_base / phase / exp_name
            
            pbar.set_description(f"Processing {phase}/{exp_name}")
            
            try:
                visualizer = OptimizationVisualizer(checkpoint_path, output_dir, device=device)
                visualizer.generate_all(splits=splits)
                pbar.write(f"✓ {phase}/{exp_name}")
            except Exception as e:
                pbar.write(f"✗ {phase}/{exp_name} - Error: {str(e)[:100]}")
            
            pbar.update(1)
    
    print(f"\n{'='*80}")
    print(f"✅ Batch processing complete!")
    print(f"{'='*80}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive visualizations for optimization models')
    parser.add_argument("--checkpoint", type=str, help="Path to best_checkpoint.pth (single mode)")
    parser.add_argument("--batch", action="store_true", help="Process all checkpoints in runs/Optimization/")
    parser.add_argument("--runs_dir", type=str, default="runs/Optimization")
    parser.add_argument("--output_base", type=str, default="Optimization_Visualization")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    if args.batch:
        return process_batch(Path(args.runs_dir), Path(args.output_base), args.device, args.splits)
    
    if not args.checkpoint:
        parser.error("Either --checkpoint or --batch is required")
    
    checkpoint_path = Path(args.checkpoint)
    
    # Auto-detect output directory from checkpoint path
    parts = checkpoint_path.parts
    if "Optimization" in parts:
        opt_idx = parts.index("Optimization")
        phase = parts[opt_idx + 1]
        experiment = parts[opt_idx + 2].split('_2026')[0]
        output_dir = Path(args.output_base) / phase / experiment
    else:
        print("❌ Cannot auto-detect output path")
        return 1
    
    visualizer = OptimizationVisualizer(checkpoint_path, output_dir, device=args.device)
    visualizer.generate_all(splits=args.splits)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
