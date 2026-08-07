"""
Comprehensive inference + visualization for **real clinical** Case 1/2 only.

**QA / qualitative figures only** — metrics in ``metrics.csv`` from this script use
local boundary helpers, not the canonical ``MetricsEvaluator`` training pipeline.
For paper tables, use checkpoints evaluated via training ``metrics.csv`` or
``bapmos.inference_output.export_bundle``.

By default, looks for runs under ``BAPMOS/runs/<bundle>/Baseline`` (trainer default).
Legacy layouts under ``data/prostate/caseN/runs/`` are also scanned.

Runs inference on trained Baseline (+ optional Optimization) models and generates:
- Prediction masks
- GT + Pred overlay visualizations
- Per-organ overlays
- Difference maps
- Metrics (Dice, HD95, MSD)
"""

from pathlib import Path

# Path setup not needed - using package-relative imports

import numpy as np
import torch
import cv2
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from ..data.organ_registry import (
    REAL_CLINICAL_CLASS_ID_TO_DISPLAY,
    REAL_CLINICAL_ORGAN_KEYS,
    REAL_CLINICAL_ORGAN_TO_CLASS,
)
from ..paths import case_real_data_dir, dataset_bundle_tag, project_root, resolve_model_checkpoint, resolve_under_project
from .dataset_multi_organ import (
    MultiOrganDataset,
    sample_per_organ_points_with_negatives,
    compute_per_organ_boxes,
)


def stable_rng_from_id(sample_id: str, base_seed: int):
    """Create stable RNG from sample ID."""
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def dice_coefficient(pred, gt):
    """Compute Dice coefficient."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union


def compute_hd95_msd(pred_mask, gt_mask):
    """Compute HD95 and MSD metrics."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return 999.0, 999.0
    
    # Get surfaces
    pred_border = pred ^ binary_erosion(pred, structure=np.ones((3,3)))
    gt_border = gt ^ binary_erosion(gt, structure=np.ones((3,3)))
    
    # Distance transforms
    dt_pred = distance_transform_edt(~pred_border)
    dt_gt = distance_transform_edt(~gt_border)
    
    # Get distances
    distances_pred_to_gt = dt_gt[pred_border]
    distances_gt_to_pred = dt_pred[gt_border]
    
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    
    hd95 = np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0
    msd = np.mean(all_distances) if len(all_distances) > 0 else 0.0
    
    return hd95, msd


def create_overlay_image(image, gt_mask, pred_mask, class_id=None):
    """
    Create overlay visualization.
    Green = Ground Truth, Red = Prediction, Yellow = Overlap
    
    Args:
        image: Grayscale image (H, W)
        gt_mask: Ground truth mask (H, W) - multiclass
        pred_mask: Prediction mask (H, W) - multiclass
        class_id: If specified, visualize only this class. If None, visualize all.
    """
    # Convert to RGB
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = image.copy()
    
    if class_id is not None:
        # Single class visualization
        gt_binary = (gt_mask == class_id).astype(np.uint8)
        pred_binary = (pred_mask == class_id).astype(np.uint8)
    else:
        # All organs (any non-zero)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)
    
    # Create color masks
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255  # Green channel
    
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255  # Red channel
    
    yellow = np.zeros_like(overlay)
    yellow[:, :, 1] = 255  # Green
    yellow[:, :, 2] = 255  # Red
    
    # Apply colors
    overlap = (gt_binary & pred_binary).astype(bool)
    gt_only = (gt_binary & ~pred_binary).astype(bool)
    pred_only = (pred_binary & ~gt_binary).astype(bool)
    
    # Blend (only if regions exist)
    alpha = 0.4
    if overlap.any():
        overlay[overlap] = cv2.addWeighted(overlay[overlap], 1-alpha, yellow[overlap], alpha, 0)
    if gt_only.any():
        overlay[gt_only] = cv2.addWeighted(overlay[gt_only], 1-alpha, green[gt_only], alpha, 0)
    if pred_only.any():
        overlay[pred_only] = cv2.addWeighted(overlay[pred_only], 1-alpha, red[pred_only], alpha, 0)
    
    return overlay


class RealDataInference:
    """Inference + Visualization for real data models."""
    
    CLASS_NAMES = {0: "Background", **REAL_CLINICAL_CLASS_ID_TO_DISPLAY}
    
    def __init__(self, checkpoint_path, output_dir, data_root=None):
        """
        Args:
            checkpoint_path: Path to best_checkpoint.pth
            output_dir: Where to save visualizations
            data_root: Path to data directory (default: ``data/prostate/case1``)
        """
        from bapmos.paths import real_case_dataset_dir

        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root) if data_root is not None else real_case_dataset_dir("case1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint and config
        print(f"\n{'='*80}")
        print(f"Loading: {self.checkpoint_path.parent.name}")
        print(f"{'='*80}")
        
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.prompt_type = self.config.get("prompt_type", "box")
        self.num_classes = self.config.get("num_classes", 5)
        self.seed = self.config.get("seed", 42)
        
        # Handle optimization methods with prompt_strategy
        if "prompt_strategy" in ckpt:
            self.prompt_strategy = ckpt["prompt_strategy"]
            print(f"Optimization method: {self.prompt_strategy}")
        else:
            self.prompt_strategy = self.prompt_type  # Baseline
        
        print(f"Prompt type: {self.prompt_type}")
        print(f"Num classes: {self.num_classes}")
        print(f"Seed: {self.seed}")
        
        # Load model
        sam_checkpoint = self.config.get("sam_checkpoint", "models/sam_base/sam_vit_b_01ec64.pth")
        ckpt_path = resolve_model_checkpoint(sam_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_checkpoint!r} "
                f"(tried project root {project_root()})"
            )
        sam_checkpoint = str(ckpt_path)
        
        print(f"SAM checkpoint: {sam_checkpoint}")
        
        self.model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = ResizeLongestSide(1024)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "overlays").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "per_organ").mkdir(parents=True, exist_ok=True)
        
        # Load test dataset
        self.test_dataset = MultiOrganDataset(
            data_root=str(self.data_root),
            split="test",
            splits_subdir=self.config.get("splits_subdir", "splits_stratified"),
        )
        
        print(f"Test set size: {len(self.test_dataset)}")
        
    def run_inference(self):
        """Run inference on test set and save visualizations + metrics."""
        results = []
        
        print(f"\nRunning inference on {len(self.test_dataset)} test images...")
        
        for idx in tqdm(range(len(self.test_dataset)), desc="Test images"):
            sample = self.test_dataset[idx]
            
            image = sample['image']  # (H, W, 3) RGB from dataset
            gt_mask = sample['mask']  # (H, W) multiclass
            filename = sample['filename']
            image_id = filename.replace('.png', '')  # Extract image ID
            
            # Dataset already returns RGB, so use directly
            image_rgb = image.astype(np.uint8)  # Ensure uint8
            
            # Transform and normalize
            input_image = self.transform.apply_image(image_rgb)
            input_image_torch = torch.as_tensor(input_image, device=self.device, dtype=torch.float32).permute(2, 0, 1)
            
            # Normalize with SAM's mean and std
            pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
            pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3, 1, 1)
            input_image_torch = (input_image_torch - pixel_mean) / pixel_std
            
            # Pad to model size
            h, w = input_image_torch.shape[-2:]
            padh = self.model.image_encoder.img_size - h
            padw = self.model.image_encoder.img_size - w
            input_image_torch = torch.nn.functional.pad(input_image_torch, (0, padw, 0, padh))
            
            # Create per-organ prompts
            rng = stable_rng_from_id(image_id, self.seed)
            
            # Run inference - process each organ separately
            with torch.no_grad():
                # Get image embeddings once
                image_embedding = self.model.image_encoder(input_image_torch[None, ...])
                
                # Create multiclass prediction mask
                pred_mask = np.zeros_like(gt_mask)
                
                # Process each organ (class IDs from registry, not loop index)
                for organ_name in REAL_CLINICAL_ORGAN_KEYS:
                    class_id = REAL_CLINICAL_ORGAN_TO_CLASS[organ_name]
                    if "box" in self.prompt_type or self.prompt_strategy == "box":
                        # Get box for this organ
                        boxes_dict = compute_per_organ_boxes(gt_mask)
                        box = boxes_dict.get(organ_name)
                        if box is None:
                            continue  # Skip missing organs
                        
                        box_t = self.transform.apply_boxes(np.array([box]), image_rgb.shape[:2])
                        box_t = torch.as_tensor(box_t, device=self.device, dtype=torch.float32)
                        
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=None,
                            boxes=box_t.unsqueeze(1),  # Add extra dimension
                            masks=None,
                        )
                        
                    elif "point" in self.prompt_type or self.prompt_strategy == "points":
                        # Get points for this organ
                        points_dict = sample_per_organ_points_with_negatives(
                            gt_mask,
                            num_pos_per_organ=1,
                            num_neg=3,
                            ring_width=20,
                            rng=rng
                        )
                        point_data = points_dict.get(organ_name)
                        if point_data is None:
                            continue  # Skip missing organs
                        
                        pts_t = self.transform.apply_coords(point_data['points'], image_rgb.shape[:2])
                        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                        lab_t = torch.as_tensor(point_data['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
                        
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=(pts_t, lab_t),
                            boxes=None,
                            masks=None,
                        )
                        
                    else:
                        # Box + points combination
                        boxes_dict = compute_per_organ_boxes(gt_mask)
                        box = boxes_dict.get(organ_name)
                        points_dict = sample_per_organ_points_with_negatives(
                            gt_mask,
                            num_pos_per_organ=1,
                            num_neg=3,
                            ring_width=20,
                            rng=rng
                        )
                        point_data = points_dict.get(organ_name)
                        
                        if box is None or point_data is None:
                            continue  # Skip if either is missing
                        
                        box_t = self.transform.apply_boxes(np.array([box]), image_rgb.shape[:2])
                        box_t = torch.as_tensor(box_t, device=self.device, dtype=torch.float32)
                        
                        pts_t = self.transform.apply_coords(point_data['points'], image_rgb.shape[:2])
                        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
                        lab_t = torch.as_tensor(point_data['labels'], device=self.device, dtype=torch.int32).unsqueeze(0)
                        
                        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                            points=(pts_t, lab_t),
                            boxes=box_t.unsqueeze(1),  # Add extra dimension
                            masks=None,
                        )
                    
                    # Decode mask for this organ
                    low_res_mask, _ = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # Upscale to original size
                    mask = torch.nn.functional.interpolate(
                        low_res_mask,
                        size=image_rgb.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    mask_np = (mask.squeeze().cpu().numpy() > 0.0).astype(np.uint8)
                    pred_mask[mask_np > 0] = class_id
            
            # Compute metrics
            dice_scores = []
            hd95_scores = []
            msd_scores = []
            
            for organ_idx in range(1, self.num_classes):
                gt_organ = (gt_mask == organ_idx).astype(np.uint8)
                pred_organ = (pred_mask == organ_idx).astype(np.uint8)
                
                dice = dice_coefficient(pred_organ, gt_organ)
                hd95, msd = compute_hd95_msd(pred_organ, gt_organ)
                
                dice_scores.append(dice)
                hd95_scores.append(hd95)
                msd_scores.append(msd)
            
            avg_dice = np.mean(dice_scores)
            avg_hd95 = np.mean([h for h in hd95_scores if h < 999])  # Exclude missing organs
            avg_msd = np.mean([m for m in msd_scores if m < 999])
            
            results.append({
                'image_id': image_id,
                'dice': avg_dice,
                'hd95': avg_hd95,
                'msd': avg_msd,
                **{f'dice_class{i}': dice_scores[i-1] for i in range(1, self.num_classes)},
                **{f'hd95_class{i}': hd95_scores[i-1] for i in range(1, self.num_classes)},
                **{f'msd_class{i}': msd_scores[i-1] for i in range(1, self.num_classes)},
            })
            
            # Save prediction mask
            pred_path = self.output_dir / "test" / "predictions" / f"{image_id}_pred.png"
            cv2.imwrite(str(pred_path), pred_mask.astype(np.uint8))
            
            # Create overlays
            # 1. All organs overlay
            overlay_all = create_overlay_image(image, gt_mask, pred_mask, class_id=None)
            overlay_path = self.output_dir / "test" / "overlays" / f"{image_id}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_all)
            
            # 2. Per-organ overlays
            for organ_idx in range(1, self.num_classes):
                organ_name = self.CLASS_NAMES[organ_idx]
                overlay_organ = create_overlay_image(image, gt_mask, pred_mask, class_id=organ_idx)
                organ_path = self.output_dir / "test" / "per_organ" / f"{image_id}_{organ_name}.png"
                cv2.imwrite(str(organ_path), overlay_organ)
        
        # Save metrics
        df = pd.DataFrame(results)
        metrics_path = self.output_dir / "test" / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Average Dice:  {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
        print(f"Average HD95:  {df['hd95'].mean():.2f} ± {df['hd95'].std():.2f} mm")
        print(f"Average MSD:   {df['msd'].mean():.2f} ± {df['msd'].std():.2f} mm")
        print(f"\nPer-class Dice:")
        for i in range(1, self.num_classes):
            print(f"  {self.CLASS_NAMES[i]:10s}: {df[f'dice_class{i}'].mean():.4f} ± {df[f'dice_class{i}'].std():.4f}")
        print(f"\n✓ Saved to: {self.output_dir}")
        print(f"  - {len(df)} predictions in test/predictions/")
        print(f"  - {len(df)} overlays in test/overlays/")
        print(f"  - {len(df) * (self.num_classes-1)} per-organ overlays in test/per_organ/")
        print(f"  - metrics.csv with detailed results")
        
        return df


def main():
    """Run inference on all trained models."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case1",
                       help="case1 or case2")
    parser.add_argument("--model_type", type=str, default=None,
                       help="Baseline or Optimization (None = all)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Specific model name (None = all)")
    args = parser.parse_args()
    
    base_dir = case_real_data_dir(args.case)
    bundle = dataset_bundle_tag(str(base_dir))

    # Prefer trainer default layout under BAPMOS/runs/<bundle>/…
    project_runs = project_root() / "runs" / bundle
    legacy_runs = base_dir / "runs"
    if project_runs.is_dir():
        runs_dir = project_runs
    elif legacy_runs.is_dir():
        runs_dir = legacy_runs
    else:
        runs_dir = project_runs  # default even if empty (clearer errors below)

    outputs_dir = project_root() / "output" / "real_data" / args.case / "visualizations"
    if not outputs_dir.parent.exists():
        # Fall back next to the case bundle when output/ is unused
        outputs_dir = base_dir / "outputs" / "visualizations"
    data_root = (base_dir / "data") if (base_dir / "data").is_dir() else base_dir
    
    # Find all model directories
    model_dirs = []
    
    if args.model_type:
        # Specific type
        type_dir = runs_dir / args.model_type
        if args.model_name:
            # Specific model
            model_dirs = [type_dir / args.model_name]
        else:
            # All models of this type
            # Baseline has checkpoints directly, Optimization has nested structure
            if args.model_type == "Baseline":
                model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d / "best_checkpoint.pth").exists()]
            else:
                # Optimization has method/run structure: find all runs across all methods
                for method_dir in type_dir.iterdir():
                    if method_dir.is_dir():
                        # Get all runs in this method
                        model_dirs.extend([d for d in method_dir.iterdir() if d.is_dir() and (d / "best_checkpoint.pth").exists()])
    else:
        # All models
        for model_type in ["Baseline", "Optimization"]:
            type_dir = runs_dir / model_type
            if type_dir.exists():
                if model_type == "Baseline":
                    model_dirs.extend([d for d in type_dir.iterdir() if d.is_dir() and (d / "best_checkpoint.pth").exists()])
                else:
                    # Optimization: nested structure
                    for method_dir in type_dir.iterdir():
                        if method_dir.is_dir():
                            model_dirs.extend([d for d in method_dir.iterdir() if d.is_dir() and (d / "best_checkpoint.pth").exists()])
    
    print(f"\n{'='*80}")
    print(f"INFERENCE + VISUALIZATION - {args.case.upper()}")
    print(f"{'='*80}")
    print(f"Found {len(model_dirs)} models to process")
    print(f"Data root: {data_root}")
    print(f"Output: {outputs_dir}")
    print(f"{'='*80}\n")
    
    # Process each model
    for model_dir in model_dirs:
        checkpoint_path = model_dir / "best_checkpoint.pth"
        
        # Determine model type
        if "Baseline" in str(model_dir):
            model_type = "Baseline"
        else:
            model_type = "Optimization"
        
        # Create output directory matching structure
        output_dir = outputs_dir / model_type / model_dir.name
        
        # Run inference
        try:
            inferencer = RealDataInference(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                data_root=data_root
            )
            inferencer.run_inference()
        except Exception as e:
            print(f"\n❌ ERROR processing {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✓ ALL INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {outputs_dir}/")


if __name__ == "__main__":
    main()
