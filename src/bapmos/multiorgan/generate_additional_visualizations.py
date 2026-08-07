"""
Generate Additional Visualizations for Real Data (Matching Simulation Structure)

Takes existing predictions and generates:
- simple_overlays (colored masks without annotations)
- difference (error maps with legend)
- difference_v2, difference_v3, difference_v4, difference_5 (variants)
- gt_vs_pred (side-by-side comparison)
- predictions_multiclass (multiclass mask copies)
- cleaned/ (all above without text annotations)

This matches the visualization structure from simulation data.
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..data.organ_registry import REAL_CLINICAL_COLORS_BGR, REAL_CLINICAL_ORGANS
from ..paths import find_combined_masks_dir, find_training_images_dir
from ..pdf_export import PDF_EXPORT_DPI

# BGR for OpenCV; class IDs follow on-disk masks
ORGAN_COLORS = REAL_CLINICAL_COLORS_BGR
ORGAN_NAMES = {0: "Background"}
for _o in REAL_CLINICAL_ORGANS:
    ORGAN_NAMES[_o.class_id] = _o.evaluator_label

_REAL_FG_CLASS_IDS = [o.class_id for o in REAL_CLINICAL_ORGANS]


def create_simple_overlay(image, pred_mask, with_text=True):
    """
    Create simple colored overlay showing predicted masks.
    
    Args:
        image: (H, W) or (H, W, 3) grayscale or RGB image
        pred_mask: (H, W) multiclass prediction
        with_text: Add text annotations (False for cleaned version)
    """
    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = image.copy()
    
    overlay = image.copy()
    
    # Apply semi-transparent colors for each organ
    for class_id in _REAL_FG_CLASS_IDS:
        organ_mask = (pred_mask == class_id).astype(np.uint8)
        
        if organ_mask.sum() > 0:
            color = ORGAN_COLORS[class_id]
            colored_mask = np.zeros_like(overlay)
            colored_mask[organ_mask == 1] = color
            
            # Blend with 40% opacity
            overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
    
    if with_text:
        # Add organ names in top-left corner
        y_offset = 20
        for class_id in _REAL_FG_CLASS_IDS:
            if (pred_mask == class_id).any():
                color = ORGAN_COLORS[class_id]
                text = ORGAN_NAMES[class_id]
                cv2.putText(overlay, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20
    
    return overlay


def create_difference_map(gt_mask, pred_mask, with_legend=True):
    """
    Create error/difference map.
    Green = Correct, Red = False Positive, Blue = Missed
    
    Args:
        gt_mask: (H, W) ground truth multiclass
        pred_mask: (H, W) prediction multiclass
        with_legend: Add legend (False for cleaned version)
    """
    H, W = gt_mask.shape
    diff_map = np.ones((H, W, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Collect all pixels across all organs
    all_tp = np.zeros((H, W), dtype=bool)
    all_fp = np.zeros((H, W), dtype=bool)
    all_fn = np.zeros((H, W), dtype=bool)
    
    for class_id in _REAL_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        # True positive: green
        tp = gt_binary & pred_binary
        all_tp |= tp
        
        # False positive: red
        fp = (~gt_binary) & pred_binary
        all_fp |= fp
        
        # False negative: blue
        fn = gt_binary & (~pred_binary)
        all_fn |= fn
    
    # Apply colors (BGR)
    diff_map[all_tp] = [0, 255, 0]   # Green
    diff_map[all_fp] = [0, 0, 255]   # Red
    diff_map[all_fn] = [255, 0, 0]   # Blue
    
    if with_legend:
        # Add legend
        legend_height = 120
        legend_width = 250
        legend_x = 10
        legend_y = 10
        
        # White background for legend
        legend_bg = diff_map[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width].copy()
        overlay = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        diff_map[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = cv2.addWeighted(
            legend_bg, 0.3, overlay, 0.7, 0
        )
        
        # Border
        cv2.rectangle(diff_map, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), 2)
        
        # Legend items
        item_height = 30
        box_size = 20
        
        # Title
        cv2.putText(diff_map, "Legend:", 
                   (legend_x + 10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Correct (Green)
        cv2.rectangle(diff_map, 
                     (legend_x + 10, legend_y + 35), 
                     (legend_x + 10 + box_size, legend_y + 35 + box_size),
                     (0, 255, 0), -1)
        cv2.putText(diff_map, "Correct", 
                   (legend_x + 40, legend_y + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # False (Red)
        cv2.rectangle(diff_map, 
                     (legend_x + 10, legend_y + 35 + item_height), 
                     (legend_x + 10 + box_size, legend_y + 35 + item_height + box_size),
                     (0, 0, 255), -1)
        cv2.putText(diff_map, "False", 
                   (legend_x + 40, legend_y + 52 + item_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Missed (Blue)
        cv2.rectangle(diff_map, 
                     (legend_x + 10, legend_y + 35 + 2*item_height), 
                     (legend_x + 10 + box_size, legend_y + 35 + 2*item_height + box_size),
                     (255, 0, 0), -1)
        cv2.putText(diff_map, "Missed", 
                   (legend_x + 40, legend_y + 52 + 2*item_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return diff_map


def create_difference_v3(image, gt_mask, pred_mask, with_legend=True):
    """
    Create difference overlaid on original image (semi-transparent).
    
    Args:
        image: (H, W) or (H, W, 3) original image
        gt_mask: (H, W) ground truth
        pred_mask: (H, W) prediction
        with_legend: Add legend
    """
    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = image.copy()
    
    diff_map = image.copy()
    overlay = image.copy()
    
    # Collect pixels
    all_tp = np.zeros(gt_mask.shape, dtype=bool)
    all_fp = np.zeros(gt_mask.shape, dtype=bool)
    all_fn = np.zeros(gt_mask.shape, dtype=bool)
    
    for class_id in _REAL_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        all_tp |= (gt_binary & pred_binary)
        all_fp |= (~gt_binary & pred_binary)
        all_fn |= (gt_binary & ~pred_binary)
    
    # Apply colors with transparency
    alpha = 0.5
    overlay[all_tp] = [0, 255, 0]
    overlay[all_fp] = [0, 0, 255]
    overlay[all_fn] = [255, 0, 0]
    
    mask_any = all_tp | all_fp | all_fn
    diff_map[mask_any] = cv2.addWeighted(
        image[mask_any], 1-alpha,
        overlay[mask_any], alpha,
        0
    )
    
    if with_legend:
        # Same legend as difference
        legend_height = 120
        legend_width = 250
        legend_x = 10
        legend_y = 10
        
        legend_bg = diff_map[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width].copy()
        overlay_legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        diff_map[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = cv2.addWeighted(
            legend_bg, 0.3, overlay_legend, 0.7, 0
        )
        
        cv2.rectangle(diff_map, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), 2)
        
        item_height = 30
        box_size = 20
        
        cv2.putText(diff_map, "Legend:", (legend_x + 10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.rectangle(diff_map, (legend_x + 10, legend_y + 35), 
                     (legend_x + 10 + box_size, legend_y + 35 + box_size),
                     (0, 255, 0), -1)
        cv2.putText(diff_map, "Correct", (legend_x + 40, legend_y + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(diff_map, (legend_x + 10, legend_y + 35 + item_height), 
                     (legend_x + 10 + box_size, legend_y + 35 + item_height + box_size),
                     (0, 0, 255), -1)
        cv2.putText(diff_map, "False", (legend_x + 40, legend_y + 52 + item_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(diff_map, (legend_x + 10, legend_y + 35 + 2*item_height), 
                     (legend_x + 10 + box_size, legend_y + 35 + 2*item_height + box_size),
                     (255, 0, 0), -1)
        cv2.putText(diff_map, "Missed", (legend_x + 40, legend_y + 52 + 2*item_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return diff_map


def create_gt_vs_pred_overlay(image, gt_mask, pred_mask):
    """
    Create side-by-side GT (left) vs Prediction (right).
    
    Args:
        image: (H, W) or (H, W, 3) original
        gt_mask: (H, W) ground truth
        pred_mask: (H, W) prediction
    
    Returns:
        combined: (H, W*2, 3) side-by-side image
    """
    # Create overlays
    gt_overlay = create_simple_overlay(image, gt_mask, with_text=False)
    pred_overlay = create_simple_overlay(image, pred_mask, with_text=False)
    
    # Concatenate horizontally
    combined = np.concatenate([gt_overlay, pred_overlay], axis=1)
    
    # Add labels
    cv2.putText(combined, "Ground Truth", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(combined, "Prediction", (gt_overlay.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return combined


def save_difference_as_pdf(gt_mask, pred_mask, output_path):
    """Save difference map as PDF for publication."""
    H, W = gt_mask.shape
    
    # Collect pixels
    all_tp = np.zeros((H, W), dtype=bool)
    all_fp = np.zeros((H, W), dtype=bool)
    all_fn = np.zeros((H, W), dtype=bool)
    
    for class_id in _REAL_FG_CLASS_IDS:
        gt_binary = (gt_mask == class_id)
        pred_binary = (pred_mask == class_id)
        
        all_tp |= (gt_binary & pred_binary)
        all_fp |= (~gt_binary & pred_binary)
        all_fn |= (gt_binary & ~pred_binary)
    
    # Create RGB image (matplotlib uses RGB not BGR)
    diff_map_rgb = np.ones((H, W, 3), dtype=np.uint8) * 200
    diff_map_rgb[all_tp] = [0, 255, 0]
    diff_map_rgb[all_fp] = [255, 0, 0]
    diff_map_rgb[all_fn] = [0, 0, 255]
    
    dpi = PDF_EXPORT_DPI
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(diff_map_rgb)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


save_difference_as_eps = save_difference_as_pdf


def process_model_visualizations(model_dir, data_root):
    """
    Process all test images for one model directory.
    
    Args:
        model_dir: Path to model output (e.g., outputs/visualizations/Baseline/case1_multiorgan_box_realdata)
        data_root: Path to data directory for loading original images/masks
    """
    test_dir = model_dir / "test"
    if not test_dir.exists():
        print(f"  Skipping {model_dir.name} (no test directory)")
        return
    
    predictions_dir = test_dir / "predictions"
    if not predictions_dir.exists():
        print(f"  Skipping {model_dir.name} (no predictions)")
        return

    try:
        images_dir = find_training_images_dir(Path(data_root))
        masks_dir = find_combined_masks_dir(Path(data_root))
    except FileNotFoundError as exc:
        print(f"  Skipping {model_dir.name} ({exc})")
        return
    
    # Create output directories
    simple_overlays_dir = test_dir / "simple_overlays"
    difference_dir = test_dir / "difference"
    difference_v2_dir = test_dir / "difference_v2"
    difference_v3_dir = test_dir / "difference_v3"
    difference_v4_dir = test_dir / "difference_v4"
    difference_5_dir = test_dir / "difference_5"  # Alias for difference without text
    gt_vs_pred_dir = test_dir / "gt_vs_pred"
    predictions_multiclass_dir = test_dir / "predictions_multiclass"
    
    # Cleaned versions (no text)
    cleaned_dir = test_dir / "cleaned"
    cleaned_simple_dir = cleaned_dir / "simple_overlays"
    cleaned_diff_dir = cleaned_dir / "difference"
    cleaned_diff_v2_dir = cleaned_dir / "difference_v2"
    cleaned_diff_v3_dir = cleaned_dir / "difference_v3"
    cleaned_diff_5_dir = cleaned_dir / "difference_5"
    cleaned_gt_vs_pred_dir = cleaned_dir / "gt_vs_pred"
    
    for d in [simple_overlays_dir, difference_dir, difference_v2_dir, difference_v3_dir,
              difference_v4_dir, difference_5_dir, gt_vs_pred_dir, predictions_multiclass_dir,
              cleaned_simple_dir, cleaned_diff_dir, cleaned_diff_v2_dir, cleaned_diff_v3_dir,
              cleaned_diff_5_dir, cleaned_gt_vs_pred_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each prediction
    pred_files = list(predictions_dir.glob("*.png"))
    
    for pred_file in tqdm(pred_files, desc=f"  {model_dir.parent.name}/{model_dir.name}"):
        case_id = pred_file.stem.replace("_pred", "")
        
        # Load files (images/ or case*_dicom_png/; combined_masks via path helpers)
        image_path = images_dir / f"{case_id}.png"
        mask_path = masks_dir / f"{case_id}_combined_mask.png"
        
        if not image_path.exists() or not mask_path.exists():
            print(f"    Warning: Missing files for {case_id}")
            continue
        
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt_mask is None or pred_mask is None:
            print(f"    Warning: Failed to load {case_id}")
            continue
        
        # Generate visualizations
        
        # 1. Simple overlays
        simple_overlay = create_simple_overlay(image, pred_mask, with_text=True)
        cv2.imwrite(str(simple_overlays_dir / f"{case_id}.png"), simple_overlay)
        
        simple_overlay_clean = create_simple_overlay(image, pred_mask, with_text=False)
        cv2.imwrite(str(cleaned_simple_dir / f"{case_id}.png"), simple_overlay_clean)
        
        # 2. Difference maps
        diff_map = create_difference_map(gt_mask, pred_mask, with_legend=True)
        cv2.imwrite(str(difference_dir / f"{case_id}.png"), diff_map)
        cv2.imwrite(str(difference_v2_dir / f"{case_id}.png"), diff_map)  # v2 is same as difference
        
        diff_map_clean = create_difference_map(gt_mask, pred_mask, with_legend=False)
        cv2.imwrite(str(cleaned_diff_dir / f"{case_id}.png"), diff_map_clean)
        cv2.imwrite(str(cleaned_diff_v2_dir / f"{case_id}.png"), diff_map_clean)
        cv2.imwrite(str(difference_5_dir / f"{case_id}.png"), diff_map_clean)
        cv2.imwrite(str(cleaned_diff_5_dir / f"{case_id}.png"), diff_map_clean)
        
        # 3. Difference v3 (on original image)
        diff_v3 = create_difference_v3(image, gt_mask, pred_mask, with_legend=True)
        cv2.imwrite(str(difference_v3_dir / f"{case_id}.png"), diff_v3)
        
        diff_v3_clean = create_difference_v3(image, gt_mask, pred_mask, with_legend=False)
        cv2.imwrite(str(cleaned_diff_v3_dir / f"{case_id}.png"), diff_v3_clean)
        
        # 4. Difference v4 (PDF)
        save_difference_as_pdf(gt_mask, pred_mask, str(difference_v4_dir / f"{case_id}.pdf"))
        
        # 5. GT vs Pred side-by-side
        gt_vs_pred = create_gt_vs_pred_overlay(image, gt_mask, pred_mask)
        cv2.imwrite(str(gt_vs_pred_dir / f"{case_id}.png"), gt_vs_pred)
        cv2.imwrite(str(cleaned_gt_vs_pred_dir / f"{case_id}.png"), gt_vs_pred)  # No text in this one
        
        # 6. Predictions multiclass (copy)
        cv2.imwrite(str(predictions_multiclass_dir / f"{case_id}_pred_multiclass.png"), pred_mask)


def main():
    """Process all models."""
    import argparse

    from ..paths import case_real_data_dir
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["case1", "case2"], required=True)
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=None,
        help="Override bundle root (default: data/prostate/case1 or case2 via case_real_data_dir)",
    )
    args = parser.parse_args()
    
    base_dir = args.base_dir.expanduser().resolve() if args.base_dir else case_real_data_dir(args.case)
    
    viz_dir = base_dir / "outputs" / "visualizations"
    data_root = (base_dir / "data") if (base_dir / "data").is_dir() else base_dir
    
    print(f"\n{'='*80}")
    print(f"GENERATING ADDITIONAL VISUALIZATIONS - {args.case.upper()}")
    print(f"{'='*80}")
    print(f"Visualization dir: {viz_dir}")
    print(f"Data root: {data_root}")
    print(f"{'='*80}\n")
    
    # Find all model directories
    model_dirs = []
    for model_type in ["Baseline", "Optimization"]:
        type_dir = viz_dir / model_type
        if type_dir.exists():
            model_dirs.extend([d for d in type_dir.iterdir() if d.is_dir() and (d / "test").exists()])
    
    print(f"Found {len(model_dirs)} models to process\n")
    
    # Process each model
    for model_dir in model_dirs:
        process_model_visualizations(model_dir, data_root)
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
