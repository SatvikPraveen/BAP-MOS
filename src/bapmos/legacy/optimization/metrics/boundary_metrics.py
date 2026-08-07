"""
Boundary-Aware Metrics for Medical Image Segmentation

Implements:
1. Hausdorff Distance 95th percentile (HD95) in mm
2. Mean Surface Distance (MSD / ASSD) in mm  
3. Mean Symmetric Boundary Distance in mm
4. Dice coefficient (secondary, reporting only)
5. IoU (secondary, reporting only)

All distance metrics are computed in millimeters using pixel spacing.
Handles empty masks appropriately.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, Tuple, Optional
import cv2


def compute_boundary_mask(binary_mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask.
    
    Args:
        binary_mask: (H, W) binary mask {0, 1}
        thickness: Boundary thickness in pixels
    
    Returns:
        boundary_mask: (H, W) binary boundary mask
    """
    # Erode mask
    kernel = np.ones((2*thickness+1, 2*thickness+1), np.uint8)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
    
    # Boundary = original - eroded
    boundary = binary_mask.astype(np.uint8) - eroded
    
    return boundary


def get_boundary_points(binary_mask: np.ndarray) -> np.ndarray:
    """
    Get (x, y) coordinates of boundary pixels.
    
    Args:
        binary_mask: (H, W) binary mask {0, 1}
    
    Returns:
        points: (N, 2) array of (x, y) coordinates
    """
    boundary = compute_boundary_mask(binary_mask, thickness=1)
    coords = np.argwhere(boundary > 0)  # (N, 2) in (row, col) format
    
    if len(coords) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to (x, y) format
    points = coords[:, [1, 0]]  # Swap columns: (col, row) = (x, y)
    
    return points


def hausdorff_distance_95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> Optional[float]:
    """
    Compute 95th percentile Hausdorff Distance in millimeters.
    
    HD95(P, G) = 95th percentile of {d(p, G) for p in P} ∪ {d(g, P) for g in G}
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
        pixel_spacing: (spacing_x, spacing_y) in mm/pixel
    
    Returns:
        HD95 in mm, or None if either mask has no boundary
    """
    # Get boundary points
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)
    
    # Handle empty boundaries
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return None
    
    # Compute distances from pred to GT
    distances_pred_to_gt = []
    for p in pred_boundary:
        # Convert to mm
        p_mm = p * np.array(pixel_spacing)
        gt_mm = gt_boundary * np.array(pixel_spacing)
        
        # Euclidean distances to all GT boundary points
        dists = np.sqrt(np.sum((gt_mm - p_mm)**2, axis=1))
        min_dist = np.min(dists)
        distances_pred_to_gt.append(min_dist)
    
    # Compute distances from GT to pred
    distances_gt_to_pred = []
    for g in gt_boundary:
        # Convert to mm
        g_mm = g * np.array(pixel_spacing)
        pred_mm = pred_boundary * np.array(pixel_spacing)
        
        # Euclidean distances to all pred boundary points
        dists = np.sqrt(np.sum((pred_mm - g_mm)**2, axis=1))
        min_dist = np.min(dists)
        distances_gt_to_pred.append(min_dist)
    
    # Combine all distances
    all_distances = np.array(distances_pred_to_gt + distances_gt_to_pred)
    
    # 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def mean_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> Optional[float]:
    """
    Compute Mean Surface Distance (MSD / ASSD) in millimeters.
    
    MSD(P, G) = mean( {d(p, G) for p in P} ∪ {d(g, P) for g in G} )
    
    Also known as Average Symmetric Surface Distance (ASSD).
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
        pixel_spacing: (spacing_x, spacing_y) in mm/pixel
    
    Returns:
        MSD in mm, or None if either mask has no boundary
    """
    # Get boundary points
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)
    
    # Handle empty boundaries
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return None
    
    # Compute distances from pred to GT
    distances_pred_to_gt = []
    for p in pred_boundary:
        # Convert to mm
        p_mm = p * np.array(pixel_spacing)
        gt_mm = gt_boundary * np.array(pixel_spacing)
        
        # Euclidean distances to all GT boundary points
        dists = np.sqrt(np.sum((gt_mm - p_mm)**2, axis=1))
        min_dist = np.min(dists)
        distances_pred_to_gt.append(min_dist)
    
    # Compute distances from GT to pred
    distances_gt_to_pred = []
    for g in gt_boundary:
        # Convert to mm
        g_mm = g * np.array(pixel_spacing)
        pred_mm = pred_boundary * np.array(pixel_spacing)
        
        # Euclidean distances to all pred boundary points
        dists = np.sqrt(np.sum((pred_mm - g_mm)**2, axis=1))
        min_dist = np.min(dists)
        distances_gt_to_pred.append(min_dist)
    
    # Mean of all distances
    all_distances = np.array(distances_pred_to_gt + distances_gt_to_pred)
    msd = np.mean(all_distances)
    
    return float(msd)


def boundary_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> Optional[float]:
    """
    Compute mean symmetric boundary distance in millimeters.
    
    Same as MSD / ASSD (included for API completeness).
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
        pixel_spacing: (spacing_x, spacing_y) in mm/pixel
    
    Returns:
        Boundary distance in mm, or None if either mask has no boundary
    """
    return mean_surface_distance(pred_mask, gt_mask, pixel_spacing)


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient (secondary metric, reporting only).
    
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
    
    Returns:
        Dice coefficient in [0, 1]
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    
    if union == 0:
        # Both masks empty
        return 1.0
    
    dice = (2.0 * intersection) / union
    return float(dice)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (secondary metric, reporting only).
    
    IoU = |P ∩ G| / |P ∪ G|
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
    
    Returns:
        IoU in [0, 1]
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    if union == 0:
        # Both masks empty
        return 1.0
    
    iou = intersection / union
    return float(iou)


def compute_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    organ_name: str = "organ"
) -> Dict:
    """
    Compute all metrics for a single slice/organ.
    
    Args:
        pred_mask: (H, W) binary prediction mask {0, 1}
        gt_mask: (H, W) binary ground truth mask {0, 1}
        pixel_spacing: (spacing_x, spacing_y) in mm/pixel
        organ_name: Name of organ for logging
    
    Returns:
        dict: {
            "organ": str,
            "hd95_mm": float or None,
            "msd_mm": float or None,
            "boundary_dist_mm": float or None,
            "dice": float,
            "iou": float,
            "gt_empty": bool,
            "pred_empty": bool,
            "valid_boundary": bool,  # True if boundary metrics are valid
            "false_positive_empty_gt": bool,
            "empty_pred_failure": bool
        }
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Check empty cases
    gt_empty = (gt_binary.sum() == 0)
    pred_empty = (pred_binary.sum() == 0)
    
    # Compute secondary metrics (always defined)
    dice = dice_coefficient(pred_binary, gt_binary)
    iou = iou_score(pred_binary, gt_binary)
    
    # Compute boundary-aware metrics
    if gt_empty and pred_empty:
        # Case A: Both empty → perfect match, but no boundary to measure
        hd95 = None
        msd = None
        boundary_dist = None
        valid_boundary = False
        false_positive_empty_gt = False
        empty_pred_failure = False
    
    elif gt_empty and not pred_empty:
        # Case B: GT empty, Pred non-empty → false positive
        hd95 = None
        msd = None
        boundary_dist = None
        valid_boundary = False
        false_positive_empty_gt = True
        empty_pred_failure = False
    
    elif not gt_empty and pred_empty:
        # Case C: GT non-empty, Pred empty → catastrophic failure
        hd95 = None
        msd = None
        boundary_dist = None
        valid_boundary = False
        false_positive_empty_gt = False
        empty_pred_failure = True
    
    else:
        # Case D: Both non-empty → compute boundary metrics
        hd95 = hausdorff_distance_95(pred_binary, gt_binary, pixel_spacing)
        msd = mean_surface_distance(pred_binary, gt_binary, pixel_spacing)
        boundary_dist = msd  # Same as MSD
        
        # Check if boundary metrics are valid
        valid_boundary = (hd95 is not None and msd is not None)
        false_positive_empty_gt = False
        empty_pred_failure = False
    
    return {
        "organ": organ_name,
        "hd95_mm": hd95,
        "msd_mm": msd,
        "boundary_dist_mm": boundary_dist,
        "dice": dice,
        "iou": iou,
        "gt_empty": gt_empty,
        "pred_empty": pred_empty,
        "valid_boundary": valid_boundary,
        "false_positive_empty_gt": false_positive_empty_gt,
        "empty_pred_failure": empty_pred_failure
    }


if __name__ == "__main__":
    # Test the metrics
    print("\n=== Boundary Metrics Test ===\n")
    
    # Create test masks
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[30:70, 30:70] = 1  # 40x40 square
    
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[25:65, 25:65] = 1  # Slightly shifted 40x40 square
    
    # Assume pixel spacing of 0.5 mm/pixel
    spacing = (0.5, 0.5)
    
    # Compute metrics
    metrics = compute_all_metrics(pred, gt, spacing, organ_name="test")
    
    print("Test case: 40x40 square vs slightly shifted square")
    print(f"Pixel spacing: {spacing} mm/pixel")
    print(f"\nPrimary metrics (boundary-aware):")
    print(f"  HD95: {metrics['hd95_mm']:.3f} mm" if metrics['hd95_mm'] else "  HD95: None")
    print(f"  MSD:  {metrics['msd_mm']:.3f} mm" if metrics['msd_mm'] else "  MSD:  None")
    print(f"\nSecondary metrics (reporting only):")
    print(f"  Dice: {metrics['dice']:.3f}")
    print(f"  IoU:  {metrics['iou']:.3f}")
    print(f"\nFlags:")
    print(f"  Valid boundary: {metrics['valid_boundary']}")
    print(f"  GT empty: {metrics['gt_empty']}")
    print(f"  Pred empty: {metrics['pred_empty']}")
