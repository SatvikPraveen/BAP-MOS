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
from typing import Dict, Tuple, Optional
import cv2
from scipy.spatial import cKDTree


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
    
    # Boundary = original - eroded (inner boundary: foreground pixels that lie
    # within `thickness` px of background; 8-connected for thickness=1)
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


def _directed_min_distances(
    src: np.ndarray, dst: np.ndarray, pixel_spacing: Tuple[float, float]
) -> np.ndarray:
    """For each src point, the Euclidean distance (mm) to the nearest dst point.

    Exact nearest-neighbour search (k-d tree) on the spacing-scaled coordinates. Each distance
    is computed with the same arithmetic as the all-pairs form (subtract, square, sum the two
    coordinates, sqrt), so results equal the brute-force minimum (tested).
    """
    spacing = np.array(pixel_spacing)
    src_mm = src * spacing
    dst_mm = dst * spacing
    dist, _ = cKDTree(dst_mm).query(src_mm, k=1)
    return np.asarray(dist, dtype=np.float64)


def _symmetric_boundary_distances(
    pred_mask: np.ndarray, gt_mask: np.ndarray, pixel_spacing: Tuple[float, float]
) -> Optional[np.ndarray]:
    """Pred-to-GT and GT-to-pred boundary distances (mm), or None if either boundary is empty."""
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return None
    return np.concatenate([
        _directed_min_distances(pred_boundary, gt_boundary, pixel_spacing),
        _directed_min_distances(gt_boundary, pred_boundary, pixel_spacing),
    ])


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
    all_distances = _symmetric_boundary_distances(pred_mask, gt_mask, pixel_spacing)
    if all_distances is None:
        return None
    return float(np.percentile(all_distances, 95))


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
    all_distances = _symmetric_boundary_distances(pred_mask, gt_mask, pixel_spacing)
    if all_distances is None:
        return None
    return float(np.mean(all_distances))


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
        # One distance computation serves both metrics.
        all_distances = _symmetric_boundary_distances(pred_binary, gt_binary, pixel_spacing)
        hd95 = None if all_distances is None else float(np.percentile(all_distances, 95))
        msd = None if all_distances is None else float(np.mean(all_distances))
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
