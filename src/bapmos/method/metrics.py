"""
Boundary metrics for ``bapmos.method`` (standalone; no import from legacy core).

Mean Surface Distance (MSD / ASSD) is used for curriculum rewards and validation.
Empty boundaries return ``None`` so callers can skip missing surfaces.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def compute_boundary_mask(binary_mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Extract boundary pixels from a binary mask."""
    kernel = np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
    return binary_mask.astype(np.uint8) - eroded


def get_boundary_points(binary_mask: np.ndarray) -> np.ndarray:
    """Return (N, 2) boundary coordinates in (x, y) order."""
    boundary = compute_boundary_mask(binary_mask, thickness=1)
    coords = np.argwhere(boundary > 0)
    if len(coords) == 0:
        return np.array([]).reshape(0, 2)
    return coords[:, [1, 0]]


def mean_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pixel_spacing: Tuple[float, float] = (1.0, 1.0),
) -> Optional[float]:
    """
    Mean Surface Distance (MSD / ASSD) in millimeters (or pixel-units if spacing is 1).

    MSD(P, G) = mean( {d(p, G) for p in P} ∪ {d(g, P) for g in G} )
    """
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)

    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return None

    spacing = np.array(pixel_spacing, dtype=np.float64)
    distances_pred_to_gt = []
    for p in pred_boundary:
        p_mm = p * spacing
        gt_mm = gt_boundary * spacing
        dists = np.sqrt(np.sum((gt_mm - p_mm) ** 2, axis=1))
        distances_pred_to_gt.append(float(np.min(dists)))

    distances_gt_to_pred = []
    for g in gt_boundary:
        g_mm = g * spacing
        pred_mm = pred_boundary * spacing
        dists = np.sqrt(np.sum((pred_mm - g_mm) ** 2, axis=1))
        distances_gt_to_pred.append(float(np.min(dists)))

    all_distances = np.array(distances_pred_to_gt + distances_gt_to_pred)
    return float(np.mean(all_distances))
