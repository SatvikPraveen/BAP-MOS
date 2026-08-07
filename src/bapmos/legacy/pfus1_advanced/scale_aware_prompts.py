"""
Scale-aware prompt *geometry* for PFUS1-advanced experiments.

The RNG (``stable_rng_from_id(sample_id, seed + epoch)``) is unchanged — only the
sampling *rules* (ring width, box margin) adapt to image or organ scale.

Canonical ``configs/pfus1`` uses fixed ``ring_width: 20`` and is not affected unless
``ring_width_mode`` / ``box_margin_mode`` are set explicitly in that config.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from bapmos.multiorgan.dataset_multi_organ import (
    REAL_CLINICAL_ORGAN_TO_CLASS,
    sample_negative_points_from_ring,
    sample_positive_points,
)


def is_scale_aware_prompt_geometry(cfg: Dict[str, Any]) -> bool:
    """True when any non-fixed prompt geometry rule is enabled."""
    ring_mode = str(cfg.get("ring_width_mode", "fixed")).lower()
    box_mode = str(cfg.get("box_margin_mode", "none")).lower()
    return ring_mode != "fixed" or box_mode not in ("none", "fixed", "")


def resolve_ring_width(
    cfg: Dict[str, Any],
    *,
    image_hw: Optional[Tuple[int, int]] = None,
    organ_area_px: Optional[int] = None,
) -> int:
    """
    Resolve a single ring width (image-level or organ-level modes).

    Modes (``ring_width_mode``):
      - ``fixed``: ``ring_width`` pixels
      - ``scale_image``: ``max(ring_width_min, round(ring_width_image_frac * min(H, W)))``
      - ``scale_organ``: ``max(ring_width_min, round(ring_width_organ_frac * sqrt(area)))``
    """
    mode = str(cfg.get("ring_width_mode", "fixed")).lower()
    if mode == "fixed":
        return int(cfg.get("ring_width", 20))

    ring_min = int(cfg.get("ring_width_min", 8))

    if mode == "scale_image":
        if image_hw is None:
            return int(cfg.get("ring_width", 20))
        h, w = image_hw
        frac = float(cfg.get("ring_width_image_frac", 0.03))
        return max(ring_min, int(round(frac * min(h, w))))

    if mode == "scale_organ":
        if organ_area_px is None or organ_area_px <= 0:
            return int(cfg.get("ring_width", 20))
        frac = float(cfg.get("ring_width_organ_frac", 0.10))
        return max(ring_min, int(round(frac * math.sqrt(float(organ_area_px)))))

    raise ValueError(f"Unknown ring_width_mode: {mode!r}")


def resolve_ring_width_for_organ(
    cfg: Dict[str, Any],
    organ_mask: np.ndarray,
    *,
    image_hw: Tuple[int, int],
) -> int:
    """Per-organ ring width (organ-scale mode uses that organ's area)."""
    mode = str(cfg.get("ring_width_mode", "fixed")).lower()
    area = int(organ_mask.sum())
    if mode == "scale_organ":
        return resolve_ring_width(cfg, organ_area_px=area)
    return resolve_ring_width(cfg, image_hw=image_hw)


def expand_box_proportional(
    box: Union[np.ndarray, Sequence[int]],
    margin_frac: float,
    *,
    image_hw: Optional[Tuple[int, int]] = None,
    margin_min_px: int = 1,
) -> np.ndarray:
    """
    Expand box by ``margin_frac`` of its width/height (scale-aware box context).

    ``box``: [x_min, y_min, x_max, y_max] (list or ndarray from ``compute_bounding_box``)
    """
    box_arr = np.asarray(box, dtype=np.int64)
    x_min, y_min, x_max, y_max = (int(v) for v in box_arr.tolist())
    bw = max(1, int(x_max - x_min))
    bh = max(1, int(y_max - y_min))
    mx = max(margin_min_px, int(round(margin_frac * bw)))
    my = max(margin_min_px, int(round(margin_frac * bh)))

    x_min = x_min - mx
    y_min = y_min - my
    x_max = x_max + mx
    y_max = y_max + my

    if image_hw is not None:
        h, w = image_hw
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)

    if x_max <= x_min + 1:
        x_max = x_min + 2
    if y_max <= y_min + 1:
        y_max = y_min + 2
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


def apply_box_margin(
    box: Optional[Union[np.ndarray, Sequence[int]]],
    cfg: Dict[str, Any],
    *,
    image_hw: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Apply configured box margin before optional pixel jitter."""
    if box is None:
        return None
    mode = str(cfg.get("box_margin_mode", "none")).lower()
    if mode in ("none", "fixed", ""):
        return box
    if mode == "proportional":
        frac = float(cfg.get("box_margin_frac", 0.05))
        min_px = int(cfg.get("box_margin_min_px", 1))
        return expand_box_proportional(
            box, frac, image_hw=image_hw, margin_min_px=min_px
        )
    raise ValueError(f"Unknown box_margin_mode: {mode!r}")


def sample_per_organ_points_with_scale_aware_geometry(
    multi_class_mask: np.ndarray,
    cfg: Dict[str, Any],
    *,
    num_pos_per_organ: int = 1,
    num_neg: int = 3,
    rng=None,
    organ_to_class: Optional[Dict[str, int]] = None,
) -> Dict[str, Optional[Dict[str, np.ndarray]]]:
    """
    Per-organ positive/negative points with scale-aware ring width per organ.

    Uses the same ``rng`` as fixed-geometry sampling — only ``ring_width`` varies.
    """
    if organ_to_class is None:
        organ_to_class = REAL_CLINICAL_ORGAN_TO_CLASS
    if rng is None:
        rng = np.random.default_rng()

    image_hw = (int(multi_class_mask.shape[0]), int(multi_class_mask.shape[1]))
    point_sets: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    class_to_key = {cid: key for key, cid in organ_to_class.items()}

    for class_id in sorted(class_to_key.keys()):
        organ_name = class_to_key[class_id]
        organ_mask = (multi_class_mask == class_id).astype(np.uint8)

        if organ_mask.sum() == 0:
            point_sets[organ_name] = None
            continue

        ring_w = resolve_ring_width_for_organ(cfg, organ_mask, image_hw=image_hw)
        pos_points = sample_positive_points(organ_mask, num_pos_per_organ, rng=rng)
        if pos_points is None:
            point_sets[organ_name] = None
            continue

        neg_points = sample_negative_points_from_ring(organ_mask, ring_w, num_neg, rng=rng)
        if neg_points is not None:
            all_points = np.vstack([pos_points, neg_points])
            all_labels = np.concatenate([
                np.ones(len(pos_points), dtype=np.int32),
                np.zeros(len(neg_points), dtype=np.int32),
            ])
        else:
            all_points = pos_points
            all_labels = np.ones(len(pos_points), dtype=np.int32)

        point_sets[organ_name] = {"points": all_points, "labels": all_labels}

    return point_sets


def prompt_geometry_summary(cfg: Dict[str, Any], *, image_hw: Tuple[int, int]) -> Dict[str, Any]:
    """Log-friendly snapshot of resolved geometry for a frame size."""
    return {
        "ring_width_mode": cfg.get("ring_width_mode", "fixed"),
        "ring_width_px": resolve_ring_width(cfg, image_hw=image_hw),
        "box_margin_mode": cfg.get("box_margin_mode", "none"),
        "box_margin_frac": cfg.get("box_margin_frac"),
        "prompt_geometry_profile": cfg.get("prompt_geometry_profile", "unspecified"),
    }
