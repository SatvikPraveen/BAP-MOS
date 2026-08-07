"""
Organ-scale prompt geometry for BAP-MOS.

Deterministic prompt generation from ground-truth multi-class masks.

Scale-aware negative ring width (organ area A_o in pixels)::

    r_o = max(6, ⌊0.1 · √A_o⌋)

implemented as ``max(6, int(0.1 * sqrt(A_o)))``.

Point prompts: one positive inside M_o; three negatives from the exclusive ring
dilate(M_o, r_o) \\ M_o.

Randomness is deterministic per (sample_id, epoch) via ``stable_rng_from_id``.

Asymmetric fallback: if point sampling fails (tiny or distorted organ), silently
set ``effective_mode='box'`` so training never crashes on empty rings.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np

from bapmos.method.types import Action, EffectiveMode, OrganPrompts


def stable_rng_from_id(sample_id: str, base_seed: int) -> np.random.Generator:
    """FNV-1a-style 32-bit hash → deterministic NumPy Generator (not a stdlib hash)."""
    h = 2166136261
    for c in sample_id + str(base_seed):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def ring_width_px(organ_area: int, r_min: int = 6, alpha: float = 0.1) -> int:
    """
    Scale-organ ring: r_o = max(r_min, ⌊α · √A_o⌋) with α=0.1, r_min=6.
    """
    if organ_area <= 0:
        return r_min
    return max(r_min, int(alpha * math.sqrt(float(organ_area))))


def resolve_ring_width(
    organ_area: int,
    *,
    ring_mode: str = "scale_organ",
    ring_width_fixed: int = 20,
    r_min: int = 6,
    alpha: float = 0.1,
) -> int:
    """
    Resolve negative-point ring width in pixels.

    - ``fixed``: constant ``ring_width_fixed`` (legacy BAP-MOS, e.g. 20 px).
    - ``scale_organ``: r_o = max(r_min, int(0.1 * sqrt(A_o))).
    """
    mode = str(ring_mode).lower().replace("-", "_")
    if mode == "fixed":
        return max(1, int(ring_width_fixed))
    if mode in ("scale_organ", "organ", "scale"):
        return ring_width_px(organ_area, r_min=r_min, alpha=alpha)
    raise ValueError(f"Unknown ring_mode: {ring_mode!r}; use 'fixed' or 'scale_organ'")


def compute_bounding_box(binary_mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight axis-aligned box [x_min, y_min, x_max, y_max] or None if empty."""
    if binary_mask.sum() == 0:
        return None
    rows, cols = np.where(binary_mask > 0)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


def jitter_box(
    box: np.ndarray,
    jitter_px: int,
    image_hw: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Optional uniform edge jitter during training."""
    if jitter_px <= 0:
        return box.copy()
    h, w = image_hw
    x_min, y_min, x_max, y_max = (int(v) for v in box.tolist())
    x_min = max(0, x_min - rng.integers(0, jitter_px + 1))
    y_min = max(0, y_min - rng.integers(0, jitter_px + 1))
    x_max = min(w - 1, x_max + rng.integers(0, jitter_px + 1))
    y_max = min(h - 1, y_max + rng.integers(0, jitter_px + 1))
    if x_max <= x_min + 1:
        x_max = x_min + 2
    if y_max <= y_min + 1:
        y_max = y_min + 2
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int64)


def sample_positive_points(
    binary_mask: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Sample (num_points, 2) positive (x, y) inside mask."""
    coords = np.argwhere(binary_mask > 0)
    if len(coords) == 0:
        return None
    replace = len(coords) < num_points
    idx = rng.choice(len(coords), size=num_points, replace=replace)
    sampled = coords[idx]
    return sampled[:, [1, 0]].astype(np.float64)


def sample_negative_points_from_ring(
    organ_mask: np.ndarray,
    ring_width: int,
    num_neg: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Sample negatives from dilate(M, r) \\ M."""
    k = 2 * ring_width + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(organ_mask.astype(np.uint8), kernel, iterations=1)
    ring = dilated.astype(np.int16) - organ_mask.astype(np.int16)
    ring = (ring > 0).astype(np.uint8)
    if ring.sum() == 0:
        return None
    coords = np.argwhere(ring > 0)
    replace = len(coords) < num_neg
    idx = rng.choice(len(coords), size=num_neg, replace=replace)
    sampled = coords[idx]
    return sampled[:, [1, 0]].astype(np.float64)


def sample_point_prompts(
    organ_mask: np.ndarray,
    *,
    num_pos: int = 1,
    num_neg: int = 3,
    ring_mode: str = "scale_organ",
    ring_width_fixed: int = 20,
    r_min: int = 6,
    alpha: float = 0.1,
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Return dict with 'points' (P, 2) and 'labels' (P,) or None on failure.
    """
    area = int(organ_mask.sum())
    if area == 0:
        return None
    r_o = resolve_ring_width(
        area,
        ring_mode=ring_mode,
        ring_width_fixed=ring_width_fixed,
        r_min=r_min,
        alpha=alpha,
    )
    pos = sample_positive_points(organ_mask, num_pos, rng)
    if pos is None:
        return None
    neg = sample_negative_points_from_ring(organ_mask, r_o, num_neg, rng)
    if neg is None:
        return None
    points = np.vstack([pos, neg])
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)]
    )
    return {"points": points, "labels": labels}


def build_organ_prompt(
    organ_key: str,
    organ_mask: np.ndarray,
    mode: Action,
    *,
    image_hw: tuple[int, int],
    rng: np.random.Generator,
    train: bool,
    box_jitter_px: int = 0,
    num_pos: int = 1,
    num_neg: int = 3,
    ring_mode: str = "scale_organ",
    ring_width_fixed: int = 20,
    r_min: int = 6,
    alpha: float = 0.1,
) -> Optional[OrganPrompts]:
    """
    Build prompts for one organ; downgrade to box silently when points fail.
    """
    box = compute_bounding_box(organ_mask)
    if box is None:
        return None
    if train and box_jitter_px > 0:
        box = jitter_box(box, box_jitter_px, image_hw, rng)

    effective_mode: EffectiveMode = mode
    points_xy: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    if mode in ("point", "both"):
        point_set = sample_point_prompts(
            organ_mask,
            num_pos=num_pos,
            num_neg=num_neg,
            ring_mode=ring_mode,
            ring_width_fixed=ring_width_fixed,
            r_min=r_min,
            alpha=alpha,
            rng=rng,
        )
        if point_set is None:
            effective_mode = "box"
        else:
            points_xy = point_set["points"]
            labels = point_set["labels"]

    if effective_mode == "box":
        return OrganPrompts(
            organ_key=organ_key,
            mode=mode,
            effective_mode="box",
            box_xyxy=box,
            points_xy=None,
            labels=None,
        )

    return OrganPrompts(
        organ_key=organ_key,
        mode=mode,
        effective_mode=effective_mode,
        box_xyxy=box,
        points_xy=points_xy,
        labels=labels,
    )


def build_prompts_for_slice(
    multi_class_mask: np.ndarray,
    organ_to_class: Mapping[str, int],
    modes: Mapping[str, Action],
    sample_id: str,
    epoch: int,
    base_seed: int,
    train: bool,
    *,
    box_jitter_px: int = 0,
    num_pos: int = 1,
    num_neg: int = 3,
    ring_mode: str = "scale_organ",
    ring_width_fixed: int = 20,
    r_min: int = 6,
    alpha: float = 0.1,
) -> Dict[str, OrganPrompts]:
    """
    Generate per-organ prompts for all organs listed in ``modes`` that are present.

    One shared RNG is advanced in ``modes`` iteration order, so callers must pass
    a deterministically ordered mapping (insertion-ordered dict is fine in Py3.7+).
    """
    h, w = multi_class_mask.shape[:2]
    rng = stable_rng_from_id(f"{sample_id}|{epoch}", base_seed)
    out: Dict[str, OrganPrompts] = {}

    for organ_key, mode in modes.items():
        if organ_key not in organ_to_class:
            continue
        class_id = organ_to_class[organ_key]
        organ_mask = (multi_class_mask == class_id).astype(np.uint8)
        if organ_mask.sum() == 0:
            continue
        prompt = build_organ_prompt(
            organ_key,
            organ_mask,
            mode,
            image_hw=(h, w),
            rng=rng,
            train=train,
            box_jitter_px=box_jitter_px,
            num_pos=num_pos,
            num_neg=num_neg,
            ring_mode=ring_mode,
            ring_width_fixed=ring_width_fixed,
            r_min=r_min,
            alpha=alpha,
        )
        if prompt is not None:
            out[organ_key] = prompt

    return out


def present_organ_keys(
    multi_class_mask: np.ndarray,
    organ_to_class: Mapping[str, int],
) -> List[str]:
    """Organs with at least one foreground pixel."""
    present = []
    for key, cid in organ_to_class.items():
        if (multi_class_mask == cid).any():
            present.append(key)
    return present
