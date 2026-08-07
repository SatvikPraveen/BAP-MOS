"""
Image-only PFUS1 ultrasound field standardization (no mask leakage).

Pipeline per frame:
  1. Optional right-side legend / annotation strip removal (heuristic).
  2. Ultrasound cone bounding box from non-black foreground (+ margin).
  3. Aspect-preserving resize + letterbox pad to a fixed canvas.

The same crop box and letterbox mapping are applied to masks in
:mod:`bapmos.legacy.pfus1_advanced.build_advanced_bundle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessParams:
    """Recorded per frame for reproducibility and mask warping."""

    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    scale: float
    pad_left: int
    pad_top: int
    canvas_w: int
    canvas_h: int
    src_w: int
    src_h: int


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8, copy=False)


def detect_legend_x0(gray: np.ndarray, *, min_strip_frac: float = 0.12) -> Optional[int]:
    """
    Heuristic: right-side legend / overlay panel (bright, non-cone columns).

    Returns x coordinate (inclusive) where legend strip starts, or None.
    """
    h, w = gray.shape[:2]
    if w < 80:
        return None

    fg = gray > 12
    col_fg_frac = fg.mean(axis=0)
    col_mean = gray.mean(axis=0)

    # Ultrasound cone usually occupies left/mid columns; legend is a bright right strip.
    search_from = int(w * (1.0 - min_strip_frac))
    x_candidates = []
    for x in range(search_from, w - 2):
        if col_fg_frac[x] < 0.08 and col_mean[x] > 90:
            x_candidates.append(x)
        elif col_fg_frac[x] < 0.15 and col_mean[x] > 140:
            x_candidates.append(x)

    if not x_candidates:
        return None
    return int(x_candidates[0])


def detect_ultrasound_bbox(
    gray: np.ndarray,
    *,
    intensity_thresh: int = 12,
    margin_frac: float = 0.02,
    legend_x0: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Largest foreground region bounding box with fractional margin."""
    h, w = gray.shape[:2]
    x_max = legend_x0 if legend_x0 is not None else w
    roi = gray[:, :x_max]
    fg = roi > intensity_thresh
    if not fg.any():
        return 0, 0, w, h

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    mx = max(2, int(round(margin_frac * (x1 - x0))))
    my = max(2, int(round(margin_frac * (y1 - y0))))
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    return x0, y0, x1, y1


def letterbox_resize(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    fill: int = 0,
    interp: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize preserving aspect ratio, pad to (target_w, target_h).

    Returns (out, scale, pad_left, pad_top).
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image for letterbox")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    if image.ndim == 3:
        border_value = (fill, fill, fill)
    else:
        border_value = fill

    out = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
    return out, scale, pad_left, pad_top


def preprocess_ultrasound_frame(
    image: np.ndarray,
    *,
    canvas_w: int = 768,
    canvas_h: int = 576,
    intensity_thresh: int = 12,
    margin_frac: float = 0.02,
    strip_legend: bool = True,
) -> Tuple[np.ndarray, PreprocessParams]:
    """
    Full image-only preprocess. Returns (letterboxed_gray_or_rgb, params).
    """
    gray = _as_gray_u8(image)
    src_h, src_w = gray.shape[:2]

    legend_x0 = detect_legend_x0(gray) if strip_legend else None
    x0, y0, x1, y1 = detect_ultrasound_bbox(
        gray,
        intensity_thresh=intensity_thresh,
        margin_frac=margin_frac,
        legend_x0=legend_x0,
    )
    cropped = gray[y0:y1, x0:x1]
    out, scale, pad_left, pad_top = letterbox_resize(
        cropped, canvas_w, canvas_h, fill=0, interp=cv2.INTER_LINEAR
    )

    params = PreprocessParams(
        crop_x0=x0,
        crop_y0=y0,
        crop_x1=x1,
        crop_y1=y1,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        src_w=src_w,
        src_h=src_h,
    )
    return out, params


def warp_mask_with_params(mask: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """Apply the same crop + letterbox (nearest) to a multiclass mask."""
    cropped = mask[params.crop_y0 : params.crop_y1, params.crop_x0 : params.crop_x1]
    out, _, _, _ = letterbox_resize(
        cropped,
        params.canvas_w,
        params.canvas_h,
        fill=0,
        interp=cv2.INTER_NEAREST,
    )
    return out.astype(mask.dtype, copy=False)
