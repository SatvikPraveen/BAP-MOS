"""
Grey-background TP / FP / FN difference maps (difference_v1 style).

Per-pixel multiclass confusion on a neutral canvas — no ultrasound underlay.
Matches the publication layout: green=TP, red=FP, blue=FN, grey=background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from bapmos.pdf_export import INFERENCE_PANEL_PDF_DPI, PDF_EXPORT_DPI

DIFFERENCE_V1_BG_RGB = (200, 200, 200)
DIFFERENCE_V1_TP_RGB = (0, 255, 0)
DIFFERENCE_V1_FP_RGB = (255, 0, 0)
DIFFERENCE_V1_FN_RGB = (0, 0, 255)

PNG_PANEL_DPI = PDF_EXPORT_DPI


def _align_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    h, w = hw
    if m.shape[:2] == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def build_difference_v1_rgb(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    foreground_class_ids: Sequence[int],
    image_hw: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Return RGB uint8 difference map on grey background."""
    if image_hw is None:
        image_hw = gt_mask.shape[:2]
    gt = _align_mask(gt_mask, image_hw)
    pred = _align_mask(pred_mask, image_hw)
    h, w = gt.shape[:2]

    all_tp = np.zeros((h, w), dtype=bool)
    all_fp = np.zeros((h, w), dtype=bool)
    all_fn = np.zeros((h, w), dtype=bool)

    for class_id in foreground_class_ids:
        gt_bin = gt == int(class_id)
        pred_bin = pred == int(class_id)
        all_tp |= gt_bin & pred_bin
        all_fp |= (~gt_bin) & pred_bin
        all_fn |= gt_bin & (~pred_bin)

    rgb = np.ones((h, w, 3), dtype=np.uint8)
    rgb[:] = DIFFERENCE_V1_BG_RGB
    rgb[all_tp] = DIFFERENCE_V1_TP_RGB
    rgb[all_fp] = DIFFERENCE_V1_FP_RGB
    rgb[all_fn] = DIFFERENCE_V1_FN_RGB
    return rgb


def save_difference_v1_panel(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    output_path: Path,
    foreground_class_ids: Sequence[int],
    image_hw: Optional[tuple[int, int]] = None,
    caption: Optional[str] = None,
    save_png: bool = True,
    save_pdf: bool = True,
    pdf_dpi: int = INFERENCE_PANEL_PDF_DPI,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save difference_v1 PNG (+ optional PDF) with top-centered TP/FP/FN legend.

    ``output_path`` must be a ``.png`` or ``.pdf`` file path (not a directory).
    A PNG is always written at ``PNG_PANEL_DPI`` (= ``PDF_EXPORT_DPI``, 350) when
    ``save_png`` is true; an optional sibling PDF uses the same DPI via ``pdf_dpi``.
    Both are matplotlib ``savefig`` from the in-memory figure (never PNG→PDF).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        raise ValueError(
            f"output_path must end with .png or .pdf (got {output_path})"
        )
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf") if save_pdf else None

    diff_rgb = build_difference_v1_rgb(
        gt_mask,
        pred_mask,
        foreground_class_ids=foreground_class_ids,
        image_hw=image_hw,
    )
    h, w = diff_rgb.shape[:2]
    panel_w_in = 3.25
    panel_h_in = panel_w_in * h / max(w, 1)
    legend_h_in = 0.32
    caption_h_in = 0.22 if caption else 0.0
    pad = 0.06

    fig_h = panel_h_in + legend_h_in + caption_h_in
    fig = plt.figure(figsize=(panel_w_in, fig_h))
    ratios = [legend_h_in, panel_h_in]
    if caption:
        ratios.append(caption_h_in)
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.04)

    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    legend_elements = [
        Patch(
            facecolor=tuple(c / 255.0 for c in DIFFERENCE_V1_TP_RGB),
            edgecolor="black",
            linewidth=0.5,
            label="TP",
        ),
        Patch(
            facecolor=tuple(c / 255.0 for c in DIFFERENCE_V1_FP_RGB),
            edgecolor="black",
            linewidth=0.5,
            label="FP",
        ),
        Patch(
            facecolor=tuple(c / 255.0 for c in DIFFERENCE_V1_FN_RGB),
            edgecolor="black",
            linewidth=0.5,
            label="FN",
        ),
    ]
    ax_legend.legend(
        handles=legend_elements,
        loc="center",
        ncol=3,
        fontsize=10,
        frameon=True,
    )

    ax = fig.add_subplot(gs[1])
    ax.imshow(diff_rgb)
    ax.axis("off")

    if caption:
        ax_cap = fig.add_subplot(gs[2])
        ax_cap.axis("off")
        ax_cap.text(0.5, 0.5, caption, ha="center", va="center", fontsize=11)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        fig.savefig(str(png_path), dpi=PNG_PANEL_DPI, bbox_inches="tight", pad_inches=pad)
    if pdf_path is not None and save_pdf:
        fig.savefig(
            str(pdf_path),
            dpi=int(pdf_dpi),
            format="pdf",
            bbox_inches="tight",
            pad_inches=pad,
        )
    plt.close(fig)
    return (png_path if save_png else None), (pdf_path if save_pdf else None)
