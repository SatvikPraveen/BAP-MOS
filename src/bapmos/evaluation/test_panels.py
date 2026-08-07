"""
Multiclass test-split figure panels (taxonomy-aware).

PNG previews and publication PDFs are separate ``matplotlib.figure.Figure.savefig``
calls from the same in-memory figure (numpy RGB arrays via ``ax.imshow``).
PDFs are **never** produced by reading or embedding an on-disk PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from bapmos.data.organ_registry import OrganDefinition
from bapmos.pdf_export import INFERENCE_PANEL_PDF_DPI, PDF_EXPORT_DPI

PNG_PANEL_DPI = PDF_EXPORT_DPI
PDF_PANEL_EXPORT_DPI = INFERENCE_PANEL_PDF_DPI


def distance_unit_short(taxonomy_name: str) -> str:
    """PFUS1 uses pixel-native spacing; clinical/simulation use mm from DICOM-style spacing."""
    if taxonomy_name == "pfus1_eight_organ":
        return "px"
    return "mm"


def _matplotlib_save_panel_png(fig: Any, path: Path, *, pad_inches: float) -> None:
    """PNG via matplotlib at ``PNG_PANEL_DPI`` (= 350)."""
    fig.savefig(str(path), dpi=PNG_PANEL_DPI, bbox_inches="tight", pad_inches=pad_inches)


def _matplotlib_save_panel_pdf(fig: Any, path: Path, *, pad_inches: float, dpi: int) -> None:
    """PDF via matplotlib ``savefig(..., format='pdf')`` — not from PNG."""
    fig.savefig(
        str(path),
        dpi=int(dpi),
        format="pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
    )


def multiclass_mask_to_bgr(
    mask: np.ndarray,
    organ_definitions: Sequence[OrganDefinition],
) -> np.ndarray:
    """Render class-index mask (0=background) as a BGR image for ``cv2.imwrite``."""
    m = np.asarray(mask)
    out = np.zeros((*m.shape[:2], 3), dtype=np.uint8)
    for o in organ_definitions:
        b, g, r = o.color_bgr
        out[m == o.class_id] = (b, g, r)
    return out


def _ensure_rgb_u8(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _align_mask_to_image(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    h, w = image_hw
    if m.shape[:2] == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def semantic_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    organ_definitions: Sequence[OrganDefinition],
    *,
    alpha: float = 0.42,
) -> np.ndarray:
    """Semi-transparent multiclass overlay (RGB uint8)."""
    base = _ensure_rgb_u8(image_rgb).astype(np.float32)
    out = base.copy()
    for o in organ_definitions:
        m = mask == o.class_id
        if not np.any(m):
            continue
        r, g, b = float(o.color_bgr[2]), float(o.color_bgr[1]), float(o.color_bgr[0])
        col = np.array([r, g, b], dtype=np.float32)
        for c in range(3):
            ch = out[..., c]
            ch[m] = (1.0 - alpha) * ch[m] + alpha * col[c]
            out[..., c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def difference_blend_rgb(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    organ_definitions: Sequence[OrganDefinition],
) -> np.ndarray:
    """Legacy ultrasound-underlay TP/FP/FN blend (RGB).

    **Prefer** :func:`bapmos.evaluation.difference_v1.build_difference_v1_rgb` /
    :func:`save_difference_v1_panel` for public inference_output exports.
    Kept for historical ``legacy.optimization.inference`` callers.
    """
    img = _ensure_rgb_u8(image_rgb).astype(np.float32)
    gt = np.asarray(gt_mask)
    pred = np.asarray(pred_mask)
    h, w = gt.shape
    all_tp = np.zeros((h, w), dtype=bool)
    all_fp = np.zeros((h, w), dtype=bool)
    all_fn = np.zeros((h, w), dtype=bool)

    for o in organ_definitions:
        cid = o.class_id
        gt_b = gt == cid
        pr_b = pred == cid
        all_tp |= gt_b & pr_b
        all_fp |= (~gt_b) & pr_b
        all_fn |= gt_b & (~pr_b)

    overlay = img.copy()
    overlay[all_tp] = (0.0, 255.0, 0.0)
    overlay[all_fp] = (255.0, 0.0, 0.0)
    overlay[all_fn] = (0.0, 0.0, 255.0)

    mask_any = all_tp | all_fp | all_fn
    out = img.copy()
    out[mask_any] = 0.5 * img[mask_any] + 0.5 * overlay[mask_any]
    return np.clip(out, 0, 255).astype(np.uint8)


def save_pred_gt_difference_map(
    *,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_stem: str,
    output_dir: Path,
    organ_definitions: Sequence[OrganDefinition],
    taxonomy_name: Optional[str] = None,
    save_pdf: bool = True,
    save_png: bool = True,
    pdf_dpi: int = INFERENCE_PANEL_PDF_DPI,
) -> tuple[Path, Optional[Path]]:
    """
    Legacy ultrasound-underlay difference panel.

    **Public exports use** :func:`bapmos.evaluation.difference_v1.save_difference_v1_panel`.
    This helper remains for ``legacy.optimization.inference`` only.

    Writes ``difference/{stem}_diff.png`` and/or ``{stem}_diff.pdf`` at ``pdf_dpi``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img = _ensure_rgb_u8(image_rgb)
    gt_u8 = _align_mask_to_image(gt_mask, img.shape[:2])
    pr_u8 = _align_mask_to_image(pred_mask, img.shape[:2])

    diff_dir = output_dir / "difference"
    diff_dir.mkdir(parents=True, exist_ok=True)

    diff_rgb = difference_blend_rgb(img, gt_u8, pr_u8, organ_definitions)
    h, w = img.shape[:2]
    panel_w_in = 3.25
    panel_h_in = panel_w_in * h / max(w, 1)
    legend_h_in = 0.32
    pad = 0.06
    fig = plt.figure(figsize=(panel_w_in, panel_h_in + legend_h_in))
    gs = fig.add_gridspec(2, 1, height_ratios=[legend_h_in, panel_h_in], hspace=0.04)
    ax_legend = fig.add_subplot(gs[0])
    ax_legend.axis("off")
    legend_elements = [
        Patch(facecolor=(0.0, 0.9, 0.0), edgecolor="black", linewidth=0.5, label="TP"),
        Patch(facecolor=(0.9, 0.0, 0.0), edgecolor="black", linewidth=0.5, label="FP"),
        Patch(facecolor=(0.0, 0.0, 0.9), edgecolor="black", linewidth=0.5, label="FN"),
    ]
    ax_legend.legend(handles=legend_elements, loc="center", ncol=3, fontsize=9, frameon=True)
    ax = fig.add_subplot(gs[1])
    ax.imshow(diff_rgb)
    ax.axis("off")

    png_path = diff_dir / f"{sample_stem}_diff.png"
    pdf_path: Optional[Path] = None
    if save_pdf:
        pdf_path = diff_dir / f"{sample_stem}_diff.pdf"
        _matplotlib_save_panel_pdf(fig, pdf_path, pad_inches=pad, dpi=pdf_dpi)
    if save_png:
        _matplotlib_save_panel_png(fig, png_path, pad_inches=pad)
    plt.close(fig)
    return png_path, pdf_path


# Back-compat alias
save_difference_maps_by_label = save_pred_gt_difference_map


def save_overlay_pred_pair_png_pdf(
    *,
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    sample_stem: str,
    output_dir: Path,
    organ_definitions: Sequence[OrganDefinition],
    taxonomy_name: Optional[str] = None,
    save_pdf: bool = True,
    save_png: bool = True,
    pdf_dpi: int = INFERENCE_PANEL_PDF_DPI,
) -> tuple[Path, Optional[Path]]:
    """
    Side-by-side figure: left = ultrasound + prediction overlay, right = colored mask.

    PNG preview and PDF are separate matplotlib ``savefig`` calls (PDF at ``pdf_dpi``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    img = _ensure_rgb_u8(image_rgb)
    pr_u8 = _align_mask_to_image(pred_mask, img.shape[:2])
    left_rgb = semantic_overlay(img, pr_u8, organ_definitions)
    right_rgb = cv2.cvtColor(multiclass_mask_to_bgr(pr_u8, organ_definitions), cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    panel_w_in = 3.0
    panel_h_in = panel_w_in * h / max(w, 1)
    pad = 0.08
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w_in, panel_h_in + 0.35))
    for ax, (title, data) in zip(
        axes,
        [("Overlay (prediction on ultrasound)", left_rgb), ("Prediction mask", right_rgb)],
    ):
        ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if taxonomy_name:
        du = distance_unit_short(taxonomy_name)
        fig.suptitle(f"Test — {taxonomy_name} ({du})", fontsize=9, y=1.02)

    stem_path = output_dir / f"{sample_stem}_viz"
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.05)
    png_path = Path(f"{stem_path}.png")
    pdf_path: Optional[Path] = None
    if save_pdf:
        pdf_path = Path(f"{stem_path}.pdf")
        _matplotlib_save_panel_pdf(fig, pdf_path, pad_inches=pad, dpi=pdf_dpi)
    if save_png:
        _matplotlib_save_panel_png(fig, png_path, pad_inches=pad)
    plt.close(fig)
    return png_path, pdf_path


def save_multiclass_test_panel_png_pdf(
    *,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_stem: str,
    output_dir: Path,
    organ_definitions: Tuple[OrganDefinition, ...],
    taxonomy_name: str,
    pdf_dpi: int = INFERENCE_PANEL_PDF_DPI,
) -> tuple[Path, Path]:
    """
    Save ``{sample_stem}_panel.png`` and ``{sample_stem}_panel.pdf`` under output_dir.

    PDF is matplotlib-native at ``pdf_dpi`` (default 1000), not converted from PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)

    img = _ensure_rgb_u8(image_rgb)
    gt_u8 = np.asarray(gt_mask).astype(np.uint8)
    pr_u8 = np.asarray(pred_mask).astype(np.uint8)

    gt_ov = semantic_overlay(img, gt_u8, organ_definitions)
    pred_ov = semantic_overlay(img, pr_u8, organ_definitions)
    err_ov = difference_blend_rgb(img, gt_u8, pr_u8, organ_definitions)

    h, w = img.shape[:2]
    panel_w_in = 3.25
    panel_h_in = panel_w_in * h / max(w, 1)
    fig_w = 4 * panel_w_in
    fig_h = panel_h_in + 0.55
    pad = 0.08

    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=False)
    panels = [
        ("Ultrasound", img),
        ("Ground truth", gt_ov),
        ("Prediction", pred_ov),
        ("Error (TP/FP/FN)", err_ov),
    ]
    for ax, (title, data) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    legend_elements = [
        Patch(facecolor=(0.0, 0.9, 0.0), edgecolor="black", linewidth=0.5, label="TP"),
        Patch(facecolor=(0.9, 0.0, 0.0), edgecolor="black", linewidth=0.5, label="FP"),
        Patch(facecolor=(0.0, 0.0, 0.9), edgecolor="black", linewidth=0.5, label="FN"),
    ]
    axes[3].legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.92)

    du = distance_unit_short(taxonomy_name)
    fig.suptitle(
        f"Test split — {taxonomy_name} (boundary distances reported in {du})",
        fontsize=9,
        y=1.02,
    )

    stem_path = output_dir / f"{sample_stem}_panel"
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.06)
    png_path = Path(f"{stem_path}.png")
    pdf_path = Path(f"{stem_path}.pdf")
    _matplotlib_save_panel_pdf(fig, pdf_path, pad_inches=pad, dpi=pdf_dpi)
    _matplotlib_save_panel_png(fig, png_path, pad_inches=pad)
    plt.close(fig)
    return png_path, pdf_path
