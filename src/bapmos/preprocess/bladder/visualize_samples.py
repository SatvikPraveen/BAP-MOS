"""
Save overlay visualizations (image + colored multiclass mask contours / fill).

Reads raw PNGs and masks produced by ``convert_json_polygons_to_masks``.

Optional **350 dpi PDF** (``bapmos.pdf_export.PDF_EXPORT_DPI``) next to each PNG via
``--write_pdf``, or PDF-only export from existing ``sample_*.png`` files via
``--export_pdf_only``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from bapmos.paths import project_root
from bapmos.pdf_export import PDF_EXPORT_DPI
from bapmos.preprocess.bladder.constants import PFUS1_ALL_LABELS
from bapmos.preprocess.bladder.dataset import parse_sample_line


def _root() -> Path:
    return project_root()


def _palette_bgr() -> Dict[int, Tuple[int, int, int]]:
    # Distinct BGR colors for 8 classes (OpenCV).
    return {
        1: (180, 120, 80),
        2: (0, 255, 255),
        3: (0, 165, 255),
        4: (147, 20, 255),
        5: (255, 0, 180),
        6: (60, 76, 40),
        7: (255, 0, 0),
        8: (0, 255, 0),
    }


def overlay_mask_bgr(image_gray: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    pal = _palette_bgr()
    color_layer = np.zeros_like(base)
    for cid, col in pal.items():
        m = mask == cid
        if not m.any():
            continue
        color_layer[m] = np.array(col, dtype=np.float32)
    fused = (1.0 - alpha) * base + alpha * color_layer
    return np.clip(fused, 0, 255).astype(np.uint8)


def append_legend_column(
    vis_bgr: np.ndarray,
    title: str | None = None,
    margin: int = 10,
    swatch: int = 16,
    line_gap: int = 4,
) -> np.ndarray:
    """
    Append a white column on the right with color swatches and structure names
    (matches ``_palette_bgr`` / ``PFUS1_ALL_LABELS``).
    """
    h, w = vis_bgr.shape[:2]
    pal = _palette_bgr()
    entries: List[Tuple[int, str, Tuple[int, int, int]]] = [
        (cid, name, pal[cid]) for name, cid in PFUS1_ALL_LABELS
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(np.clip(0.38 * (h / 520.0), 0.32, 0.62))
    thickness = max(1, min(2, int(round(font_scale * 2.2))))

    max_text_w = 0
    line_heights: List[int] = []
    for _, name, _ in entries:
        (tw, th), baseline = cv2.getTextSize(name, font, font_scale, thickness)
        max_text_w = max(max_text_w, tw)
        line_heights.append(th + baseline + line_gap)

    title_h = 0
    title_tw = 0
    if title:
        (title_tw, title_th), tbase = cv2.getTextSize(title, font, font_scale * 0.85, thickness)
        title_h = title_th + tbase + margin + line_gap

    row_h = max(swatch + 2, max(line_heights) if line_heights else swatch)
    legend_body_h = title_h + len(entries) * row_h + margin
    canvas_h = max(h, legend_body_h + margin)
    legend_w = margin + swatch + 8 + max_text_w + margin
    canvas_w = w + legend_w

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = vis_bgr
    if canvas_h > h:
        canvas[h:, :w].fill(255)

    x0 = w + margin
    y = margin + title_h
    if title:
        y_title = margin + title_th
        cv2.putText(
            canvas,
            title,
            (x0, y_title),
            font,
            font_scale * 0.85,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )

    for cid, name, col in entries:
        y1 = y - swatch + 2
        y2 = y + 2
        cv2.rectangle(canvas, (x0, y1), (x0 + swatch, y2), col, thickness=-1)
        cv2.rectangle(canvas, (x0, y1), (x0 + swatch, y2), (50, 50, 50), thickness=1)
        cv2.putText(
            canvas,
            name,
            (x0 + swatch + 8, y),
            font,
            font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )
        y += row_h

    return canvas


def bgr_image_to_pdf(bgr: np.ndarray, pdf_path: Path, *, dpi: int) -> None:
    """One-page PDF embedding the BGR raster (e.g. overlay + legend) at ``dpi``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fig_w = max(w / dpi, 1.0)
    fig_h = max(h / dpi, 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(pdf_path), dpi=dpi, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def export_pdfs_for_existing_sample_pngs(directory: Path, *, dpi: int, overwrite: bool) -> int:
    """Write ``sample_*.pdf`` next to each ``sample_*.png`` under ``directory``."""
    if not directory.is_dir():
        return -1
    n = 0
    for png_path in sorted(directory.glob("sample_*.png")):
        pdf_path = png_path.with_suffix(".pdf")
        if pdf_path.exists() and not overwrite:
            continue
        bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"WARN skip unreadable {png_path}", file=sys.stderr)
            continue
        try:
            bgr_image_to_pdf(bgr, pdf_path, dpi=dpi)
            n += 1
            print(pdf_path)
        except Exception as e:  # noqa: BLE001
            print(f"WARN PDF failed {png_path}: {e}", file=sys.stderr)
    return n


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_file",
        type=Path,
        default=None,
        help="train.txt or test.txt listing Pxxx/frame_yyy (not needed with --export_pdf_only).",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=_root() / "data/bladder/pfus1_raw",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=_root() / "data/bladder/pfus1/masks/combined_masks",
    )
    parser.add_argument("--n", type=int, default=8, help="Max samples to render")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_root() / "data/bladder/pfus1/reports/viz_samples",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Save overlay only (no on-image class labels / swatches).",
    )
    parser.add_argument(
        "--write_pdf",
        action="store_true",
        help=f"Also save a one-page PDF (dpi={PDF_EXPORT_DPI}) next to each generated PNG.",
    )
    parser.add_argument(
        "--export_pdf_only",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Only write PDFs for existing sample_*.png under DIR (dpi="
            f"{PDF_EXPORT_DPI}); skips PNG generation. Ignores --split_file."
        ),
    )
    parser.add_argument(
        "--overwrite_pdf",
        action="store_true",
        help="With --export_pdf_only, replace existing PDFs.",
    )
    args = parser.parse_args(argv)

    dpi = int(PDF_EXPORT_DPI)

    if args.export_pdf_only is not None:
        d = args.export_pdf_only
        if not d.is_absolute():
            d = _root() / d
        d = d.resolve()
        n = export_pdfs_for_existing_sample_pngs(d, dpi=dpi, overwrite=args.overwrite_pdf)
        if n < 0:
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
        print(f"Done. Wrote {n} PDF(s) at dpi={dpi} under {d}")
        return 0

    split_file = args.split_file
    if split_file is None:
        print("ERROR: provide --split_file for PNG generation, or use --export_pdf_only DIR", file=sys.stderr)
        return 1
    if not split_file.is_absolute():
        split_file = _root() / split_file
    split_file = split_file.resolve()
    if not split_file.is_file():
        print(f"ERROR: split_file not found: {split_file}", file=sys.stderr)
        return 1

    image_root = args.image_root if args.image_root.is_absolute() else _root() / args.image_root
    mask_root = args.mask_root if args.mask_root.is_absolute() else _root() / args.mask_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else _root() / args.out_dir
    image_root = image_root.resolve()
    mask_root = mask_root.resolve()
    out_dir = out_dir.resolve()

    lines: List[str] = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        print("ERROR: no samples in split file", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lines = [f"{cid}: {name}" for name, cid in PFUS1_ALL_LABELS]
    (out_dir / "legend_classes.txt").write_text(
        "\n".join(legend_lines) + "\n", encoding="utf-8"
    )

    n = min(args.n, len(lines))
    for i, key in enumerate(lines[:n]):
        patient, stem = parse_sample_line(key)
        img_path = image_root / patient / f"{stem}.png"
        mask_path = mask_root / f"{patient}_{stem}_combined_mask.png"
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None:
            print(f"WARN skip {key}: missing image or mask", file=sys.stderr)
            continue
        if im.shape[:2] != mask.shape[:2]:
            print(f"WARN skip {key}: shape mismatch", file=sys.stderr)
            continue
        vis = overlay_mask_bgr(im, mask, args.alpha)
        if not args.no_legend:
            title = f"{patient} {stem}"
            vis = append_legend_column(vis, title=title)
        out_path = out_dir / f"sample_{i:02d}_{patient}_{stem}.png"
        cv2.imwrite(str(out_path), vis)
        print(out_path)
        if args.write_pdf:
            pdf_path = out_path.with_suffix(".pdf")
            try:
                bgr_image_to_pdf(vis, pdf_path, dpi=dpi)
                print(pdf_path)
            except Exception as e:  # noqa: BLE001
                print(f"WARN PDF {out_path}: {e}", file=sys.stderr)

    print(f"Done. legend -> {out_dir / 'legend_classes.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
