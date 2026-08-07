"""
Export PFUS1 JSON polygons to an RTSTRUCT-style mask bundle (PFUS1 only).

Writes under ``--masks_root`` (default ``data/bladder/pfus1/masks/``):

- ``<JSON label>/`` — per-structure **binary** ``{Pxxx_frame_yyy}_mask.png`` (0 / 255).
- ``combined_masks/{Pxxx_frame_yyy}_combined_mask.png`` — **training** multiclass uint8
  (class ids 0–8; same semantics as before, but **flat** filenames like clinical MR stems).
- ``combined_masks/combined_mask_preview/`` — **QA only** greyscale PNG + optional one-page PDF
  per slice (mirrors ``bapmos.preprocess.prostate.rtstruct_export_slice_masks``; not used by training).
- Optional **subset PDF pass**: ``--preview_pdf_first_n_per_patient N`` writes PDFs only for the
  first ``N`` frames per patient (sorted ``frame_*.png``), rasterized at **350 dpi** (``PDF_EXPORT_DPI``),
  without re-exporting combined masks. Runs after the main export and respects ``--overwrite``.
  This subset pass is **independent** of ``--no_preview`` / ``--no_pdf`` on the main pass.
- ``combined_masks/label_mapping.json`` — taxonomy, paint order, preview LUT, per-folder counts.
- ``combined_masks/export_summary.json`` — written/skipped counts and warning totals.

Raw acquisitions live under ``data/bladder/pfus1_raw/`` (``Pxxx/frame_*.png`` + ``.json``);
non-PFUS1 pipelines are untouched.

Painting order for combined: ascending ``class_id`` (higher id wins overlaps).
Out-of-bounds polygon vertices rely on OpenCV ``fillPoly`` clipping behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from bapmos.paths import project_root
from bapmos.pdf_export import PDF_EXPORT_DPI
from bapmos.preprocess.bladder.constants import JSON_LABEL_TO_CLASS_ID, PFUS1_ALL_LABELS


def _root() -> Path:
    return project_root()


PREVIEW_SUBDIR_NAME = "combined_mask_preview"


def _poly_to_pts_int(poly: List[List[float]]) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected Nx2 polygon, got shape {arr.shape}")
    pts = np.round(arr).astype(np.int32)
    return pts.reshape(-1, 1, 2)


def collect_contours_by_class(
    annotations: List[Dict[str, Any]],
) -> Tuple[Dict[int, List[np.ndarray]], List[str]]:
    by_class: Dict[int, List[np.ndarray]] = {}
    warnings: List[str] = []
    for obj in annotations:
        label = obj.get("label")
        pol = obj.get("pol")
        if label not in JSON_LABEL_TO_CLASS_ID:
            warnings.append(f"Unknown label {label!r} skipped")
            continue
        cid = JSON_LABEL_TO_CLASS_ID[label]
        if not isinstance(pol, list) or len(pol) < 3:
            warnings.append(f"Label {label!r}: polygon too small, skipped")
            continue
        try:
            pts = _poly_to_pts_int(pol)
        except ValueError as e:
            warnings.append(f"Label {label!r}: {e}")
            continue
        by_class.setdefault(cid, []).append(pts)
    return by_class, warnings


def rasterize_frame(
    image_shape_hw: Tuple[int, int],
    annotations: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    h, w = image_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    by_class, warnings = collect_contours_by_class(annotations)
    for _, cid in sorted(PFUS1_ALL_LABELS, key=lambda x: x[1]):
        contours = by_class.get(cid)
        if not contours:
            continue
        cv2.fillPoly(mask, contours, int(cid))
    return mask, warnings


def flat_file_key(patient: str, frame_stem: str) -> str:
    """``P000`` + ``frame_000`` → ``P000_frame_000`` (unique across patients)."""
    return f"{patient}_{frame_stem}"


def preview_lut_pfus1() -> Dict[int, int]:
    """Grey levels for 0..8 (QA PNG only; training uses raw class ids)."""
    return {i: int(round(i * 255.0 / 8.0)) for i in range(9)}


def combined_multiclass_to_preview_u8(combined: np.ndarray) -> np.ndarray:
    lut = preview_lut_pfus1()
    out = np.zeros(combined.shape, dtype=np.uint8)
    for cid, grey in lut.items():
        out[combined == cid] = grey
    return out


def _write_preview_pdf_single_page(grey_u8: np.ndarray, pdf_path: Path, *, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = grey_u8.shape
    fig_w = max(w / dpi, 1.0)
    fig_h = max(h / dpi, 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(grey_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_axis_off()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(pdf_path), dpi=dpi, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _clean_legacy_nested_combined_masks(combined_dir: Path) -> int:
    """
    Remove legacy ``combined_masks/Pxxx/`` trees so flat stems stay unambiguous.
    Returns number of directories removed.
    """
    n = 0
    if not combined_dir.is_dir():
        return 0
    for child in list(combined_dir.iterdir()):
        if child.is_dir() and re.fullmatch(r"P\d+", child.name):
            shutil.rmtree(child, ignore_errors=False)
            n += 1
    return n


def export_frame_rtstruct_style(
    *,
    png_path: Path,
    json_path: Path,
    patient: str,
    masks_root: Path,
    write_per_organ: bool,
    write_preview_png: bool,
    write_preview_pdf: bool,
    combined_count: List[int],
    preview_png_count: List[int],
    preview_pdf_count: List[int],
) -> Tuple[bool, List[str]]:
    """Returns (wrote_combined, warnings)."""
    warnings: List[str] = []
    stem = png_path.stem
    file_key = flat_file_key(patient, stem)
    combined_dir = masks_root / "combined_masks"
    preview_dir = combined_dir / PREVIEW_SUBDIR_NAME
    combined_path = combined_dir / f"{file_key}_combined_mask.png"

    im = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        warnings.append(f"{png_path}: failed to read image")
        return False, warnings
    h, w = im.shape[:2]

    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)
    if not isinstance(ann, list):
        warnings.append(f"{json_path}: root JSON must be a list")
        return False, warnings

    combined, rw = rasterize_frame((h, w), ann)
    warnings.extend(rw)

    if write_per_organ:
        for label_name, cid in PFUS1_ALL_LABELS:
            bin_path = masks_root / label_name / f"{file_key}_mask.png"
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            bin_mask = ((combined == cid).astype(np.uint8) * 255)
            cv2.imwrite(str(bin_path), bin_mask)

    combined_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(combined_path), combined)
    combined_count[0] += 1

    if write_preview_png or write_preview_pdf:
        preview_dir.mkdir(parents=True, exist_ok=True)
        prev_u8 = combined_multiclass_to_preview_u8(combined)
        if write_preview_png:
            prev_png = preview_dir / f"{file_key}_combined_mask_preview.png"
            cv2.imwrite(str(prev_png), prev_u8)
            preview_png_count[0] += 1
        if write_preview_pdf:
            prev_pdf = preview_dir / f"{file_key}_combined_mask_preview.pdf"
            try:
                _write_preview_pdf_single_page(prev_u8, prev_pdf, dpi=PDF_EXPORT_DPI)
                preview_pdf_count[0] += 1
            except Exception as e:  # noqa: BLE001
                warnings.append(f"PDF skip {file_key}: {e}")

    return True, warnings


def write_label_mapping(masks_root: Path) -> None:
    """Write ``combined_masks/label_mapping.json`` using on-disk file counts."""
    combined_dir = masks_root / "combined_masks"
    folders: Dict[str, Any] = {}
    for label_name, cid in PFUS1_ALL_LABELS:
        d = masks_root / label_name
        n = len(list(d.glob("*_mask.png"))) if d.is_dir() else 0
        folders[label_name] = {
            "class_id": cid,
            "json_label": label_name,
            "png_written": n,
        }
    n_combined = (
        len([p for p in combined_dir.glob("*_combined_mask.png")]) if combined_dir.is_dir() else 0
    )
    lut = preview_lut_pfus1()
    doc = {
        "taxonomy": "pfus1",
        "paint_order_class_ids": [cid for _, cid in sorted(PFUS1_ALL_LABELS, key=lambda x: x[1])],
        "folders": folders,
        "n_combined_masks_on_disk": n_combined,
        "training_uses": "{file_key}_combined_mask.png in combined_masks/ (uint8 class_id per pixel, 0=bg, 1..8).",
        "qa_preview_dir": f"combined_masks/{PREVIEW_SUBDIR_NAME}/",
        "qa_preview_only": (
            f"Per-frame PNG + PDF under combined_masks/{PREVIEW_SUBDIR_NAME}/ "
            f"(PDF rasterized at {PDF_EXPORT_DPI} dpi) — not used by PFUS1Dataset."
        ),
        "preview_pdf_dpi": PDF_EXPORT_DPI,
        "preview_grey_by_class_id": {str(k): v for k, v in lut.items()},
        "file_key_pattern": "{patient}_{frame_stem}",
        "example_file_key": "P000_frame_000",
    }
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "label_mapping.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )


def write_first_n_preview_pdfs_per_patient(
    raw_root: Path,
    masks_root: Path,
    first_n: int,
    overwrite: bool,
) -> int:
    """
    Write ``*_combined_mask_preview.pdf`` (dpi = ``PDF_EXPORT_DPI``, 350) for the first
    ``first_n`` frames per patient only. Reads existing multiclass combined masks.
    """
    combined_dir = masks_root / "combined_masks"
    preview_dir = combined_dir / PREVIEW_SUBDIR_NAME
    preview_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(PDF_EXPORT_DPI)
    written = 0
    patients = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    for patient_dir in patients:
        patient = patient_dir.name
        png_paths = sorted(patient_dir.glob("frame_*.png"))
        for idx, png_path in enumerate(png_paths):
            if idx >= first_n:
                break
            stem = png_path.stem
            file_key = flat_file_key(patient, stem)
            pdf_path = preview_dir / f"{file_key}_combined_mask_preview.pdf"
            if pdf_path.exists() and not overwrite:
                continue
            combined_path = combined_dir / f"{file_key}_combined_mask.png"
            if not combined_path.is_file():
                continue
            combined = cv2.imread(str(combined_path), cv2.IMREAD_GRAYSCALE)
            if combined is None:
                continue
            prev_u8 = combined_multiclass_to_preview_u8(combined)
            try:
                _write_preview_pdf_single_page(prev_u8, pdf_path, dpi=dpi)
                written += 1
            except Exception as e:  # noqa: BLE001
                print(f"WARN PDF {file_key}: {e}", file=sys.stderr)
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=_root() / "data/bladder/pfus1_raw",
        help="Directory containing patient folders P000, P001, ... (each with frame_*.png + .json).",
    )
    parser.add_argument(
        "--masks_root",
        type=Path,
        default=_root() / "data/bladder/pfus1/masks",
        help="RTSTRUCT-style bundle root (per-label dirs + combined_masks/).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_per_organ",
        action="store_true",
        help="Skip binary masks under each label folder.",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip main-pass combined_mask_preview/ PNG and PDF (subset PDF pass still allowed).",
    )
    parser.add_argument(
        "--no_pdf",
        action="store_true",
        help="With main-pass previews on, write PNG only (skip PDF). Does not block "
        "--preview_pdf_first_n_per_patient.",
    )
    parser.add_argument(
        "--clean_legacy_nested",
        action="store_true",
        help="Remove legacy combined_masks/Pxxx/ subfolders before export.",
    )
    parser.add_argument(
        "--preview_pdf_first_n_per_patient",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After the main pass, write combined_mask_preview PDFs (dpi=350) only for the "
            "first N frames per patient (sorted frame_*.png). Independent of --no_preview / "
            "--no_pdf on the main pass; does not re-export combined masks."
        ),
    )
    args = parser.parse_args(argv)

    raw_root = args.raw_root
    if not raw_root.is_dir():
        print(f"ERROR: raw_root not found: {raw_root}", file=sys.stderr)
        return 1

    masks_root = args.masks_root.resolve()
    combined_dir = masks_root / "combined_masks"
    if args.clean_legacy_nested:
        nrm = _clean_legacy_nested_combined_masks(combined_dir)
        if nrm:
            print(f"Removed {nrm} legacy nested patient folder(s) under {combined_dir}")

    patients = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    if not patients:
        print(f"ERROR: no P* patient folders under {raw_root}", file=sys.stderr)
        return 1

    combined_count = [0]
    preview_png_count = [0]
    preview_pdf_count = [0]
    total_skipped = 0
    all_warn: List[str] = []

    write_per_organ = not args.no_per_organ
    write_preview = not args.no_preview
    write_pdf = write_preview and not args.no_pdf
    write_prev_png = write_preview

    for pdir in patients:
        patient = pdir.name
        for png_path in sorted(pdir.glob("frame_*.png")):
            stem = png_path.stem
            json_path = pdir / f"{stem}.json"
            if not json_path.is_file():
                all_warn.append(f"{patient}/{stem}: missing JSON")
                total_skipped += 1
                continue

            file_key = flat_file_key(patient, stem)
            combined_path = combined_dir / f"{file_key}_combined_mask.png"
            if combined_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            did, warns = export_frame_rtstruct_style(
                png_path=png_path,
                json_path=json_path,
                patient=patient,
                masks_root=masks_root,
                write_per_organ=write_per_organ,
                write_preview_png=write_prev_png,
                write_preview_pdf=write_pdf,
                combined_count=combined_count,
                preview_png_count=preview_png_count,
                preview_pdf_count=preview_pdf_count,
            )
            for wn in warns:
                all_warn.append(f"{patient}/{stem}: {wn}")
            if not did:
                total_skipped += 1

    write_label_mapping(masks_root)

    subset_pdf = 0
    if args.preview_pdf_first_n_per_patient is not None:
        if args.preview_pdf_first_n_per_patient <= 0:
            print(
                "ERROR: --preview_pdf_first_n_per_patient must be a positive integer",
                file=sys.stderr,
            )
            return 1
        subset_pdf = write_first_n_preview_pdfs_per_patient(
            raw_root, masks_root, args.preview_pdf_first_n_per_patient, args.overwrite
        )
        print(
            f"Preview PDF subset: first {args.preview_pdf_first_n_per_patient} frames/patient "
            f"at dpi={PDF_EXPORT_DPI} → wrote {subset_pdf} PDF(s) under "
            f"{(masks_root / 'combined_masks' / PREVIEW_SUBDIR_NAME).resolve()}"
        )

    for wn in all_warn[:40]:
        print(f"WARN {wn}")
    if len(all_warn) > 40:
        print(f"WARN ... {len(all_warn) - 40} more warnings")

    summary = {
        "raw_root": str(Path(raw_root).resolve()),
        "masks_root": str(masks_root),
        "combined_written": combined_count[0],
        "skipped": total_skipped,
        "preview_png": preview_png_count[0],
        "preview_pdf": preview_pdf_count[0],
        "subset_preview_pdf": subset_pdf,
        "n_warnings": len(all_warn),
        "warnings_sample": all_warn[:40],
    }
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"Done. masks_root={masks_root} "
        f"combined_written={combined_count[0]} skipped={total_skipped} "
        f"preview_png={preview_png_count[0]} preview_pdf={preview_pdf_count[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
