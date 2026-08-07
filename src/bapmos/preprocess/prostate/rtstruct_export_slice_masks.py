"""
Export RTSTRUCT contours to per-slice PNG masks for BAP-MOS preprocessing.

Writes the same layout training expects under ``<dataset_root>/masks/``:

- One subfolder per exported ROI (binary ``{MR_stem}_mask.png``, value 255 foreground)
- ``combined_masks/{MR_stem}_combined_mask.png`` — **training** multiclass labels (raw class
  ids 0–4 clinical / 0–3 simulation); looks nearly black in a normal viewer.
- ``combined_masks/combined_mask_preview/{MR_stem}_combined_mask_preview.png`` and
  matching ``{MR_stem}_combined_mask_preview.pdf`` (one PDF per slice, ``PDF_EXPORT_DPI``) —
  **QA only**. Training still uses ``*_combined_mask.png`` only
  (see ``bapmos.paths.find_combined_masks_dir``).

When ``preprocessing/slice_organ_presence/<case>.json`` exists (from
``bapmos.preprocess.delineation.report_slice_organ_presence``), PNG-guided export
intersects stems with slices marked ``multiclass_status: read`` and at least one organ
(excludes ``Background`` and non-read rows) for smoother alignment with analysis.

Usage (from BAPMOS checkout root)::

    python -m bapmos.preprocess.prostate.rtstruct_export_slice_masks \\
        --dicom-dir \"data/real_data/Case 1/Dicom\" \\
        --masks-root preprocessing/real_data/case1/masks \\
        --taxonomy clinical \\
        --training-png-dir preprocessing/real_data/case1/case1_dicom_png

Canonical batch path (DICOM→PNG then masks)::

    python -m bapmos.preprocess.prostate.run_rtstruct_masks --case all

**Mask discovery fallback:** some code paths use ``find_combined_masks_dir_with_repo_fallback``
(with ``allow_test_output_fallback=True``) for PNG-guided export alignment. That is **QA /
export compatibility only**. Training loaders use strict ``find_combined_masks_dir`` under
the canonical dataset root (see ``docs/PREPROCESS.md`` and ``data/prostate/README.md``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from bapmos.data.organ_registry import (
    REAL_CLINICAL_ORGANS,
    SIMULATION_THREE_ORGANS,
)
from bapmos.paths import find_combined_masks_dir_with_repo_fallback, project_root
from bapmos.pdf_export import PDF_EXPORT_DPI
from bapmos.preprocess.prostate.convert_dicom_to_png import (
    dicom_slice_is_blank_for_training,
    should_skip_path,
)

REPO_ROOT = project_root()

logger = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False

MANIFEST_NAME = "rtstruct_roi_class_map.json"


def _ensure_rtstruct_export_logging() -> None:
    """Configure file+stderr handlers once (not at import time)."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True
    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_dir / "rtstruct_export_masks.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(sh)
    logger.propagate = False



def _norm_roi(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def rtstruct_roi_to_class_id(roi_name: str, taxonomy: str) -> Optional[int]:
    """
    Map RTSTRUCT ``ROIName`` string to training class id, or None to skip ROI.
    """
    n = _norm_roi(roi_name).replace(" ", "_")

    if not n:
        return None
    if "avoid" in n or n.startswith("z_") or "shell" in n and "avoid" in n:
        return None
    if "ring" in n and "mm" in n:
        return None
    if "couch" in n or "external" == n or n == "body":
        return None

    if taxonomy == "simulation":
        if "rectum" in n:
            return 1
        if "bladder" in n:
            return 2
        if "ptv" in n or "prostate" in n or "gtv" in n or "ctv" in n:
            return 3
        return None

    if taxonomy == "clinical":
        if "urethra" in n:
            return 4
        if "rectum" in n:
            return 3
        if "bladder" in n:
            return 1
        if "ptv" in n or "prostate" in n or "gtv" in n or ("ctv" in n and "prostate" in n):
            return 2
        return None

    raise ValueError(f"Unknown taxonomy {taxonomy!r}; use 'clinical' or 'simulation'")


def taxonomy_class_order(taxonomy: str) -> List[int]:
    """Paint combined mask in this class-id order (later IDs overwrite earlier on overlap)."""
    if taxonomy == "clinical":
        return [o.class_id for o in REAL_CLINICAL_ORGANS]
    if taxonomy == "simulation":
        return [o.class_id for o in SIMULATION_THREE_ORGANS]
    raise ValueError(taxonomy)


# Preview PNG only (training uses raw class ids in *_combined_mask.png).
_PREVIEW_LUT_CLINICAL: Dict[int, int] = {0: 0, 1: 70, 2: 140, 3: 210, 4: 255}
_PREVIEW_LUT_SIMULATION: Dict[int, int] = {0: 0, 1: 85, 2: 170, 3: 255}


def combined_multiclass_to_preview_u8(combined: np.ndarray, taxonomy: str) -> np.ndarray:
    """Map multiclass label image to greyscale for human viewing (not for training)."""
    lut = _PREVIEW_LUT_CLINICAL if taxonomy == "clinical" else _PREVIEW_LUT_SIMULATION
    out = np.zeros(combined.shape, dtype=np.uint8)
    for cid, grey in lut.items():
        out[combined == cid] = grey
    return out


PREVIEW_SUBDIR_NAME = "combined_mask_preview"


def _slice_sort_key(info: Dict[str, Any]) -> float:
    """Sort slices for PDF page order (best-effort z / slice index)."""
    sl = info.get("slice_location")
    if sl is not None:
        return float(sl)
    ipp = info.get("image_position") or []
    if len(ipp) >= 3:
        return float(ipp[2])
    return 0.0


def _write_preview_pdf_single_page(grey_u8: np.ndarray, pdf_path: Path, *, dpi: int) -> None:
    """One-page PDF of a greyscale preview rasterized at ``dpi`` (use ``PDF_EXPORT_DPI``)."""
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


def _masks_root_skips_parent_training_png_discovery(masks_root: Path) -> bool:
    """
    True when ``masks_root`` is not under a canonical dataset ``…/masks`` tree
    (e.g. ``test_output/…/masks`` or ``preprocessing/<export_bundle>/<case>/masks``), so
    ``find_training_images_dir(masks_root.parent)`` must not run on ``parent``.
    """
    parts_lower = {str(x).lower() for x in masks_root.parts}
    if "test_output" in parts_lower:
        return True
    try:
        rel = Path(masks_root).resolve().relative_to((REPO_ROOT / "preprocessing").resolve())
    except ValueError:
        return False
    parts = rel.parts
    if len(parts) < 3 or parts[-1].lower() != "masks":
        return False
    if parts[0] in ("real_data", "simulation_data"):
        return False
    return True


def resolve_training_png_directory(masks_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    """
    Directory of slice PNGs used to decide which MR stems are kept.

    If ``explicit`` is set and contains ``*.png``, use it. Otherwise try
    ``find_training_images_dir(masks_root.parent)`` when ``masks_root`` is **not** under
    ``test_output/`` or ``preprocessing/<export_bundle>/…/masks`` (there, ``parent`` is not the
    real dataset root and auto-discovery is skipped).
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if p.is_dir() and any(p.glob("*.png")):
            return p
        return None
    if _masks_root_skips_parent_training_png_discovery(masks_root):
        return None
    from bapmos.paths import find_training_images_dir

    try:
        cand = find_training_images_dir(masks_root.parent)
        if cand.is_dir() and any(cand.glob("*.png")):
            return cand.resolve()
    except FileNotFoundError:
        pass
    return None


def combined_mask_has_any_organ_pixel(mask_u8: np.ndarray, taxonomy: str) -> bool:
    """
    True if the multiclass combined mask contains any mapped organ label (same class IDs as
    ``bapmos.preprocess.prostate.create_stratified_splits`` organ presence /
    ``--exclude-background``).
    """
    for cid in taxonomy_class_order(taxonomy):
        if np.any(mask_u8 == cid):
            return True
    return False


def infer_slice_organ_presence_json(training_png_dir: Path) -> Optional[Path]:
    """
    Map a canonical preprocessing ``*_dicom_png`` folder to
    ``preprocessing/slice_organ_presence/{case1|case2|simulation}.json``.

    Returns None for ad-hoc paths (e.g. pytest temp dirs) so tests are not coupled to repo JSON.
    """
    try:
        rel = Path(training_png_dir).resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    parts_lower = [p.lower() for p in rel.parts]
    if "simulation_data" in parts_lower and "simulation_dicom_png" in parts_lower:
        key = "simulation"
    elif "real_data" in parts_lower and "case1" in parts_lower and "case1_dicom_png" in parts_lower:
        key = "case1"
    elif "real_data" in parts_lower and "case2" in parts_lower and "case2_dicom_png" in parts_lower:
        key = "case2"
    else:
        return None
    p = REPO_ROOT / "preprocessing" / "slice_organ_presence" / f"{key}.json"
    return p if p.is_file() else None


def stems_with_at_least_one_organ_from_slice_report(report_json: Path) -> Optional[Set[str]]:
    """
    Slice stems where the organ report lists ``multiclass_status: read``, non-``Background``
    pattern, and non-empty ``organs_present``.

    Returns None if the file is missing or invalid; returns an empty set only when the file
    parses but yields no organ rows (caller may treat as skip to avoid wiping exports).
    """
    try:
        data = json.loads(report_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = data.get("slices")
    if not isinstance(rows, list):
        return None
    out: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("multiclass_status") != "read":
            continue
        if row.get("pattern") == "Background":
            continue
        organs = row.get("organs_present") or []
        if not organs:
            continue
        fn = row.get("slice") or ""
        if not isinstance(fn, str) or not fn.lower().endswith(".png"):
            continue
        out.add(Path(fn).stem)
    return out


def filter_stems_by_preprocessing_combined_organs(
    stems: Set[str],
    training_png_dir: Path,
    taxonomy: str,
) -> Tuple[Set[str], int, int]:
    """
    When preprocessing exposes ``masks/combined_masks`` (see ``find_combined_masks_dir``),
    keep only stems where ``<stem>_combined_mask.png`` **exists** and has at least one organ
    class pixel (same ids as ``create_stratified_splits.organ_presence``).

    Stems with no combined file are dropped (stratify treats these as ``missing_masks`` and
    excludes them). Stems whose combined mask is all background / empty are dropped
    (``exclude_background_only``).

    If no combined-mask directory is found yet (first-time export), returns ``stems`` unchanged.
    Returns ``(kept_stems, n_dropped_missing_combined, n_dropped_background_only)``.
    """
    try:
        mask_dir = find_combined_masks_dir_with_repo_fallback(
            Path(training_png_dir).resolve().parent,
            allow_test_output_fallback=True,
        )
    except FileNotFoundError:
        return stems, 0, 0

    keep: Set[str] = set()
    n_missing = 0
    n_bg = 0
    for stem in stems:
        mp = mask_dir / f"{stem}_combined_mask.png"
        if not mp.is_file():
            n_missing += 1
            continue
        arr = np.asarray(Image.open(mp))
        if arr.size == 0 or not combined_mask_has_any_organ_pixel(arr, taxonomy):
            n_bg += 1
            continue
        keep.add(stem)
    dropped = len(stems) - len(keep)
    if dropped:
        logger.info(
            "Organ slice filter (%s): kept %d / %d PNG stems "
            "(dropped %d without combined mask, %d background-only in existing combined)",
            mask_dir,
            len(keep),
            len(stems),
            n_missing,
            n_bg,
        )
    return keep, n_missing, n_bg


def drop_background_only_stems_using_preprocessing_combined(
    stems: Set[str],
    training_png_dir: Path,
    taxonomy: str,
) -> Tuple[Set[str], int]:
    """
    Backwards-compatible name: same as :func:`filter_stems_by_preprocessing_combined_organs`
    but returns ``(kept_stems, total_dropped)``.
    """
    kept, n_miss, n_bg = filter_stems_by_preprocessing_combined_organs(stems, training_png_dir, taxonomy)
    return kept, n_miss + n_bg


def filter_image_map_to_training_slices(
    image_map: Dict[str, Dict[str, Any]],
    *,
    training_png_dir: Optional[Path],
    dicom_dir: Path,
    taxonomy: str = "clinical",
    slice_organ_presence_json: Optional[Path] = None,
    use_slice_organ_presence_json: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Keep only MR slices that have a matching preprocessing PNG (if ``training_png_dir`` has
    any ``*.png``), else drop slices that fail ``dicom_slice_is_blank_for_training``.

    When PNG-guided, stems pass combined-mask checks (including ``test_output`` / ``preprocessing`` bundle fallbacks), then
    optionally intersect with ``preprocessing/slice_organ_presence/<case>.json`` so only slices
    with ≥1 organ in that report are kept.
    """
    allowed_stems: Optional[Set[str]] = None
    if training_png_dir is not None:
        p = Path(training_png_dir).resolve()
        if p.is_dir():
            stems = {x.stem for x in p.glob("*.png")}
            if len(stems) > 0:
                allowed_stems, _, _ = filter_stems_by_preprocessing_combined_organs(stems, p, taxonomy)
                report_path: Optional[Path] = None
                if slice_organ_presence_json is not None:
                    rp = Path(slice_organ_presence_json).expanduser().resolve()
                    report_path = rp if rp.is_file() else None
                elif use_slice_organ_presence_json:
                    report_path = infer_slice_organ_presence_json(p)
                if report_path is not None:
                    organ_stems = stems_with_at_least_one_organ_from_slice_report(report_path)
                    if organ_stems is not None and len(organ_stems) > 0:
                        n0 = len(allowed_stems)
                        allowed_stems = allowed_stems & organ_stems
                        n_drop = n0 - len(allowed_stems)
                        if n_drop:
                            logger.info(
                                "Slice organ presence %s: intersected to %d stems "
                                "(dropped %d not listed with ≥1 organ)",
                                report_path.name,
                                len(allowed_stems),
                                n_drop,
                            )
                    elif organ_stems is not None and len(organ_stems) == 0:
                        logger.warning(
                            "Slice organ report %s has no organ-bearing rows; skipping JSON intersection",
                            report_path,
                        )

    out: Dict[str, Dict[str, Any]] = {}
    for uid, info in image_map.items():
        stem = str(info.get("file_name", ""))
        fp = info.get("file_path")
        if not isinstance(fp, Path):
            fp = dicom_dir / f"{stem}.dcm"
        fp = Path(fp)

        if allowed_stems is not None:
            if stem not in allowed_stems:
                continue
        else:
            if fp.is_file() and dicom_slice_is_blank_for_training(fp):
                continue
        out[uid] = info

    logger.info(
        "Training slice filter: kept %d / %d MR slices (png_ref=%s)",
        len(out),
        len(image_map),
        str(training_png_dir) if training_png_dir else "None; per-DICOM blank gate",
    )
    if not out:
        raise RuntimeError(
            "After training-slice filter, no MR slices remain. "
            "Provide --training-png-dir with preprocessing PNGs, or ensure non-blank DICOM slices. "
            "If preprocessing combined masks exist, background-only slices (no organ labels) are removed "
            "to match stratified_splits. "
            "This is not caused by --output-bundle (that only changes the write folder under preprocessing/ or test_output/). "
            "Typical fix: run `python -m bapmos.preprocess.prostate.run_rtstruct_masks "
            "--case <case>` first so `case*_dicom_png/` exists, then re-run export."
        )
    return out


def _prune_mask_artifacts_outside_stems(masks_root: Path, allowed_stems: Set[str]) -> None:
    """Remove mask PNGs/PDFs for MR stems not in ``allowed_stems`` (stale after tighter filter)."""
    if not masks_root.is_dir():
        return
    combined = masks_root / "combined_masks"
    preview = combined / PREVIEW_SUBDIR_NAME
    for organ_dir in masks_root.iterdir():
        if not organ_dir.is_dir() or organ_dir.name == "combined_masks":
            continue
        for f in organ_dir.glob("*_mask.png"):
            stem = f.name.removesuffix("_mask.png")
            if stem not in allowed_stems:
                try:
                    f.unlink()
                except OSError:
                    pass
    if combined.is_dir():
        for f in combined.glob("*_combined_mask.png"):
            stem = f.name.removesuffix("_combined_mask.png")
            if stem not in allowed_stems:
                for path in (
                    f,
                    preview / f"{stem}_combined_mask_preview.png",
                    preview / f"{stem}_combined_mask_preview.pdf",
                ):
                    try:
                        path.unlink()
                    except OSError:
                        pass


def build_image_slice_mapping(dicom_dir: Path) -> Dict[str, Dict[str, Any]]:
    """SOPInstanceUID → slice metadata (same idea as legacy script1)."""
    logger.info("Building MR slice mapping from %s", dicom_dir)
    image_map: Dict[str, Dict[str, Any]] = {}
    dcm_files = sorted(p for p in dicom_dir.glob("*.dcm") if not should_skip_path(p))
    for dcm_file in tqdm(dcm_files, desc="DICOM headers"):
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
        except Exception as e:
            logger.warning("Skip %s: %s", dcm_file.name, e)
            continue
        if not hasattr(ds, "SOPInstanceUID"):
            continue
        uid = str(ds.SOPInstanceUID)
        ipp = getattr(ds, "ImagePositionPatient", None)
        ps = getattr(ds, "PixelSpacing", None)
        rows = int(ds.Rows) if hasattr(ds, "Rows") else None
        cols = int(ds.Columns) if hasattr(ds, "Columns") else None
        if ipp is None or ps is None or rows is None or cols is None:
            logger.warning("Incomplete geometry for %s", dcm_file.name)
            continue
        image_map[uid] = {
            "file_path": dcm_file,
            "file_name": dcm_file.stem,
            "image_position": [float(x) for x in ipp],
            "slice_location": float(ds.SliceLocation) if hasattr(ds, "SliceLocation") else None,
            "rows": rows,
            "columns": cols,
            "pixel_spacing": [float(x) for x in ps],
        }
    logger.info("Mapped %d MR slices", len(image_map))
    return image_map


def contour_to_mask(
    contour_data: np.ndarray,
    image_position: np.ndarray,
    pixel_spacing: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Project RTSTRUCT contour (mm) to slice binary mask (uint8 0 / 255)."""
    contour_pixels: List[List[int]] = []
    for point in contour_data:
        col = (point[0] - image_position[0]) / pixel_spacing[1]
        row = (point[1] - image_position[1]) / pixel_spacing[0]
        contour_pixels.append([int(round(col)), int(round(row))])

    mask = np.zeros((rows, cols), dtype=np.uint8)
    if contour_pixels:
        arr = np.array(contour_pixels, dtype=np.int32)
        cv2.fillPoly(mask, [arr], 255)
    return mask


def _safe_folder_name(roi_name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", roi_name.strip()).strip("_") or "ROI"


def extract_per_organ_masks(
    rtstruct_file: Path,
    image_map: Dict[str, Dict[str, Any]],
    masks_root: Path,
    taxonomy: str,
    structures_filter: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Write per-ROI binary masks under ``masks_root/<ROI_folder>/``.
    Returns manifest: folder_name → {class_id, roi_name}.
    """
    masks_root.mkdir(parents=True, exist_ok=True)
    ds = pydicom.dcmread(str(rtstruct_file))

    roi_names: Dict[int, str] = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            roi_names[int(roi.ROINumber)] = str(roi.ROIName)

    manifest: Dict[str, Dict[str, Any]] = {}

    if not hasattr(ds, "ROIContourSequence"):
        raise RuntimeError("ROIContourSequence missing in RTSTRUCT")

    for roi_contour in tqdm(ds.ROIContourSequence, desc="RTSTRUCT ROIs"):
        roi_num = int(roi_contour.ReferencedROINumber)
        roi_name = roi_names.get(roi_num, f"ROI_{roi_num}")
        if structures_filter is not None and roi_name not in structures_filter:
            continue

        cid = rtstruct_roi_to_class_id(roi_name, taxonomy)
        if cid is None:
            logger.debug("Skip ROI (unmapped): %s", roi_name)
            continue

        folder = _safe_folder_name(roi_name)
        structure_dir = masks_root / folder
        structure_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(roi_contour, "ContourSequence"):
            continue

        n_written = 0
        for contour in roi_contour.ContourSequence:
            if not hasattr(contour, "ContourData") or not hasattr(contour, "ContourImageSequence"):
                continue
            ref_uid = str(contour.ContourImageSequence[0].ReferencedSOPInstanceUID)
            if ref_uid not in image_map:
                continue
            info = image_map[ref_uid]
            pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            mask = contour_to_mask(
                pts,
                np.array(info["image_position"], dtype=np.float64),
                np.array(info["pixel_spacing"], dtype=np.float64),
                int(info["rows"]),
                int(info["columns"]),
            )
            out_path = structure_dir / f"{info['file_name']}_mask.png"
            cv2.imwrite(str(out_path), mask)
            n_written += 1

        if n_written:
            manifest[folder] = {"class_id": cid, "roi_name": roi_name, "png_written": n_written}
            logger.info("ROI %s → class %s (%d PNGs) → %s", roi_name, cid, n_written, structure_dir)

    map_path = masks_root / MANIFEST_NAME
    map_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", map_path)
    return manifest


def build_combined_masks(
    masks_root: Path,
    image_map: Dict[str, Dict[str, Any]],
    taxonomy: str,
) -> Path:
    """Fuse per-ROI binaries into ``masks_root/combined_masks`` using manifest class IDs."""
    manifest_path = masks_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run extract step first")

    manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined_dir = masks_root / "combined_masks"
    combined_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = combined_dir / PREVIEW_SUBDIR_NAME
    preview_dir.mkdir(parents=True, exist_ok=True)

    for stale in preview_dir.glob("*_combined_mask_previews.pdf"):
        try:
            stale.unlink()
        except OSError:
            pass

    # Remove legacy preview PNGs left next to training masks (older layout).
    for stale in combined_dir.glob("*_combined_mask_preview.png"):
        try:
            stale.unlink()
        except OSError:
            pass

    # class_id -> list of folder names (normally one folder per class)
    by_class: Dict[int, List[str]] = {}
    for folder, meta in manifest.items():
        cid = int(meta["class_id"])
        by_class.setdefault(cid, []).append(folder)

    paint_order = taxonomy_class_order(taxonomy)
    lut_preview = (
        {str(k): v for k, v in _PREVIEW_LUT_CLINICAL.items()}
        if taxonomy == "clinical"
        else {str(k): v for k, v in _PREVIEW_LUT_SIMULATION.items()}
    )
    label_doc = {
        "taxonomy": taxonomy,
        "paint_order_class_ids": paint_order,
        "folders": manifest,
        "training_uses": "{stem}_combined_mask.png in combined_masks/ (raw class_id per pixel).",
        "qa_preview_dir": f"combined_masks/{PREVIEW_SUBDIR_NAME}/",
        "qa_preview_only": (
            f"Per-slice PNG + PDF under combined_masks/{PREVIEW_SUBDIR_NAME}/ "
            f"(rasterized at {PDF_EXPORT_DPI} dpi) — not used by training."
        ),
        "preview_pdf_dpi": PDF_EXPORT_DPI,
        "preview_grey_by_class_id": lut_preview,
    }
    (combined_dir / "label_mapping.json").write_text(json.dumps(label_doc, indent=2) + "\n", encoding="utf-8")

    sorted_uids = sorted(image_map.keys(), key=lambda u: _slice_sort_key(image_map[u]))

    for uid in tqdm(sorted_uids, desc="Combined masks"):
        info = image_map[uid]
        stem = info["file_name"]
        rows, cols = int(info["rows"]), int(info["columns"])
        combined = np.zeros((rows, cols), dtype=np.uint8)
        for cid in paint_order:
            for folder in by_class.get(cid, []):
                bin_path = masks_root / folder / f"{stem}_mask.png"
                if not bin_path.is_file():
                    continue
                m = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                combined[m > 0] = cid
        out = combined_dir / f"{stem}_combined_mask.png"
        cv2.imwrite(str(out), combined)
        preview = combined_multiclass_to_preview_u8(combined, taxonomy)
        prev_png = preview_dir / f"{stem}_combined_mask_preview.png"
        cv2.imwrite(str(prev_png), preview)
        prev_pdf = preview_dir / f"{stem}_combined_mask_preview.pdf"
        try:
            _write_preview_pdf_single_page(preview, prev_pdf, dpi=PDF_EXPORT_DPI)
        except Exception as e:
            logger.warning("Preview PDF %s: %s", stem, e)

    logger.info("Combined masks → %s; QA previews → %s (PNG + PDF @ %s dpi)", combined_dir, preview_dir, PDF_EXPORT_DPI)
    return combined_dir


def maybe_remove_legacy_splits_dir(dataset_root: Path, *, dry_run: bool) -> None:
    """Remove ``splits/`` only (never ``splits_stratified/``)."""
    legacy = dataset_root / "splits"
    if not legacy.is_dir():
        return
    if dry_run:
        logger.info("Would remove legacy splits dir: %s", legacy)
        return
    shutil.rmtree(legacy)
    logger.info("Removed legacy splits dir: %s", legacy)


def run_export(
    *,
    dicom_dir: Path,
    masks_root: Path,
    taxonomy: str,
    rtstruct_path: Optional[Path],
    structures_filter: Optional[List[str]],
    remove_legacy_splits_dir: bool,
    dry_run: bool,
    training_png_dir: Optional[Path] = None,
    slice_organ_presence_json: Optional[Path] = None,
    use_slice_organ_presence_json: bool = True,
) -> None:
    _ensure_rtstruct_export_logging()
    dicom_dir = dicom_dir.expanduser().resolve()
    masks_root = masks_root.expanduser().resolve()
    if not dicom_dir.is_dir():
        raise FileNotFoundError(dicom_dir)

    rs_candidates = sorted(dicom_dir.glob("RS*.dcm")) + sorted(dicom_dir.glob("RS.*.dcm"))
    rs = rtstruct_path.expanduser().resolve() if rtstruct_path else (rs_candidates[0] if rs_candidates else None)
    if rs is None or not rs.is_file():
        raise FileNotFoundError(f"No RTSTRUCT RS*.dcm under {dicom_dir}")

    dataset_root = masks_root.parent
    if remove_legacy_splits_dir:
        maybe_remove_legacy_splits_dir(dataset_root, dry_run=dry_run)

    image_map = build_image_slice_mapping(dicom_dir)
    if not image_map:
        raise RuntimeError("No MR slices mapped; check DICOM folder")

    if dry_run:
        logger.info("Dry run: would process RTSTRUCT %s → %s", rs, masks_root)
        return

    masks_root.mkdir(parents=True, exist_ok=True)

    tpng = resolve_training_png_directory(masks_root, training_png_dir)
    image_map = filter_image_map_to_training_slices(
        image_map,
        training_png_dir=tpng,
        dicom_dir=dicom_dir,
        taxonomy=taxonomy,
        slice_organ_presence_json=slice_organ_presence_json,
        use_slice_organ_presence_json=use_slice_organ_presence_json,
    )

    _prune_mask_artifacts_outside_stems(masks_root, {str(info["file_name"]) for info in image_map.values()})

    extract_per_organ_masks(rs, image_map, masks_root, taxonomy, structures_filter)
    build_combined_masks(masks_root, image_map, taxonomy)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dicom-dir", type=Path, required=True, help="Folder with MR*.dcm and RS*.dcm")
    p.add_argument(
        "--masks-root",
        type=Path,
        required=True,
        help="e.g. preprocessing/real_data/case1/masks (combined_masks created here)",
    )
    p.add_argument("--taxonomy", choices=("clinical", "simulation"), required=True)
    p.add_argument("--rtstruct", type=Path, default=None, help="Explicit RS*.dcm (default: first under --dicom-dir)")
    p.add_argument(
        "--structures",
        type=str,
        default=None,
        help="Comma-separated exact ROINames to include (as they appear in the RTSTRUCT; "
        "default: all mappable ROIs)",
    )
    p.add_argument(
        "--training-png-dir",
        type=Path,
        default=None,
        help="Preprocessing slice PNG folder (e.g. .../case1_dicom_png). If set and non-empty, "
        "only those MR stems get masks/previews (same slice set as training).",
    )
    p.add_argument(
        "--slice-organ-json",
        type=Path,
        default=None,
        help="Optional path to preprocessing/slice_organ_presence/<case>.json (overrides auto-discovery).",
    )
    p.add_argument(
        "--no-slice-organ-json",
        action="store_true",
        help="Do not intersect stems with slice_organ_presence JSON (default: use when file exists).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    filt = [s.strip() for s in args.structures.split(",")] if args.structures else None
    run_export(
        dicom_dir=args.dicom_dir,
        masks_root=args.masks_root,
        taxonomy=args.taxonomy,
        rtstruct_path=args.rtstruct,
        structures_filter=filt,
        remove_legacy_splits_dir=False,
        dry_run=args.dry_run,
        training_png_dir=args.training_png_dir,
        slice_organ_presence_json=args.slice_organ_json,
        use_slice_organ_presence_json=not args.no_slice_organ_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
