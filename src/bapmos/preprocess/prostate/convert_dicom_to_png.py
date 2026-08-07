"""
Convert DICOM slice files to 8-bit PNG using 1st–99th percentile windowing.

Project-specific preprocessing for BAP-MOS Varian-exported TRUS volumes
(DICOM filenames often look like MR.*.dcm) — not a general-purpose DICOM
converter. RTSTRUCT / RTPLAN objects (RS.* / RP.*) are skipped.

Side effects are limited to: writing ``*.png`` under the chosen output directory
(creating that directory if needed), optionally removing a stale PNG when a slice
is reclassified as blank, and appending to ``logs/dicom_conversion.log`` under the
BAPMOS checkout when a batch conversion runs. Importing this module does not create
directories or configure global logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image

from bapmos.paths import project_root

_LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_FILE_LOG_ATTACHED = False

logger = logging.getLogger(__name__)


def _dicom_conversion_log_path() -> Path:
    return project_root() / "logs" / "dicom_conversion.log"


def _ensure_dicom_conversion_file_logging() -> None:
    """
    Append to ``logs/dicom_conversion.log`` when conversion runs.

    No logging configuration runs at import time (avoids touching the root logger or
    creating ``logs/`` until a conversion actually runs).
    """
    global _FILE_LOG_ATTACHED
    if _FILE_LOG_ATTACHED:
        return
    _FILE_LOG_ATTACHED = True
    log_path = _dicom_conversion_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_LOG_FORMAT)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # Do not reconfigure the root logger or other libraries.
    logger.propagate = False

# Same gate as RTSTRUCT / training slice lists: after percentile windowing, “blank” slices.
BLANK_SLICE_MAX_INTENSITY = 5
BLANK_SLICE_MIN_FRACTION_ABOVE = 1e-4


def is_blank_training_slice_uint8(img_u8: np.ndarray) -> bool:
    """
    True if this slice is effectively empty after ``dicom_to_uint8`` (black / no signal).

    Used by PNG export and by ``filter_image_map_to_training_slices`` so masks and
    previews are not generated for useless slices.
    """
    if img_u8.size == 0:
        return True
    if int(np.max(img_u8)) <= BLANK_SLICE_MAX_INTENSITY:
        return True
    thr = BLANK_SLICE_MAX_INTENSITY
    frac = float(np.count_nonzero(img_u8 > thr)) / float(img_u8.size)
    return frac < BLANK_SLICE_MIN_FRACTION_ABOVE


def dicom_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Percentile-based window to uint8 (matches project preprocessing description).
    """
    img = img.astype(np.float32, copy=False)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.uint8)
    vals = img[finite]
    lo, hi = np.percentile(vals, (1.0, 99.0))
    if hi <= lo + 1e-6:
        return np.full(img.shape, 128, dtype=np.uint8)
    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo)
    out = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return out


def _to_2d_grayscale(arr: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """Reduce pixel_array to 2D grayscale float for windowing."""
    if arr.ndim == 2:
        plane = arr
    elif arr.ndim == 3:
        # (H, W, C) with small C — color or label stack
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            plane = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        elif arr.shape[-1] == 1:
            plane = arr[..., 0].astype(np.float32)
        else:
            # (frames, rows, cols) — take first frame
            plane = arr[0].astype(np.float32)
    elif arr.ndim == 4:
        plane = arr[0, 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported pixel_array ndim={arr.ndim}, shape={arr.shape}")

    plane = plane.astype(np.float32, copy=False)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pmin = float(np.nanmin(plane))
        pmax = float(np.nanmax(plane))
        plane = (pmax + pmin) - plane
    return plane


def mr_dicom_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """MR pixel data → uint8 slice (same path as ``convert_one_dicom``)."""
    arr = ds.pixel_array
    plane = _to_2d_grayscale(np.asarray(arr), ds)
    return dicom_to_uint8(plane)


def dicom_slice_is_blank_for_training(dcm_path: Path) -> bool:
    """True if this MR DICOM would not be kept as a training PNG (blank after windowing)."""
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
    except Exception:
        return True
    if not hasattr(ds, "PixelData"):
        return True
    try:
        img8 = mr_dicom_to_uint8(ds)
    except Exception:
        return True
    return is_blank_training_slice_uint8(img8)


def should_skip_path(path: Path) -> bool:
    name = path.name.upper()
    if not name.endswith(".DCM"):
        return True
    if name.startswith("RS.") or name.startswith("RP."):
        return True
    return False


def convert_one_dicom(
    dcm_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Read one DICOM and write ``{stem}.png`` into out_dir.

    Returns (success, error_message).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dcm_path.stem}.png"
    if out_path.is_file() and not overwrite:
        return True, None

    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
    except Exception as e:
        return False, f"read_error: {e}"

    if not hasattr(ds, "PixelData"):
        return False, "no_pixel_data"

    try:
        arr = ds.pixel_array
    except Exception as e:
        return False, f"pixel_array: {e}"

    try:
        plane = _to_2d_grayscale(np.asarray(arr), ds)
        img8 = dicom_to_uint8(plane)
        if is_blank_training_slice_uint8(img8):
            if out_path.is_file():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            return True, "skipped_blank"
        Image.fromarray(img8, mode="L").save(out_path)
    except Exception as e:
        return False, f"save: {e}"

    return True, None


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Tuple[int, int, int, List[str]]:
    """
    Convert all eligible ``*.dcm`` under input_dir (non-recursive).

    Writes only ``*.png`` under ``output_dir`` (and creates that directory). Optional
    append-only lines to ``logs/dicom_conversion.log``. Does not read or modify DICOM files.

    Returns (n_written, n_skipped_blank, n_fail, list of failure lines).
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    _ensure_dicom_conversion_file_logging()

    failures: List[str] = []
    written = 0
    skipped_blank = 0
    fail = 0
    dcms = sorted(input_dir.glob("*.dcm"))
    for p in dcms:
        if should_skip_path(p):
            logger.info("Skip %s", p.name)
            continue
        good, err = convert_one_dicom(p, output_dir, overwrite=overwrite)
        if good:
            if err == "skipped_blank":
                skipped_blank += 1
                logger.info("Skip blank slice %s (no PNG written)", p.name)
            else:
                written += 1
                logger.info("OK %s -> %s.png", p.name, p.stem)
        else:
            fail += 1
            line = f"{p.name}: {err}"
            failures.append(line)
            logger.warning("FAIL %s", line)

    return written, skipped_blank, fail, failures


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DICOM slices in a folder to PNG.")
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing .dcm files (e.g. data/real_data/Case 1/Dicom).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write PNG files (created if missing).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNGs in the output directory.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(_LOG_FORMAT)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
    ok, sk, fail, failures = convert_directory(
        args.input.resolve(),
        args.output.resolve(),
        overwrite=args.overwrite,
    )
    logger.info("Done: %d written, %d skipped (blank), %d failed", ok, sk, fail)
    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 1 if fail > 0 else 0
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
