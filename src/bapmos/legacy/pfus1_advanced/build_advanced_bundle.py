"""
Build ``data/bladder/pfus1_advanced`` from raw PFUS1 + existing pfus1 masks.

Does **not** modify ``data/bladder/pfus1``.

Outputs:
  - ``images/Pxxx/frame_yyy.png`` — letterboxed grayscale ultrasound
  - ``masks/combined_masks/Pxxx_frame_yyy_combined_mask.png`` — warped labels
  - ``splits_patient_70_15_15_seed42/`` — copied from pfus1 (same patient protocol)
  - ``preprocess_manifest.jsonl`` — per-frame crop/letterbox metadata
  - ``reports/preprocess_stats.json`` — size / crop summary
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from bapmos.paths import (
    pfus1_advanced_bundle_dir,
    pfus1_bundle_dir,
    pfus1_image_root,
    project_root,
)
from bapmos.pfus1.dataset_pfus1 import parse_sample_line
from bapmos.legacy.pfus1_advanced.ultrasound_preprocess import (
    preprocess_ultrasound_frame,
    warp_mask_with_params,
)


def _root() -> Path:
    return project_root()


def _collect_all_sample_keys(raw_root: Path) -> List[str]:
    keys: List[str] = []
    for patient_dir in sorted(raw_root.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("P"):
            continue
        for png in sorted(patient_dir.glob("frame_*.png")):
            stem = png.stem
            if (patient_dir / f"{stem}.json").is_file():
                keys.append(f"{patient_dir.name}/{stem}")
    return keys


def build_bundle(
    *,
    out_root: Path,
    raw_root: Path,
    mask_root: Path,
    splits_src: Path,
    canvas_w: int,
    canvas_h: int,
    strip_legend: bool,
    overwrite: bool,
    max_frames: int | None,
) -> Dict[str, Any]:
    images_out = out_root / "images"
    masks_out = out_root / "masks" / "combined_masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    sample_keys = _collect_all_sample_keys(raw_root)
    if max_frames is not None:
        sample_keys = sample_keys[: max_frames]

    manifest_path = out_root / "preprocess_manifest.jsonl"
    manifest_lines: List[str] = []
    skipped = 0
    processed = 0
    canvas_sizes: List[tuple[int, int]] = []

    for key in sample_keys:
        patient, stem = parse_sample_line(key)
        img_path = raw_root / patient / f"{stem}.png"
        mask_path = mask_root / f"{patient}_{stem}_combined_mask.png"
        out_img = images_out / patient / f"{stem}.png"
        out_mask = masks_out / f"{patient}_{stem}_combined_mask.png"

        if out_img.is_file() and out_mask.is_file() and not overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Missing image: {img_path}")
        if mask is None:
            raise IOError(f"Missing mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch {key}: image {image.shape} vs mask {mask.shape}"
            )

        proc_img, params = preprocess_ultrasound_frame(
            image,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            strip_legend=strip_legend,
        )
        proc_mask = warp_mask_with_params(mask, params)

        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), proc_img)
        cv2.imwrite(str(out_mask), proc_mask)

        rec = {
            "sample_key": key,
            "src_size": [int(params.src_w), int(params.src_h)],
            "crop": [params.crop_x0, params.crop_y0, params.crop_x1, params.crop_y1],
            "scale": float(params.scale),
            "pad": [params.pad_left, params.pad_top],
            "canvas": [params.canvas_w, params.canvas_h],
        }
        manifest_lines.append(json.dumps(rec))
        canvas_sizes.append((params.canvas_w, params.canvas_h))
        processed += 1

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))

    splits_dst = out_root / splits_src.name
    if splits_src.is_dir():
        if splits_dst.exists():
            shutil.rmtree(splits_dst)
        shutil.copytree(splits_src, splits_dst)

    stats = {
        "processed": processed,
        "skipped_existing": skipped,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "strip_legend": strip_legend,
        "n_samples": len(sample_keys),
    }
    reports = out_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "preprocess_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/bladder/pfus1_advanced bundle.")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output bundle (default: data/bladder/pfus1_advanced via pfus1_advanced_bundle_dir)",
    )
    parser.add_argument("--raw_root", type=Path, default=None, help="Raw PFUS1 images root")
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=None,
        help="Source combined masks (default: data/bladder/pfus1/masks/combined_masks)",
    )
    parser.add_argument("--canvas_w", type=int, default=768)
    parser.add_argument("--canvas_h", type=int, default=576)
    parser.add_argument("--no_strip_legend", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None, help="Smoke: limit frames")
    args = parser.parse_args(argv)

    out_root = args.out_root or pfus1_advanced_bundle_dir()
    raw_root = args.raw_root or pfus1_image_root()
    mask_root = args.mask_root or (pfus1_bundle_dir() / "masks" / "combined_masks")
    splits_src = pfus1_bundle_dir() / "splits_patient_70_15_15_seed42"

    if not raw_root.is_dir():
        print(f"ERROR: raw_root missing: {raw_root}", file=sys.stderr)
        return 1
    if not mask_root.is_dir():
        print(f"ERROR: mask_root missing: {mask_root}", file=sys.stderr)
        return 1
    if not splits_src.is_dir():
        print(f"ERROR: splits missing: {splits_src}", file=sys.stderr)
        return 1

    stats = build_bundle(
        out_root=_root() / out_root if not out_root.is_absolute() else out_root,
        raw_root=_root() / raw_root if not raw_root.is_absolute() else raw_root,
        mask_root=_root() / mask_root if not mask_root.is_absolute() else mask_root,
        splits_src=_root() / splits_src if not splits_src.is_absolute() else splits_src,
        canvas_w=args.canvas_w,
        canvas_h=args.canvas_h,
        strip_legend=not args.no_strip_legend,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
