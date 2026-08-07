"""
Diagnostic: largest-connected-component cleanup on bladder predictions, recompute metrics.

Reads existing test predictions under ``output/pfus1/`` or ``output/pfus1_advanced/``
without overwriting originals. Writes cleaned masks + metrics to a sibling directory.

Example::

    python -m bapmos.legacy.pfus1_advanced.postprocess_bladder_cc \\
        --pred_dir output/pfus1/Optimization/bap_mos_tuned/.../test/predictions \\
        --gt_mask_root data/bladder/pfus1/masks/combined_masks \\
        --out_suffix _bladder_cc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from bapmos.organ_registry import PFUS1_ORGAN_TO_CLASS
from bapmos.paths import project_root


BLADDER_CLASS_ID = int(PFUS1_ORGAN_TO_CLASS["bladder"])


def resolve_pred_ids_dir(pred_dir: Path) -> Path:
    """BAP-MOS class-id masks: ``predictions/multiclass/*_pred_ids.png``."""
    ids_dir = pred_dir / "multiclass"
    if ids_dir.is_dir() and any(ids_dir.glob("*_pred_ids.png")):
        return ids_dir
    return pred_dir


def largest_component_binary(mask_bin: np.ndarray) -> np.ndarray:
    """Keep largest 8-connected foreground component; fill holes."""
    if mask_bin.sum() == 0:
        return mask_bin
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask_bin
    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    out = (labels == largest).astype(np.uint8)
    # fill holes
    h, w = out.shape
    flood = out.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.clip(out | holes, 0, 1).astype(np.uint8)


def cleanup_bladder_in_multiclass(pred: np.ndarray) -> np.ndarray:
    out = pred.copy()
    bladder = (out == BLADDER_CLASS_ID).astype(np.uint8)
    cleaned = largest_component_binary(bladder)
    out[out == BLADDER_CLASS_ID] = 0
    out[cleaned > 0] = BLADDER_CLASS_ID
    return out


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    union = float(pred.sum() + gt.sum())
    if union == 0:
        return 1.0
    return 2.0 * inter / union


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """MSD and HD95 in pixels (symmetric)."""
    from scipy.ndimage import distance_transform_edt

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not pred_b.any() and not gt_b.any():
        return 0.0, 0.0
    if pred_b.any() != gt_b.any():
        if not pred_b.any() or not gt_b.any():
            # one empty: use max image diagonal as penalty
            h, w = pred.shape
            d = float(np.hypot(h, w))
            return d, d

    dt_pred = distance_transform_edt(~pred_b)
    dt_gt = distance_transform_edt(~gt_b)
    surf_pred = pred_b ^ cv2.erode(pred_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    surf_gt = gt_b ^ cv2.erode(gt_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not surf_pred.any():
        surf_pred = pred_b
    if not surf_gt.any():
        surf_gt = gt_b

    d_p2g = dt_gt[surf_pred]
    d_g2p = dt_pred[surf_gt]
    all_d = np.concatenate([d_p2g, d_g2p]) if len(d_p2g) and len(d_g2p) else np.array([0.0])
    msd = float(np.mean(all_d)) if all_d.size else 0.0
    hd95 = float(np.percentile(all_d, 95)) if all_d.size else 0.0
    return msd, hd95


def evaluate_bladder_pair(pred_path: Path, gt_path: Path) -> Dict[str, float]:
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        raise IOError(f"Missing pred or gt: {pred_path}, {gt_path}")

    pred_b = (pred == BLADDER_CLASS_ID).astype(np.uint8)
    gt_b = (gt == BLADDER_CLASS_ID).astype(np.uint8)
    dice = _dice(pred_b, gt_b)
    msd, hd95 = _surface_distances(pred_b, gt_b)
    return {"dice": dice, "msd_px": msd, "hd95_px": hd95}


def run_diagnostic(
    pred_dir: Path,
    gt_mask_root: Path,
    out_dir: Path,
    *,
    stem_suffix: str = "_pred.png",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics: List[Dict[str, float]] = []
    clean_metrics: List[Dict[str, float]] = []

    for pred_path in sorted(pred_dir.glob(f"*{stem_suffix}")):
        stem = pred_path.name.replace(stem_suffix, "")
        # PFUS1 pred stems: Pxxx_frame_yyy_pred -> Pxxx_frame_yyy
        base = stem.replace("_pred", "") if stem.endswith("_pred") else stem
        gt_path = gt_mask_root / f"{base}_combined_mask.png"
        if not gt_path.is_file():
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        cleaned = cleanup_bladder_in_multiclass(pred)
        clean_path = out_dir / pred_path.name
        cv2.imwrite(str(clean_path), cleaned)

        raw_metrics.append(evaluate_bladder_pair(pred_path, gt_path))
        clean_metrics.append(evaluate_bladder_pair(clean_path, gt_path))

    def _mean(key: str, rows: List[Dict[str, float]]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "n_cases": len(raw_metrics),
        "raw": {
            "dice": _mean("dice", raw_metrics),
            "msd_px": _mean("msd_px", raw_metrics),
            "hd95_px": _mean("hd95_px", raw_metrics),
        },
        "cleaned_cc": {
            "dice": _mean("dice", clean_metrics),
            "msd_px": _mean("msd_px", clean_metrics),
            "hd95_px": _mean("hd95_px", clean_metrics),
        },
    }
    (out_dir / "bladder_cc_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bladder CC cleanup diagnostic on PFUS1 preds.")
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="BAP-MOS: .../test/predictions (uses .../multiclass/*_pred_ids.png if present)",
    )
    parser.add_argument("--gt_mask_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = project_root()
    pred_dir = root / args.pred_dir if not args.pred_dir.is_absolute() else args.pred_dir
    gt_root = root / args.gt_mask_root if not args.gt_mask_root.is_absolute() else args.gt_mask_root
    out_dir = args.out_dir or (pred_dir.parent / "predictions_bladder_cc")
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if not pred_dir.is_dir():
        print(f"ERROR: pred_dir missing: {pred_dir}", file=sys.stderr)
        return 1

    pred_ids_dir = resolve_pred_ids_dir(pred_dir)
    summary = run_diagnostic(pred_ids_dir, gt_root, out_dir, stem_suffix="_pred_ids.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
