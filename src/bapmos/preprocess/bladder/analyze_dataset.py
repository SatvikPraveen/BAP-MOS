"""
PFUS1 dataset statistics: label frequency from JSON and rasterized mask audits.

Dice / MSD / HD95 helpers here are **diagnostic only** (tables, sanity checks).
Canonical training/evaluation metrics use ``bapmos.legacy.optimization.metrics``.

This script does **not** train models or touch Case 1/2/Simulation pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from bapmos.paths import project_root
from bapmos.preprocess.bladder.constants import (
    JSON_LABEL_TO_CLASS_ID,
    PFUS1_ALL_LABELS,
    PFUS1_SUBSET_FIVE_LABELS,
)


def _root() -> Path:
    return project_root()


def polygon_area_xy(poly: List[List[float]]) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def analyze_json_frame(json_path: Path, min_area: float) -> Dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)
    present: Dict[str, bool] = {}
    areas: Dict[str, float] = {}
    if not isinstance(ann, list):
        return {"error": "not_a_list", "path": str(json_path)}
    for name, _ in PFUS1_ALL_LABELS:
        present[name] = False
        areas[name] = 0.0
    for obj in ann:
        label = obj.get("label")
        pol = obj.get("pol")
        if label not in JSON_LABEL_TO_CLASS_ID:
            continue
        if not isinstance(pol, list) or len(pol) < 3:
            continue
        a = polygon_area_xy(pol)
        areas[label] = areas.get(label, 0.0) + a
        if a >= min_area:
            present[label] = True
    return {"path": str(json_path), "present": present, "areas": areas}


def walk_json_stats(raw_root: Path, min_area: float) -> Dict[str, Any]:
    patients = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("P"))
    n_frames = 0
    label_frame_hits: DefaultDict[str, int] = defaultdict(int)
    label_area_sum: DefaultDict[str, float] = defaultdict(float)

    for pdir in patients:
        for jf in sorted(pdir.glob("frame_*.json")):
            st = analyze_json_frame(jf, min_area=min_area)
            if "error" in st:
                continue
            n_frames += 1
            for lab, ok in st["present"].items():
                if ok:
                    label_frame_hits[lab] += 1
                    label_area_sum[lab] += float(st["areas"][lab])

    rows = []
    for lab, _ in PFUS1_ALL_LABELS:
        hits = label_frame_hits[lab]
        rows.append(
            {
                "label": lab,
                "frames_with_structure_ge_min_area": hits,
                "fraction_of_frames": round(hits / max(n_frames, 1), 6),
                "sum_polygon_area_px2": round(label_area_sum[lab], 3),
            }
        )

    subset_report = {
        "subset_five_labels": list(PFUS1_SUBSET_FIVE_LABELS),
        "note": "Use subset only after reviewing full-label frequency; do not cherry-pick silently.",
        "subset_frame_hits": {lab: label_frame_hits[lab] for lab in PFUS1_SUBSET_FIVE_LABELS},
    }

    return {
        "n_patients": len(patients),
        "n_frames_scanned": n_frames,
        "min_polygon_area_px2": min_area,
        "label_frequency_json": rows,
        "subset": subset_report,
    }


def mask_class_pixel_counts(mask_dir: Path) -> Optional[Dict[str, Any]]:
    if not mask_dir.is_dir():
        return None
    totals: DefaultDict[int, int] = defaultdict(int)
    n_masks = 0
    for png in sorted(mask_dir.rglob("*_combined_mask.png")):
        m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        n_masks += 1
        for cid in range(1, 9):
            totals[cid] += int((m == cid).sum())
    id_to_name = {cid: name for name, cid in PFUS1_ALL_LABELS}
    rows = [
        {
            "class_id": cid,
            "label": id_to_name[cid],
            "foreground_pixels": totals[cid],
        }
        for cid in range(1, 9)
    ]
    return {"n_masks": n_masks, "pixel_counts": rows}


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    g = gt == class_id
    inter = np.logical_and(p, g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return 1.0
    return float((2.0 * inter) / (union + 1e-8))


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    eroded = binary_erosion(m)
    return np.logical_and(m, np.logical_not(eroded))


def symmetric_surface_distances_px(pred: np.ndarray, gt: np.ndarray, class_id: int) -> np.ndarray:
    """Concatenated directed surface distances in pixels (empty if missing foreground)."""
    p = (pred == class_id).astype(bool)
    g = (gt == class_id).astype(bool)
    if not p.any() and not g.any():
        return np.array([], dtype=np.float64)
    if not p.any() or not g.any():
        # One empty: distances ill-defined; return large sentinel for diagnostics
        return np.array([np.nan], dtype=np.float64)

    p_b = _boundary(p)
    g_b = _boundary(g)
    if not p_b.any() or not g_b.any():
        return np.array([np.nan], dtype=np.float64)

    dt_g = distance_transform_edt(~g)
    dt_p = distance_transform_edt(~p)
    d_pg = dt_g[p_b]
    d_gp = dt_p[g_b]
    return np.concatenate([d_pg, d_gp])


def msd_mm(distances_px: np.ndarray, spacing_mm: float) -> float:
    d = distances_px[np.isfinite(distances_px)]
    if d.size == 0:
        return float("nan")
    return float(np.mean(d)) * float(spacing_mm)


def hd95_mm(distances_px: np.ndarray, spacing_mm: float) -> float:
    d = distances_px[np.isfinite(distances_px)]
    if d.size == 0:
        return float("nan")
    return float(np.percentile(d, 95)) * float(spacing_mm)


def per_class_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: Sequence[int],
    spacing_mm: float,
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for cid in class_ids:
        d = symmetric_surface_distances_px(pred, gt, cid)
        out[cid] = {
            "dice": dice_coefficient(pred, gt, cid),
            "msd_mm": msd_mm(d, spacing_mm),
            "hd95_mm": hd95_mm(d, spacing_mm),
        }
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=_root() / "data/bladder/pfus1_raw",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=_root() / "data/bladder/pfus1/masks/combined_masks",
        help="If present, adds raster pixel-count summary.",
    )
    parser.add_argument("--min_polygon_area", type=float, default=1.0)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=_root() / "data/bladder/pfus1/reports/dataset_stats.json",
    )
    args = parser.parse_args(argv)

    if not args.raw_root.is_dir():
        print(f"ERROR: raw_root not found: {args.raw_root}", file=sys.stderr)
        return 1

    report: Dict[str, Any] = {}
    report["json_polygon_stats"] = walk_json_stats(args.raw_root, min_area=args.min_polygon_area)
    pix = mask_class_pixel_counts(args.mask_root)
    if pix:
        report["raster_pixel_stats"] = pix
    else:
        report["raster_pixel_stats"] = {
            "skipped": True,
            "reason": f"mask_root missing or empty: {args.mask_root}",
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json.resolve()}")

    # Human-readable summary to stdout (explicit label frequency table)
    rows = report["json_polygon_stats"]["label_frequency_json"]
    print("\nLabel frequency (JSON polygons, frame hit = area >= min_area)")
    print(f"{'label':<22} {'frames':>8} {'frac':>10}")
    for r in rows:
        print(f"{r['label']:<22} {r['frames_with_structure_ge_min_area']:>8} {r['fraction_of_frames']:>10.4f}")
    sub = report["json_polygon_stats"]["subset"]
    print("\nSubset-five (for comparison; decide explicitly in experiments):")
    for lab in PFUS1_SUBSET_FIVE_LABELS:
        h = sub["subset_frame_hits"][lab]
        tot = report["json_polygon_stats"]["n_frames_scanned"]
        print(f"  {lab:<22} {h:>8}  ({h / max(tot,1):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
