"""
Batch bladder connected-component diagnostic for PFUS1 test predictions.

Writes only under ``diagnostics/pfus1_bladder_cc/`` — never overwrites
``output/pfus1/`` predictions or ``runs/pfus1/`` checkpoints.

Methods:
  - bap_mos_tuned (multiclass PNGs in output/.../predictions)
  - nnunet (caseXXXXX.png + case_mapping.json)
  - unet / medsam_init (export test masks on the fly if missing)
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
from bapmos.paths import pfus1_bundle_dir, project_root
from bapmos.legacy.pfus1_advanced.postprocess_bladder_cc import (
    BLADDER_CLASS_ID,
    cleanup_bladder_in_multiclass,
    evaluate_bladder_pair,
    largest_component_binary,
    resolve_pred_ids_dir,
)

BLADDER_KEY = "bladder"


def _bladder_component_stats(pred: np.ndarray) -> Dict[str, Any]:
    bladder = (pred == BLADDER_CLASS_ID).astype(np.uint8)
    area = int(bladder.sum())
    if area == 0:
        return {"area_px": 0, "n_components": 0, "largest_frac": 0.0, "removed_island_px": 0}

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bladder, connectivity=8)
    n_fg = num - 1
    if n_fg <= 0:
        return {"area_px": 0, "n_components": 0, "largest_frac": 0.0, "removed_island_px": 0}

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    removed = int(area - largest)
    return {
        "area_px": area,
        "n_components": n_fg,
        "largest_frac": float(largest / area) if area else 0.0,
        "removed_island_px": removed,
    }


def _aggregate(rows: List[Dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def run_case_bap_mos(
    pred_dir: Path,
    gt_root: Path,
    out_root: Path,
    *,
    stem_suffix: str = "_pred_ids.png",
) -> Dict[str, Any]:
    ids_dir = resolve_pred_ids_dir(pred_dir)
    clean_dir = out_root / "predictions_cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, float]] = []
    clean_rows: List[Dict[str, float]] = []
    multi_comp = 0
    changed_slices = 0
    total_removed_px = 0
    n = 0

    for pred_path in sorted(ids_dir.glob(f"*{stem_suffix}")):
        stem = pred_path.name.replace(stem_suffix, "")
        base = stem.replace("_pred_ids", "").replace("_pred", "")
        gt_path = gt_root / f"{base}_combined_mask.png"
        if not gt_path.is_file():
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        stats = _bladder_component_stats(pred)
        if stats["n_components"] > 1:
            multi_comp += 1
        total_removed_px += stats["removed_island_px"]

        cleaned = cleanup_bladder_in_multiclass(pred)
        clean_path = clean_dir / pred_path.name
        cv2.imwrite(str(clean_path), cleaned)

        raw_m = evaluate_bladder_pair(pred_path, gt_path)
        clean_m = evaluate_bladder_pair(clean_path, gt_path)
        raw_rows.append(raw_m)
        clean_rows.append(clean_m)

        if (
            abs(raw_m["dice"] - clean_m["dice"]) > 1e-6
            or abs(raw_m["msd_px"] - clean_m["msd_px"]) > 1e-6
            or abs(raw_m["hd95_px"] - clean_m["hd95_px"]) > 1e-6
        ):
            changed_slices += 1
        n += 1

    return _build_summary(
        method="bap_mos_tuned",
        n=n,
        multi_comp=multi_comp,
        changed_slices=changed_slices,
        total_removed_px=total_removed_px,
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        pred_dir=str(pred_dir),
    )


def run_case_nnunet(
    pred_dir: Path,
    gt_root: Path,
    case_mapping_path: Path,
    out_root: Path,
) -> Dict[str, Any]:
    with open(case_mapping_path) as f:
        mapping = json.load(f)
    test_entries = mapping.get("test") or []

    clean_dir = out_root / "predictions_cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, float]] = []
    clean_rows: List[Dict[str, float]] = []
    multi_comp = 0
    changed_slices = 0
    total_removed_px = 0
    n = 0

    for entry in test_entries:
        case_id = entry["case_id"]
        image_id = entry["image_id"]
        pred_path = pred_dir / f"{case_id}.png"
        if not pred_path.is_file():
            pred_path = pred_dir / f"{case_id}_0000.png"
        gt_path = gt_root / f"{image_id}_combined_mask.png"
        if not pred_path.is_file() or not gt_path.is_file():
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        stats = _bladder_component_stats(pred)
        if stats["n_components"] > 1:
            multi_comp += 1
        total_removed_px += stats["removed_island_px"]

        cleaned = cleanup_bladder_in_multiclass(pred)
        out_name = f"{image_id}_pred.png"
        clean_path = clean_dir / out_name
        cv2.imwrite(str(clean_path), cleaned)

        # Evaluate bladder binary from arrays (same geometry as pred file)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_b = (pred == BLADDER_CLASS_ID).astype(np.uint8)
        gt_b = (gt == BLADDER_CLASS_ID).astype(np.uint8)
        clean_b = (cleaned == BLADDER_CLASS_ID).astype(np.uint8)

        from bapmos.legacy.pfus1_advanced.postprocess_bladder_cc import _dice, _surface_distances

        raw_m = {
            "dice": _dice(pred_b, gt_b),
            "msd_px": _surface_distances(pred_b, gt_b)[0],
            "hd95_px": _surface_distances(pred_b, gt_b)[1],
        }
        clean_m = {
            "dice": _dice(clean_b, gt_b),
            "msd_px": _surface_distances(clean_b, gt_b)[0],
            "hd95_px": _surface_distances(clean_b, gt_b)[1],
        }
        raw_rows.append(raw_m)
        clean_rows.append(clean_m)

        if (
            abs(raw_m["dice"] - clean_m["dice"]) > 1e-6
            or abs(raw_m["msd_px"] - clean_m["msd_px"]) > 1e-6
            or abs(raw_m["hd95_px"] - clean_m["hd95_px"]) > 1e-6
        ):
            changed_slices += 1
        n += 1

    return _build_summary(
        method="nnunet",
        n=n,
        multi_comp=multi_comp,
        changed_slices=changed_slices,
        total_removed_px=total_removed_px,
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        pred_dir=str(pred_dir),
    )


def run_case_from_pred_list(
    pred_paths: List[Tuple[Path, str]],
    gt_root: Path,
    out_root: Path,
    *,
    method: str,
) -> Dict[str, Any]:
    """pred_paths: list of (pred_png, image_id e.g. P001_frame_000)."""
    clean_dir = out_root / "predictions_cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, float]] = []
    clean_rows: List[Dict[str, float]] = []
    multi_comp = 0
    changed_slices = 0
    total_removed_px = 0
    n = 0

    for pred_path, image_id in pred_paths:
        gt_path = gt_root / f"{image_id}_combined_mask.png"
        if not pred_path.is_file() or not gt_path.is_file():
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        stats = _bladder_component_stats(pred)
        if stats["n_components"] > 1:
            multi_comp += 1
        total_removed_px += stats["removed_island_px"]

        cleaned = cleanup_bladder_in_multiclass(pred)
        clean_path = clean_dir / f"{image_id}_pred.png"
        cv2.imwrite(str(clean_path), cleaned)

        raw_m = evaluate_bladder_pair(pred_path, gt_path)
        clean_m = evaluate_bladder_pair(clean_path, gt_path)
        raw_rows.append(raw_m)
        clean_rows.append(clean_m)

        if (
            abs(raw_m["dice"] - clean_m["dice"]) > 1e-6
            or abs(raw_m["msd_px"] - clean_m["msd_px"]) > 1e-6
            or abs(raw_m["hd95_px"] - clean_m["hd95_px"]) > 1e-6
        ):
            changed_slices += 1
        n += 1

    return _build_summary(
        method=method,
        n=n,
        multi_comp=multi_comp,
        changed_slices=changed_slices,
        total_removed_px=total_removed_px,
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        pred_dir=str(out_root / "exported_preds"),
    )


def _build_summary(
    *,
    method: str,
    n: int,
    multi_comp: int,
    changed_slices: int,
    total_removed_px: int,
    raw_rows: List[Dict[str, float]],
    clean_rows: List[Dict[str, float]],
    pred_dir: str,
) -> Dict[str, Any]:
    raw = {
        "dice": _aggregate(raw_rows, "dice"),
        "msd_px": _aggregate(raw_rows, "msd_px"),
        "hd95_px": _aggregate(raw_rows, "hd95_px"),
    }
    cleaned = {
        "dice": _aggregate(clean_rows, "dice"),
        "msd_px": _aggregate(clean_rows, "msd_px"),
        "hd95_px": _aggregate(clean_rows, "hd95_px"),
    }
    return {
        "method": method,
        "n_cases": n,
        "pred_dir": pred_dir,
        "slices_multi_component_bladder": multi_comp,
        "slices_multi_component_frac": float(multi_comp / n) if n else 0.0,
        "slices_metrics_changed_after_cc": changed_slices,
        "slices_changed_frac": float(changed_slices / n) if n else 0.0,
        "total_removed_island_px": total_removed_px,
        "mean_removed_island_px_per_slice": float(total_removed_px / n) if n else 0.0,
        "raw": raw,
        "cleaned_cc": cleaned,
        "delta": {
            "dice": cleaned["dice"] - raw["dice"],
            "msd_px": cleaned["msd_px"] - raw["msd_px"],
            "hd95_px": cleaned["hd95_px"] - raw["hd95_px"],
        },
    }


def _load_official_bladder_summary(summary_csv: Path) -> Optional[Dict[str, float]]:
    if not summary_csv.is_file():
        return None
    import csv

    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            if row.get("organ") == "Bladder":
                return {
                    "dice": float(row["dice_mean"]),
                    "msd_px": float(row["msd_mm_mean"]),
                    "hd95_px": float(row["hd95_mm_mean"]),
                }
    return None


def main(argv: Optional[List[str]] = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="PFUS1 bladder CC batch diagnostic")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=root / "diagnostics" / "pfus1_bladder_cc",
    )
    parser.add_argument(
        "--gt_mask_root",
        type=Path,
        default=pfus1_bundle_dir() / "masks" / "combined_masks",
    )
    parser.add_argument("--skip_export", action="store_true", help="Skip U-Net/MedSAM export")
    args = parser.parse_args(argv)

    out_root = args.out_root.resolve()
    gt_root = args.gt_mask_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"methods": {}, "official_summary_bladder": {}}

    # --- BAP-MOS ---
    bap_pred = (
        root
        / "output/pfus1/Optimization/bap_mos_tuned/bap_mos_tuned_seed42_20260517_040225/test/predictions"
    )
    if bap_pred.is_dir():
        r = run_case_bap_mos(bap_pred, gt_root, out_root / "bap_mos_tuned")
        off = _load_official_bladder_summary(
            bap_pred.parent / "summary_metrics.csv"
        )
        if off:
            r["official_evaluator"] = off
        results["methods"]["bap_mos_tuned"] = r
        print(f"[ok] bap_mos_tuned n={r['n_cases']}")
    else:
        print(f"[skip] missing {bap_pred}", file=sys.stderr)

    # --- nnU-Net ---
    nn_pred = root / "runs/pfus1/nnUNet_predictions/nnunet_pfus1_2d_fold0"
    nn_map = (
        root
        / "exports/nnUNet_raw/pfus1/Dataset503_BapMosPfus1TrainVal/case_mapping.json"
    )
    if nn_pred.is_dir() and nn_map.is_file():
        r = run_case_nnunet(nn_pred, gt_root, nn_map, out_root / "nnunet")
        off = _load_official_bladder_summary(
            root
            / "output/pfus1/ExternalBaselines/nnunet2d/nnunet_pfus1_2d_fold0/test/summary_metrics.csv"
        )
        if off:
            r["official_evaluator"] = off
        results["methods"]["nnunet"] = r
        print(f"[ok] nnunet n={r['n_cases']}")
    else:
        print(f"[skip] nnunet pred or mapping missing", file=sys.stderr)

    # --- U-Net / MedSAM: export preds to diagnostics only ---
    if not args.skip_export:
        try:
            from bapmos.legacy.pfus1_advanced.export_test_preds_for_cc import export_method_test_preds

            for method, run_name, module in [
                ("unet", "unet_pfus1_multiclass", "unet"),
                ("medsam_init", "medsam_pfus1_multiclass", "medsam_init"),
            ]:
                export_dir = out_root / method / "exported_preds"
                export_dir.mkdir(parents=True, exist_ok=True)
                paths = export_method_test_preds(
                    module=module,
                    run_name=run_name,
                    run_root=root / "runs/pfus1/ExternalBaselines",
                    export_dir=export_dir,
                )
                if paths:
                    r = run_case_from_pred_list(paths, gt_root, out_root / method, method=method)
                    off = _load_official_bladder_summary(
                        root
                        / f"output/pfus1/ExternalBaselines/{module}/{run_name}/test/summary_metrics.csv"
                    )
                    if off:
                        r["official_evaluator"] = off
                    results["methods"][method] = r
                    print(f"[ok] {method} n={r['n_cases']} (exported)")
                else:
                    print(f"[skip] {method} export returned no paths", file=sys.stderr)
        except Exception as e:
            print(f"[warn] U-Net/MedSAM export skipped: {e}", file=sys.stderr)

    report_path = out_root / "bladder_cc_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {report_path}")
    _print_conclusions(results)
    return 0


def _print_conclusions(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("BLADDER CC DIAGNOSTIC — CONCLUSIONS (PFUS1 test, n=2101 slices)")
    print("=" * 72)
    for name, r in results.get("methods", {}).items():
        print(f"\n--- {name} ---")
        if r.get("n_cases", 0) == 0:
            print("  No cases processed.")
            continue
        mc = r["slices_multi_component_bladder"]
        mc_frac = 100.0 * r["slices_multi_component_frac"]
        ch = r["slices_metrics_changed_after_cc"]
        raw = r["raw"]
        cl = r["cleaned_cc"]
        d = r["delta"]
        print(f"  Multi-component bladder slices: {mc} ({mc_frac:.1f}%)")
        print(f"  Slices with metric change after CC: {ch} ({100*ch/r['n_cases']:.1f}%)")
        print(f"  Mean removed island px/slice: {r['mean_removed_island_px_per_slice']:.1f}")
        print(f"  Dice:  {raw['dice']:.4f} -> {cl['dice']:.4f}  (delta {d['dice']:+.4f})")
        print(f"  MSD:   {raw['msd_px']:.2f} -> {cl['msd_px']:.2f} px  (delta {d['msd_px']:+.2f})")
        print(f"  HD95:  {raw['hd95_px']:.2f} -> {cl['hd95_px']:.2f} px  (delta {d['hd95_px']:+.2f})")
        off = r.get("official_evaluator")
        if off:
            print(
                f"  Official evaluator (full pipeline): Dice={off['dice']:.4f} "
                f"MSD={off['msd_px']:.2f} HD95={off['hd95_px']:.2f}"
            )
        # Interpretation
        if abs(d["hd95_px"]) < 0.5 and abs(d["msd_px"]) < 0.3:
            print("  >> CC has negligible effect — boundary gap is NOT mainly small islands.")
        elif d["hd95_px"] < -2.0 or d["msd_px"] < -1.0:
            print("  >> CC improves boundaries materially — outliers/islands contribute to poor HD95/MSD.")
        else:
            print("  >> CC has modest effect — mixed island + contour error.")


if __name__ == "__main__":
    raise SystemExit(main())
