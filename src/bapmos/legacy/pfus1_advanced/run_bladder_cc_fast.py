"""Fast bladder CC diagnostic: island stats + Dice only (no per-slice HD95/MSD)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from bapmos.organ_registry import PFUS1_ORGAN_TO_CLASS
from bapmos.paths import pfus1_bundle_dir, project_root
from bapmos.legacy.pfus1_advanced.postprocess_bladder_cc import (
    BLADDER_CLASS_ID,
    cleanup_bladder_in_multiclass,
    resolve_pred_ids_dir,
)
from bapmos.legacy.pfus1_advanced.run_bladder_cc_batch import (
    _bladder_component_stats,
    _load_official_bladder_summary,
    run_case_nnunet,
)

BLADDER_CID = int(PFUS1_ORGAN_TO_CLASS["bladder"])


def _dice(pred_b: np.ndarray, gt_b: np.ndarray) -> float:
    inter = float((pred_b & gt_b).sum())
    u = float(pred_b.sum() + gt_b.sum())
    return 1.0 if u == 0 else 2.0 * inter / u


def run_bap_mos_fast(pred_dir: Path, gt_root: Path) -> dict:
    ids_dir = resolve_pred_ids_dir(pred_dir)
    raw_d, clean_d = [], []
    multi, changed, removed = 0, 0, 0
    n = 0
    for pred_path in sorted(ids_dir.glob("*_pred_ids.png")):
        image_id = pred_path.name.replace("_pred_ids.png", "")
        gt_path = gt_root / f"{image_id}_combined_mask.png"
        if not gt_path.is_file():
            continue
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_b = pred == BLADDER_CID
        gt_b = gt == BLADDER_CID
        st = _bladder_component_stats(pred)
        if st["n_components"] > 1:
            multi += 1
        removed += st["removed_island_px"]
        cleaned = cleanup_bladder_in_multiclass(pred)
        clean_b = cleaned == BLADDER_CID
        rd, cd = _dice(pred_b, gt_b), _dice(clean_b, gt_b)
        raw_d.append(rd)
        clean_d.append(cd)
        if abs(rd - cd) > 1e-8:
            changed += 1
        n += 1
    raw_m, cl_m = float(np.mean(raw_d)), float(np.mean(clean_d))
    return {
        "method": "bap_mos_tuned",
        "n_cases": n,
        "slices_multi_component_bladder": multi,
        "slices_multi_component_frac": multi / n if n else 0,
        "slices_dice_changed_after_cc": changed,
        "total_removed_island_px": removed,
        "mean_removed_island_px": removed / n if n else 0,
        "raw": {"dice": raw_m},
        "cleaned_cc": {"dice": cl_m},
        "delta": {"dice": cl_m - raw_m},
    }


def main() -> int:
    root = project_root()
    gt = pfus1_bundle_dir() / "masks" / "combined_masks"
    out = root / "diagnostics/pfus1_bladder_cc"
    out.mkdir(parents=True, exist_ok=True)

    results = {"methods": {}}

    bap_pred = (
        root
        / "output/pfus1/Optimization/bap_mos_tuned"
        / "bap_mos_tuned_seed42_20260517_040225/test/predictions"
    )
    if bap_pred.is_dir():
        r = run_bap_mos_fast(bap_pred, gt)
        r["official_evaluator"] = _load_official_bladder_summary(
            bap_pred.parent / "summary_metrics.csv"
        )
        results["methods"]["bap_mos_tuned"] = r
        print(f"BAP-MOS done n={r['n_cases']} multi_comp={r['slices_multi_component_bladder']}")

    nn_pred = root / "runs/pfus1/nnUNet_predictions/nnunet_pfus1_2d_fold0"
    nn_map = root / "exports/nnUNet_raw/pfus1/Dataset503_BapMosPfus1TrainVal/case_mapping.json"
    if nn_pred.is_dir() and nn_map.is_file():
        print("nnUNet (full surface metrics — may take several minutes)...")
        r = run_case_nnunet(nn_pred, gt, nn_map, out / "nnunet")
        r["official_evaluator"] = _load_official_bladder_summary(
            root
            / "output/pfus1/ExternalBaselines/nnunet2d/nnunet_pfus1_2d_fold0/test/summary_metrics.csv"
        )
        results["methods"]["nnunet"] = r
        print(f"nnUNet done n={r['n_cases']}")

    for method, sub in [
        ("unet", "unet_pfus1_multiclass"),
        ("medsam_init", "medsam_pfus1_multiclass"),
    ]:
        off = _load_official_bladder_summary(
            root / f"output/pfus1/ExternalBaselines/{method}/{sub}/test/summary_metrics.csv"
        )
        if off:
            results["methods"][method] = {
                "method": method,
                "note": "No archived class-id PNGs; official evaluator metrics only",
                "official_evaluator": off,
            }

    (out / "bladder_cc_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out / 'bladder_cc_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
