"""
Bladder CC diagnostic — **canonical BAP-MOS-Tuned only** (island/outlier check).

External baselines (U-Net, nnU-Net, MedSAM) are excluded; use existing
``output/pfus1/ExternalBaselines/*/test/summary_metrics.csv`` for those.

Output: ``diagnostics/pfus1_bladder_cc/subcases/canonical_pfus1/``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bapmos.paths import pfus1_bundle_dir, project_root
from bapmos.legacy.pfus1_advanced.bap_mos_subcases import discover_bap_mos_subcases, resolve_subcase_outputs
from bapmos.legacy.pfus1_advanced.postprocess_bladder_cc import resolve_pred_ids_dir
from bapmos.legacy.pfus1_advanced.run_bladder_cc_batch import (
    _load_official_bladder_summary,
    run_case_bap_mos,
)
from bapmos.legacy.pfus1_advanced.run_bladder_cc_fast import run_bap_mos_fast


CANONICAL_SUBCASE_ID = "canonical_pfus1"


def run_canonical_bladder_cc(
    out_root: Optional[Path] = None,
    *,
    write_cleaned: bool = True,
    fast_dice_only: bool = False,
) -> Dict[str, Any]:
    root = project_root()
    out_root = (out_root or root / "diagnostics/pfus1_bladder_cc").resolve()
    subdir = out_root / "subcases" / CANONICAL_SUBCASE_ID
    subdir.mkdir(parents=True, exist_ok=True)

    sc = next(s for s in discover_bap_mos_subcases() if s.subcase_id == CANONICAL_SUBCASE_ID)
    manifest = resolve_subcase_outputs(sc)
    (subdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    gt = pfus1_bundle_dir() / "masks" / "combined_masks"
    pred_dir = Path(manifest["test_output_dir"]) / "predictions" if manifest.get("test_output_dir") else None

    result: Dict[str, Any] = {
        "subcase_id": CANONICAL_SUBCASE_ID,
        "status": "missing_predictions",
        "manifest": manifest,
    }

    if pred_dir and pred_dir.is_dir():
        if fast_dice_only or not write_cleaned:
            cc = run_bap_mos_fast(pred_dir, gt)
        else:
            cc = run_case_bap_mos(pred_dir, gt, subdir)
        result.update(cc)
        result["status"] = "cc_complete"
        result["pred_ids_dir"] = str(resolve_pred_ids_dir(pred_dir))

    if manifest.get("summary_metrics_csv"):
        result["official_evaluator"] = _load_official_bladder_summary(
            Path(manifest["summary_metrics_csv"])
        )

    result["conclusion"] = _conclude(result)
    (subdir / "metrics.json").write_text(json.dumps(result, indent=2))

    report = {
        "note": "Bladder CC on canonical BAP-MOS-Tuned only. No external baselines.",
        "subcases": {CANONICAL_SUBCASE_ID: result},
    }
    (out_root / "bladder_cc_report.json").write_text(json.dumps(report, indent=2))
    return report


def _conclude(r: Dict[str, Any]) -> str:
    if r.get("status") != "cc_complete":
        return "Missing predictions — run inference first."

    raw, cl = r.get("raw", {}), r.get("cleaned_cc", {})
    d_dice = cl.get("dice", 0) - raw.get("dice", 0)
    mc = r.get("slices_multi_component_bladder", 0)
    n = r.get("n_cases", 0)
    off = r.get("official_evaluator", {})

    msg = (
        f"n={n}, multi_component_slices={mc} ({100*mc/n if n else 0:.1f}%). "
        f"Dice after CC: {raw.get('dice', 0):.4f}→{cl.get('dice', 0):.4f} (Δ{d_dice:+.4f}). "
    )
    if abs(d_dice) < 0.002:
        msg += "CC does not fix metrics — boundary gap is not mainly small islands."
    if off:
        msg += f" Official bladder: Dice={off['dice']:.3f} MSD={off['msd_px']:.2f} HD95={off['hd95_px']:.2f} px."
    return msg


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Bladder CC — canonical BAP-MOS only")
    p.add_argument("--fast", action="store_true", help="Dice + island stats only (faster)")
    p.add_argument("--no_write_cleaned", action="store_true")
    args = p.parse_args(argv)

    report = run_canonical_bladder_cc(
        fast_dice_only=args.fast,
        write_cleaned=not args.no_write_cleaned,
    )
    print(report["subcases"][CANONICAL_SUBCASE_ID].get("conclusion", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
