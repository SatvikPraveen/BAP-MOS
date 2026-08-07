"""
Summarize RTSTRUCT (RS.*.dcm) and RTPLAN (RP.*.dcm) metadata and relate it to
segmentation masks on disk (combined_masks + per-organ PNG counts).

**Path contract.** Case IDs use the package underscore form (``simulation``,
``case_1``, ``case_2``). DICOM folders are resolved via
``bapmos.preprocess.prostate.rtstruct_dicom_roots``. Mask roots use
``simulation_dataset_dir`` / ``real_case_dataset_dir`` (BAPMOS ``data/prostate/...``,
with optional parent-layout fallbacks). Outputs always write under the BAPMOS
checkout (``project_root()``).

Run from BAPMOS checkout root::

    python -m bapmos.preprocess.delineation.summarize_rtstruct_rtplan_masks \\
        --out data/prostate/rt_rs_rp_mask_summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pydicom

from bapmos.paths import (
    project_root,
    real_case_dataset_dir,
    research_tree_root,
    simulation_dataset_dir,
)
from bapmos.preprocess.prostate.rtstruct_dicom_roots import resolve_dicom_dir_for_case

# Public report IDs (underscore form). Folder keys for helpers may still be case1/case2.
DEFAULT_CASE_IDS: Tuple[str, ...] = ("simulation", "case_1", "case_2")


def _repo_root() -> Path:
    """Optional parent layout for DICOM/mask corpora; else BAPMOS checkout.

    Outputs still go under ``project_root()`` so reports land in the package tree
    even when this command reads DICOM/masks from outside the checkout.
    """
    return research_tree_root() or project_root()


def _masks_root_for_case(case_id: str) -> Path:
    if case_id == "simulation":
        return simulation_dataset_dir()
    if case_id == "case_1":
        return real_case_dataset_dir("case1")
    if case_id == "case_2":
        return real_case_dataset_dir("case2")
    raise ValueError(f"Unknown case_id {case_id!r}")


def find_rs_rp(dicom_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    rs = sorted(dicom_dir.glob("RS.*.dcm"))
    rp = sorted(dicom_dir.glob("RP.*.dcm"))
    return (rs[0] if rs else None, rp[0] if rp else None)


def referenced_series_uids_from_rtstruct(ds: pydicom.dataset.FileDataset) -> List[str]:
    uids: List[str] = []
    seq = getattr(ds, "ReferencedFrameOfReferenceSequence", None)
    if not seq:
        return uids
    for fr in seq:
        studies = getattr(fr, "RTReferencedStudySequence", None) or []
        for study in studies:
            series_items = getattr(study, "RTReferencedSeriesSequence", None) or []
            for ser in series_items:
                uid = getattr(ser, "SeriesInstanceUID", None)
                if uid:
                    uids.append(str(uid))
    return uids


def count_mr_slices_for_series(dicom_dir: Path, series_uid: str) -> int:
    """Count MR.*.dcm in ``dicom_dir`` matching ``series_uid``.

    Opens every MR file with ``stop_before_pixels=True``; acceptable for summary
    use but can be slow on very large DICOM directories.
    """
    n = 0
    for p in dicom_dir.glob("MR.*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "SeriesInstanceUID", "")) == series_uid:
            n += 1
    return n


def contour_counts_by_roi_number(ds: pydicom.dataset.FileDataset) -> Dict[int, int]:
    out: Dict[int, int] = {}
    seq = getattr(ds, "ROIContourSequence", None)
    if not seq:
        return out
    for item in seq:
        rid = int(item.ReferencedROINumber)
        contours = getattr(item, "ContourSequence", None) or []
        out[rid] = len(contours)
    return out


def summarize_rtstruct(path: Path) -> Dict[str, Any]:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    contour_n = contour_counts_by_roi_number(ds)
    rois: List[Dict[str, Any]] = []
    for item in getattr(ds, "StructureSetROISequence", None) or []:
        num = int(item.ROINumber)
        name = str(item.ROIName)
        rois.append(
            {
                "roi_number": num,
                "roi_name": name,
                "contour_objects": contour_n.get(num, 0),
            }
        )
    rois.sort(key=lambda x: x["roi_number"])
    ref_series = referenced_series_uids_from_rtstruct(ds)
    return {
        "path": str(path),
        "modality": str(getattr(ds, "Modality", "")),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "structure_set_label": str(getattr(ds, "StructureSetLabel", "")),
        "referenced_series_instance_uids": ref_series,
        "roi_count": len(rois),
        "rois": rois,
    }


def summarize_rtplan(path: Path) -> Dict[str, Any]:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    ref_rs: List[str] = []
    for item in getattr(ds, "ReferencedStructureSetSequence", None) or []:
        uid = getattr(item, "ReferencedSOPInstanceUID", None)
        if uid:
            ref_rs.append(str(uid))
    beams = getattr(ds, "BeamSequence", None) or []
    fgs = getattr(ds, "FractionGroupSequence", None) or []
    return {
        "path": str(path),
        "modality": str(getattr(ds, "Modality", "")),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "rt_plan_label": str(getattr(ds, "RTPlanLabel", "")),
        "referenced_rtstruct_sop_instance_uids": ref_rs,
        "beam_count": len(beams),
        "fraction_group_count": len(fgs),
    }


def canonical_segmentation_organ(roi_name: str) -> Optional[str]:
    """Heuristic ROI-name → coarse mask organ bucket (not a semantic ontology).

    Matching is substring / exact-name heuristics only; it does not consult
    DICOM coded concepts or a controlled vocabulary.
    """
    n = roi_name.strip().lower()
    if n in ("bladder",):
        return "Bladder"
    if n in ("rectum",):
        return "Rectum"
    if n in ("urethra",):
        return "Urethra"
    if n in ("ptv",):
        return "PTV"
    if n in ("ptv1",):
        return "PTV1"
    if "ptv1" in n.replace(" ", ""):
        return "PTV1"
    if n.startswith("ptv"):
        return "PTV"
    if "bladder" in n:
        return "Bladder"
    if "rectum" in n:
        return "Rectum"
    if "urethra" in n:
        return "Urethra"
    return None


def mask_organ_summary(masks_root: Path) -> Dict[str, Any]:
    """Compatibility scan of historical mask layouts under ``masks_root``.

    Checks, in order of preference for *combined* counts:
    ``masks/combined_masks/``, flat ``masks/*_combined_mask.png``, and (for
    per-organ binaries) sibling ``masks_png/``. These are not all equally
    canonical — prefer ``masks/combined_masks/`` for training.
    """
    combined_dir = masks_root / "masks" / "combined_masks"
    flat_dir = masks_root / "masks"
    layout = "none"
    if combined_dir.is_dir():
        combined = sorted(combined_dir.glob("*_combined_mask.png"))
        layout = "masks/combined_masks"
    elif flat_dir.is_dir() and any(flat_dir.glob("*_combined_mask.png")):
        combined = sorted(flat_dir.glob("*_combined_mask.png"))
        layout = "masks_flat"
    else:
        combined = []

    per_organ: Dict[str, int] = {}
    organ_root = masks_root / "masks"
    if organ_root.is_dir():
        for sub in organ_root.iterdir():
            if sub.is_dir() and sub.name != "combined_masks":
                per_organ[sub.name] = len(list(sub.glob("*.png")))

    extra_organ: Optional[Dict[str, int]] = None
    extra_note: Optional[str] = None
    # Simulation: per-organ exports often live under Processing/masks_png/.
    alt = masks_root.parent / "masks_png"
    if not per_organ and alt.is_dir():
        extra_organ = {}
        for sub in alt.iterdir():
            if sub.is_dir():
                extra_organ[sub.name] = len(list(sub.glob("*.png")))
        if extra_organ:
            try:
                rel = alt.relative_to(masks_root.parent.parent)
            except ValueError:
                rel = alt
            extra_note = f"Compatibility scan of sibling directory: {rel}"

    return {
        "compatibility_scan": True,
        "combined_layout": layout,
        "combined_mask_png_count": len(combined),
        "per_organ_binary_png_counts": per_organ,
        "alternate_per_organ_png_counts": extra_organ,
        "alternate_per_organ_note": extra_note,
    }


def linkage_rs_to_masks(rs_summary: Dict[str, Any]) -> Dict[str, Any]:
    """ROIs in RTSTRUCT that correspond to exported mask organs (heuristic names)."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rs_summary["rois"]:
        org = canonical_segmentation_organ(r["roi_name"])
        if org is None:
            continue
        buckets.setdefault(org, []).append(
            {
                "roi_number": r["roi_number"],
                "roi_name": r["roi_name"],
                "contour_objects": r["contour_objects"],
            }
        )
    return {"segmentation_linked_rois": buckets}


def build_case_report(
    case_id: str,
    dicom_dir: Optional[Path],
    masks_root: Path,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "case_id": case_id,
        "dicom_dir": str(dicom_dir) if dicom_dir else None,
        "masks_root": str(masks_root),
        "rtstruct_path": None,
        "rtplan_path": None,
    }
    if dicom_dir is None or not dicom_dir.is_dir():
        report["error"] = "No DICOM directory with *.dcm found"
        return report

    rs_path, rp_path = find_rs_rp(dicom_dir)
    report["rtstruct_path"] = str(rs_path) if rs_path else None
    report["rtplan_path"] = str(rp_path) if rp_path else None
    if not rs_path:
        report["error"] = "No RS.*.dcm found"
        return report

    rs_sum = summarize_rtstruct(rs_path)
    report["rtstruct"] = rs_sum
    report["rtstruct_to_mask_organs"] = linkage_rs_to_masks(rs_sum)

    if rp_path:
        rp_sum = summarize_rtplan(rp_path)
        report["rtplan"] = rp_sum
        rs_uid = rs_sum["sop_instance_uid"]
        ref = rp_sum.get("referenced_rtstruct_sop_instance_uids") or []
        report["rtplan_references_rtstruct"] = rs_uid in ref
    else:
        report["rtplan"] = None
        report["rtplan_references_rtstruct"] = None

    series_uids = rs_sum.get("referenced_series_instance_uids") or []
    if series_uids:
        uid = series_uids[0]
        report["mr_slice_count_same_series_as_rtstruct"] = count_mr_slices_for_series(
            dicom_dir, uid
        )
        report["rtstruct_referenced_series_uid"] = uid
    else:
        report["mr_slice_count_same_series_as_rtstruct"] = None
        report["rtstruct_referenced_series_uid"] = None

    if masks_root.is_dir():
        report["disk_masks"] = mask_organ_summary(masks_root)
    else:
        report["disk_masks"] = None

    return report


def render_text(reports: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for rep in reports:
        cid = rep["case_id"]
        lines.append(f"{'=' * 72}\n{cid}\n{'=' * 72}")
        lines.append(f"DICOM dir: {rep.get('dicom_dir')}")
        lines.append(f"Masks root: {rep.get('masks_root')}")
        if rep.get("error"):
            lines.append(f"ERROR: {rep['error']}")
            lines.append("")
            continue
        rs = rep["rtstruct"]
        lines.append(f"RTSTRUCT: {Path(rs['path']).name}")
        lines.append(f"  Label: {rs['structure_set_label']}")
        lines.append(f"  SOP Instance UID: {rs['sop_instance_uid']}")
        lines.append(f"  Referenced MR SeriesInstanceUID: {rep.get('rtstruct_referenced_series_uid')}")
        lines.append(
            f"  MR slice DICOM count (MR.*.dcm, that series): "
            f"{rep.get('mr_slice_count_same_series_as_rtstruct')}"
        )
        lines.append(f"  ROI definitions (total): {rs['roi_count']}")

        link = rep.get("rtstruct_to_mask_organs", {}).get("segmentation_linked_rois", {})
        lines.append("  ROIs linked to segmentation mask organs (heuristic name match):")
        if not link:
            lines.append("    (none)")
        else:
            for organ, items in sorted(link.items()):
                lines.append(f"    [{organ}]")
                for it in items:
                    lines.append(
                        f"      #{it['roi_number']} {it['roi_name']!r} — "
                        f"{it['contour_objects']} contour object(s)"
                    )

        rp = rep.get("rtplan")
        if rp:
            lines.append(f"RTPLAN: {Path(rp['path']).name}")
            lines.append(f"  Label: {rp['rt_plan_label']}")
            lines.append(f"  Beams: {rp['beam_count']}")
            lines.append(f"  Fraction groups: {rp['fraction_group_count']}")
            lines.append(
                f"  Referenced RTSTRUCT SOP UID(s): "
                f"{', '.join(rp['referenced_rtstruct_sop_instance_uids'])}"
            )
            lines.append(
                f"  RTPLAN references this RTSTRUCT: {rep.get('rtplan_references_rtstruct')}"
            )
        else:
            lines.append("RTPLAN: (missing)")

        dm = rep.get("disk_masks")
        if dm:
            lines.append("On-disk masks (compatibility scan of processed dataset):")
            lines.append(
                f"  Combined layout: {dm.get('combined_layout')}  "
                f"count={dm['combined_mask_png_count']}"
            )
            if dm["per_organ_binary_png_counts"]:
                lines.append("  Per-organ binary PNG counts:")
                for k, v in sorted(dm["per_organ_binary_png_counts"].items()):
                    lines.append(f"    {k}: {v}")
            alt = dm.get("alternate_per_organ_png_counts")
            if alt:
                lines.append(f"  {dm.get('alternate_per_organ_note', 'Alternate per-organ counts:')}")
                for k, v in sorted(alt.items()):
                    lines.append(f"    {k}: {v}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize RS/RP DICOM and relate to mask folders."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <BAPMOS>/data/prostate/rt_rs_rp_mask_summary).",
    )
    args = p.parse_args(argv)

    out_default = project_root() / "data" / "prostate" / "rt_rs_rp_mask_summary"
    out_arg = args.out or out_default
    out_dir = out_arg.resolve() if out_arg.is_absolute() else (project_root() / out_arg).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for case_id in DEFAULT_CASE_IDS:
        dicom_dir = resolve_dicom_dir_for_case(case_id)
        masks_root = _masks_root_for_case(case_id)
        reports.append(build_case_report(case_id, dicom_dir, masks_root))

    json_path = out_dir / "summary.json"
    txt_path = out_dir / "summary.txt"
    json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(reports), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if any(r.get("error") for r in reports):
        print(
            "Note: one or more cases missing DICOM/RS (see summary). "
            f"Lookup root used for DICOM/masks: {_repo_root()}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
