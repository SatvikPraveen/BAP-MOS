"""
Batch preprocessing: MR DICOM → PNG under ``preprocessing/``, then RTSTRUCT → masks.

Maps each corpus to the layout expected by ``bapmos.paths`` dataset helpers:

- **case1** — MR+RS DICOM (``data/real_data/Case 1/Dicom`` or another path from
  ``bapmos.preprocess.prostate.rtstruct_dicom_roots``) →
  ``preprocessing/real_data/case1/case1_dicom_png/`` and
  ``preprocessing/real_data/case1/masks/`` (taxonomy **clinical**).
- **case2** — same pattern under Case 2 → ``preprocessing/real_data/case2/...`` (**clinical**).
- **simulation** — ``data/simulation_data/dicom`` (if present) → ``preprocessing/simulation_data/...``
  (**simulation**).

Stratified splits under ``splits_stratified/`` are never modified. Optional
``--remove-legacy-splits-dir`` deletes only ``<dataset_root>/splits`` if present.

This is the **canonical** prostate mask-export entry point. After masks exist, run
``python -m bapmos.preprocess.prostate.create_stratified_splits``, then build the pooled
corpus with ``python -m bapmos.preprocess.prostate`` — see ``docs/PREPROCESS.md``.

Examples::

    python -m bapmos.preprocess.prostate.run_rtstruct_masks --case case1
    python -m bapmos.preprocess.prostate.run_rtstruct_masks --case all --skip-dicom-png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from bapmos.paths import real_case_dataset_dir, simulation_dataset_dir
from bapmos.preprocess.prostate.convert_dicom_to_png import convert_directory
from bapmos.preprocess.prostate.rtstruct_dicom_roots import (
    candidate_dicom_dirs_for_case,
    resolve_dicom_dir_for_case,
)
from bapmos.preprocess.prostate.rtstruct_export_slice_masks import run_export


def _resolve_case(case: str) -> Tuple[str, Path, Path, Path, str]:
    """
    Returns (canonical_key, dataset_root, dicom_dir, png_out_dir, taxonomy).

    ``dicom_dir`` is the first existing path from ``candidate_dicom_dirs_for_case``
    that contains ``*.dcm`` (see ``bapmos.preprocess.prostate.rtstruct_dicom_roots``).
    """
    k = case.strip().lower().replace("-", "_")
    if k in ("case1", "case_1"):
        canon = "case1"
        dicom_dir = resolve_dicom_dir_for_case(canon)
        root = real_case_dataset_dir("case1")
        png = root / "case1_dicom_png"
        tax = "clinical"
    elif k in ("case2", "case_2"):
        canon = "case2"
        dicom_dir = resolve_dicom_dir_for_case(canon)
        root = real_case_dataset_dir("case2")
        png = root / "case2_dicom_png"
        tax = "clinical"
    elif k in ("simulation", "sim", "simulation_data"):
        canon = "simulation"
        dicom_dir = resolve_dicom_dir_for_case(canon)
        root = simulation_dataset_dir()
        png = root / "simulation_dicom_png"
        tax = "simulation"
    else:
        raise ValueError(f"Unknown --case {case!r}; use case1, case2, simulation, or all")

    if dicom_dir is None:
        tried = ", ".join(str(p) for p in candidate_dicom_dirs_for_case(canon))
        raise FileNotFoundError(f"No DICOM folder with *.dcm for {canon}. Tried: {tried}")

    return canon, root, dicom_dir, png, tax


def run_one(
    case: str,
    *,
    skip_dicom_png: bool,
    overwrite_png: bool,
    remove_legacy_splits_dir: bool,
    dry_run: bool,
    rtstruct: Optional[Path],
    structures: Optional[str],
    slice_organ_presence_json: Optional[Path],
    use_slice_organ_presence_json: bool,
) -> None:
    _key, dataset_root, dicom_dir, png_dir, taxonomy = _resolve_case(case)
    if not skip_dicom_png:
        if dry_run:
            print(f"[dry-run] would convert DICOM {dicom_dir} -> {png_dir}")
        else:
            w, sk, fail, failures = convert_directory(
                dicom_dir.resolve(),
                png_dir.resolve(),
                overwrite=overwrite_png,
            )
            print(f"DICOM→PNG ({case}): {w} written, {sk} skipped (blank), {fail} fail -> {png_dir}")
            for line in failures[:20]:
                print(line, file=sys.stderr)
            if fail and w == 0 and sk == 0:
                raise RuntimeError(f"DICOM conversion produced no PNGs for {case}")

    filt = [s.strip() for s in structures.split(",")] if structures else None
    run_export(
        dicom_dir=dicom_dir,
        masks_root=dataset_root / "masks",
        taxonomy=taxonomy,
        rtstruct_path=rtstruct,
        structures_filter=filt,
        remove_legacy_splits_dir=remove_legacy_splits_dir,
        dry_run=dry_run,
        training_png_dir=png_dir,
        slice_organ_presence_json=slice_organ_presence_json,
        use_slice_organ_presence_json=use_slice_organ_presence_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, help="case1 | case2 | simulation | all")
    p.add_argument("--skip-dicom-png", action="store_true", help="Only run RTSTRUCT mask export")
    p.add_argument("--overwrite-png", action="store_true", help="Replace existing PNGs when converting")
    p.add_argument(
        "--remove-legacy-splits-dir",
        action="store_true",
        help="Delete dataset_root/splits if it exists (never touches splits_stratified/)",
    )
    p.add_argument("--rtstruct", type=Path, default=None, help="Explicit RS*.dcm path")
    p.add_argument(
        "--structures",
        type=str,
        default=None,
        help="Comma-separated exact ROINames to export (ROIName strings as they appear in the RTSTRUCT)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--slice-organ-json",
        type=Path,
        default=None,
        help="Optional slice_organ_presence/*.json (default: auto under preprocessing/).",
    )
    p.add_argument(
        "--no-slice-organ-json",
        action="store_true",
        help="Do not intersect stems with slice_organ_presence JSON.",
    )
    args = p.parse_args(argv)

    raw = args.case.strip().lower()
    if raw == "all":
        cases = ["case1", "case2", "simulation"]
    else:
        cases = [args.case]

    for c in cases:
        run_one(
            c,
            skip_dicom_png=args.skip_dicom_png,
            overwrite_png=args.overwrite_png,
            remove_legacy_splits_dir=args.remove_legacy_splits_dir,
            dry_run=args.dry_run,
            rtstruct=args.rtstruct,
            structures=args.structures,
            slice_organ_presence_json=args.slice_organ_json,
            use_slice_organ_presence_json=not args.no_slice_organ_json,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
