"""
Per-slice organ presence from combined multiclass masks.

For **every** slice PNG under ``images/`` or ``case*_dicom_png/`` (same enumeration as
``find_training_images_dir``), reads the matching ``<stem>_combined_mask.png`` and records
which foreground organs appear (same class ids as stratified splits).

**Where masks are found (in order)**

1. ``--combined-masks-dir`` if you pass it (must contain ``*_combined_mask.png``).
   Affects **only this report** — not dataset configuration or split generation.
2. ``<dataset_root>/masks/...`` via ``bapmos.paths.find_combined_masks_dir`` and a shallow search.
3. **Fallback:** ``test_output/<any_bundle>/<case>/masks/combined_masks/`` and the same under
   optional parent-layout ``preprocessing/<any_bundle>/…`` (whichever folder has the most
   ``*_combined_mask.png`` files). This covers RTSTRUCT exports when canonical ``masks/`` is empty.

The chosen directory is written into the JSON report — use this for **audit**, not as the
silent source of truth for training. Training and final splits require canonical
``<dataset_root>/masks/combined_masks/``.

CSV column ``multiclass_status`` is ``read`` | ``missing_file`` | ``no_mask_dir``.
Organ indicator columns are always present in the CSV but are **meaningful only** when
``multiclass_status == "read"`` (otherwise they are all zero).

Report case IDs use the package underscore form (``simulation``, ``case_1``, ``case_2``).
On-disk fallback folders under ``test_output`` / ``preprocessing`` still use ``case1`` /
``case2`` directory names.

Usage (from BAPMOS checkout root)::

    python -m bapmos.preprocess.delineation.report_slice_organ_presence --all \\
        --out data/prostate/slice_organ_presence

    python -m bapmos.preprocess.delineation.report_slice_organ_presence \\
        --dataset /path/to/case_1 \\
        --out data/prostate/slice_organ_presence
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from bapmos.paths import find_combined_masks_dir as find_combined_masks_dir_core
from bapmos.paths import (
    dataset_bundle_tag,
    preprocessing_root,
    project_root,
    real_case_dataset_dir,
    research_tree_root,
    simulation_dataset_dir,
)

from bapmos.preprocess.prostate.create_stratified_splits import (
    REAL_CLINICAL,
    SIMULATION,
    OrganTaxonomy,
    combined_mask_path,
    detect_taxonomy,
    list_images,
    organ_presence,
    pattern_label,
)

MISSING_COMBINED: str = "Missing_combined_mask"
NO_MASK_DIR: str = "No_combined_masks_directory"
STATUS_NO_MASK_DIR: str = "no_mask_dir"


def _repo_root() -> Path:
    """Optional parent layout for preprocessing corpora; else BAPMOS checkout."""
    return research_tree_root() or project_root()


def _default_dataset_jobs() -> List[Tuple[str, Path]]:
    return [
        ("simulation", simulation_dataset_dir()),
        ("case_1", real_case_dataset_dir("case1")),
        ("case_2", real_case_dataset_dir("case2")),
    ]


def _infer_taxonomy_when_no_mask_dir(dataset_root: Path) -> OrganTaxonomy:
    """Clinical vs simulation when ``find_combined_masks_dir`` cannot run yet."""
    tag = dataset_bundle_tag(dataset_root)
    if tag == "simulation":
        return SIMULATION
    root = dataset_root.resolve()
    if root.name in ("simulation_data", "simulation") or "simulation_data" in root.parts:
        return SIMULATION
    return REAL_CLINICAL


def _case_key_for_test_output_fallback(dataset_root: Path) -> Optional[str]:
    """Map dataset root to folder name under ``test_output/<bundle>/`` (case1/case2)."""
    tag = dataset_bundle_tag(dataset_root)
    if tag == "simulation":
        return "simulation"
    if tag == "case_1":
        return "case1"
    if tag == "case_2":
        return "case2"
    # Legacy path fragments when bundle tag is ambiguous.
    s = str(dataset_root.resolve()).replace("\\", "/")
    if "simulation_data" in s or s.rstrip("/").endswith("/simulation"):
        return "simulation"
    if "/case_1" in s or s.rstrip("/").endswith("case_1") or "/case1" in s or s.rstrip("/").endswith("case1"):
        return "case1"
    if "/case_2" in s or s.rstrip("/").endswith("case_2") or "/case2" in s or s.rstrip("/").endswith("case2"):
        return "case2"
    return None


def _public_case_id(dataset_root: Path, fallback_name: Optional[str] = None) -> str:
    """Stable report id: simulation | case_1 | case_2 | directory name."""
    tag = dataset_bundle_tag(dataset_root)
    if tag in ("simulation", "case_1", "case_2"):
        return tag
    name = fallback_name or dataset_root.name
    key = name.lower().replace("-", "_")
    if key in ("case1", "case_1"):
        return "case_1"
    if key in ("case2", "case_2"):
        return "case_2"
    if key in ("simulation", "simulation_data"):
        return "simulation"
    return name


def _rtstruct_case_cli(case_id: str) -> str:
    """``run_rtstruct_masks --case`` still expects case1/case2 (no underscore)."""
    return {"case_1": "case1", "case_2": "case2"}.get(case_id, case_id)


def _collect_candidate_combined_dirs(dataset_root: Path, repo_root: Path) -> List[Path]:
    """All plausible directories that might contain ``*_combined_mask.png``."""
    found: List[Path] = []
    try:
        found.append(find_combined_masks_dir_core(dataset_root))
    except FileNotFoundError:
        pass

    masks = dataset_root / "masks"
    if masks.is_dir():
        by_parent: Dict[Path, int] = defaultdict(int)
        for p in masks.rglob("*_combined_mask.png"):
            parts_lower = {x.lower() for x in p.parts}
            if "combined_mask_preview" in parts_lower:
                continue
            by_parent[p.parent] += 1
        found.extend(by_parent.keys())

    key = _case_key_for_test_output_fallback(dataset_root)
    test_out = repo_root / "test_output"
    if key and test_out.is_dir():
        try:
            for bundle in sorted(test_out.iterdir()):
                if not bundle.is_dir():
                    continue
                cand = bundle / key / "masks" / "combined_masks"
                if cand.is_dir():
                    found.append(cand)
        except OSError:
            pass

    prep = preprocessing_root()
    if key and prep.is_dir():
        try:
            for bundle in sorted(prep.iterdir()):
                if not bundle.is_dir():
                    continue
                cand = bundle / key / "masks" / "combined_masks"
                if cand.is_dir():
                    found.append(cand)
        except OSError:
            pass

    # de-dupe by resolved path
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in found:
        try:
            r = p.resolve()
        except OSError:
            r = p
        k = str(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def _count_combined_masks(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob("*_combined_mask.png"))


def _pick_combined_masks_dir(candidates: List[Path]) -> Optional[Path]:
    """Prefer the directory with the most ``*_combined_mask.png`` files."""
    best: Optional[Path] = None
    best_n = 0
    for c in candidates:
        n = _count_combined_masks(c)
        if n > best_n:
            best_n = n
            best = c
    return best if best_n > 0 else None


def discover_combined_masks_dir(
    dataset_root: Path,
    repo_root: Path,
    *,
    override: Optional[Path] = None,
) -> Tuple[Optional[Path], str, List[str]]:
    """
    Resolve folder containing training ``*_combined_mask.png``.

    Returns ``(chosen_dir_or_None, selection_source, candidate_paths)``.

    ``selection_source`` is ``override`` | ``canonical`` | ``fallback`` | ``none``.

    Prefer the canonical ``find_combined_masks_dir(dataset_root)`` result whenever it
    exists (including when that path is a symlink whose ``.resolve()`` target lies
    outside the dataset root). Only if canonical lookup fails do we fall back to the
    largest candidate pool under dataset ``masks/**``, ``test_output``, and
    ``preprocessing`` (not recency). Override affects **only this report**.
    """
    if override is not None:
        p = override.expanduser().resolve()
        if p.is_dir() and any(p.glob("*_combined_mask.png")):
            return p, "override", [str(p)]
        return None, "none", [str(p)]

    candidates = _collect_candidate_combined_dirs(dataset_root, repo_root)
    cand_strs = [str(c) for c in candidates]

    try:
        canonical = find_combined_masks_dir_core(dataset_root)
        # Keep the path as returned (may be a symlink); resolve only for existence checks.
        if canonical.is_dir() and any(canonical.glob("*_combined_mask.png")):
            return canonical, "canonical", cand_strs
    except FileNotFoundError:
        pass

    chosen = _pick_combined_masks_dir(candidates)
    if chosen is None:
        return None, "none", cand_strs
    return chosen, "fallback", cand_strs


def load_combined_label_array(path: Path) -> np.ndarray:
    """2D integer class map; coerce to grayscale ``L`` for predictable modes."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.int64)
    return arr


def _path_for_index(path: Path, root: Path) -> str:
    """Prefer path relative to ``root``; fall back to absolute when outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _warn_no_multiclass_masks(case_id: str, dataset_root: Path, meta: Dict[str, Any]) -> None:
    if meta.get("slice_count", 0) == 0:
        return
    if meta.get("slices_with_combined_file_analyzed", 0) > 0:
        return
    cli_case = _rtstruct_case_cli(case_id)
    print(
        f"\n*** WARNING [{case_id}]: No multiclass *_combined_mask.png was read under\n"
        f"    {dataset_root.resolve()}\n"
        "    CSV organ columns will be all 0; see column 'multiclass_status' "
        f"({NO_MASK_DIR!r} / {MISSING_COMBINED!r}).\n"
        "    Fix: run mask export, e.g.\n"
        f"      python -m bapmos.preprocess.prostate.run_rtstruct_masks --case {cli_case}\n"
        "    This report also scans test_output/*/…/combined_masks/ and preprocessing/*/…/combined_masks/ "
        "and uses the largest pool.\n"
        "    Override path:  --combined-masks-dir /path/to/combined_masks\n",
        file=sys.stderr,
    )


def analyze_dataset(
    dataset_root: Path,
    *,
    repo_root: Path,
    combined_masks_override: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    images = list_images(dataset_root)
    mask_dir, selection_source, candidate_dirs = discover_combined_masks_dir(
        dataset_root, repo_root, override=combined_masks_override
    )
    if mask_dir is not None:
        taxonomy = detect_taxonomy(mask_dir)
    else:
        taxonomy = _infer_taxonomy_when_no_mask_dir(dataset_root)

    rows: List[Dict[str, Any]] = []
    missing_mask: List[str] = []

    for img_path in images:
        fname = img_path.name
        if mask_dir is None:
            rows.append(
                {
                    "slice": fname,
                    "multiclass_status": STATUS_NO_MASK_DIR,
                    "pattern": NO_MASK_DIR,
                    "organs_present": [],
                }
            )
            continue
        mp = combined_mask_path(img_path, mask_dir)
        if not mp.is_file():
            missing_mask.append(fname)
            rows.append(
                {
                    "slice": fname,
                    "multiclass_status": "missing_file",
                    "pattern": MISSING_COMBINED,
                    "organs_present": [],
                }
            )
            continue
        mask = load_combined_label_array(mp)
        pat = organ_presence(mask, taxonomy)
        # ``pat`` aligns 1:1 with ``taxonomy.display_names`` by OrganTaxonomy contract.
        present = [taxonomy.display_names[i] for i, ok in enumerate(pat) if ok]
        rows.append(
            {
                "slice": fname,
                "multiclass_status": "read",
                "pattern": pattern_label(pat, taxonomy),
                "organs_present": present,
            }
        )

    n_analyzed = sum(1 for r in rows if r["multiclass_status"] == "read")
    n_bg = sum(1 for r in rows if r["pattern"] == "Background")
    meta = {
        "dataset_root": str(dataset_root),
        "taxonomy": taxonomy.name,
        "organ_columns": list(taxonomy.display_names),
        "combined_masks_directory": str(mask_dir.resolve()) if mask_dir is not None else None,
        "combined_masks_selection_source": selection_source,
        "combined_masks_selection_rule": (
            "explicit_override"
            if selection_source == "override"
            else "max_number_of_combined_mask_png_files"
        ),
        "combined_masks_candidate_directories": candidate_dirs,
        "using_mask_dir_fallback_outside_dataset": selection_source == "fallback",
        "explicit_combined_masks_dir": str(combined_masks_override.resolve())
        if combined_masks_override is not None
        else None,
        "total_slice_pngs": len(images),
        "slices_with_combined_file_analyzed": n_analyzed,
        "background_only_slice_count": n_bg,
        "missing_combined_file_count": len(missing_mask),
        "slice_count": len(images),
        "missing_masks": missing_mask,
    }
    return rows, meta


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    organ_columns: List[str],
) -> None:
    fieldnames = ["slice", "multiclass_status", "pattern"] + organ_columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            pat = r["organs_present"]
            out = {
                "slice": r["slice"],
                "multiclass_status": r["multiclass_status"],
                "pattern": r["pattern"],
            }
            for name in organ_columns:
                out[name] = 1 if name in pat else 0
            w.writerow(out)


def write_text_summary(
    path: Path,
    case_id: str,
    meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"{case_id} — organ presence per slice")
    lines.append("=" * 72)
    lines.append(f"Dataset: {meta['dataset_root']}")
    lines.append(f"Taxonomy: {meta['taxonomy']} ({', '.join(meta['organ_columns'])})")
    lines.append(f"Combined masks directory: {meta.get('combined_masks_directory') or '(none found)'}")
    lines.append(
        f"  selection: source={meta.get('combined_masks_selection_source')}  "
        f"rule={meta.get('combined_masks_selection_rule')}"
    )
    if meta.get("using_mask_dir_fallback_outside_dataset"):
        lines.append(
            "  (masks were taken from outside the dataset root, e.g. test_output/…/combined_masks)"
        )
    lines.append(f"Total slice PNGs listed: {meta['slice_count']}")
    lines.append(
        f"  With combined mask read: {meta['slices_with_combined_file_analyzed']}  "
        f"(background-only in mask: {meta['background_only_slice_count']})  "
        f"missing combined file: {meta['missing_combined_file_count']}"
    )
    if meta["missing_masks"]:
        lines.append(f"Missing combined masks ({len(meta['missing_masks'])}): ")
        for m in meta["missing_masks"][:10]:
            lines.append(f"  {m}")
        if len(meta["missing_masks"]) > 10:
            lines.append("  ...")
    lines.append("")
    ctr = Counter(r["pattern"] for r in rows)
    lines.append("Pattern histogram (one row per slice PNG):")
    for pat, n in ctr.most_common():
        lines.append(f"  {n:4d}  {pat}")
    lines.append("")
    lines.append("Per slice (slice filename → organs present):")
    for r in rows:
        if r["pattern"] == MISSING_COMBINED:
            orgs = "(no matching *_combined_mask.png)"
        elif r["pattern"] == NO_MASK_DIR:
            orgs = "(no combined_masks directory with *_combined_mask.png yet)"
        else:
            orgs = ", ".join(r["organs_present"]) if r["organs_present"] else "(background only)"
        lines.append(f"  {r['slice']}")
        lines.append(f"    → {orgs}  [{r['pattern']}]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.delineation.report_slice_organ_presence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dataset",
        type=Path,
        help="Dataset root (contains images/ and masks/).",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Run simulation, case_1, and case_2 (via bapmos.paths dataset helpers).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=project_root() / "data" / "prostate" / "slice_organ_presence",
        help="Output directory (created).",
    )
    p.add_argument(
        "--combined-masks-dir",
        type=Path,
        default=None,
        help=(
            "Explicit folder containing *_combined_mask.png (with --dataset only). "
            "Affects only this report, not dataset config or splits."
        ),
    )
    args = p.parse_args(argv)
    if args.all and args.combined_masks_dir is not None:
        print("error: --combined-masks-dir cannot be used with --all", file=sys.stderr)
        return 2

    combined_override: Optional[Path] = None
    if args.combined_masks_dir is not None:
        raw = args.combined_masks_dir.expanduser()
        cm = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not cm.is_dir() or not any(cm.glob("*_combined_mask.png")):
            print(
                f"error: --combined-masks-dir must be a directory with *_combined_mask.png files: {cm}",
                file=sys.stderr,
            )
            return 2
        combined_override = cm

    out_dir = args.out if args.out.is_absolute() else (project_root() / args.out)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, Path]] = []
    if args.all:
        jobs = _default_dataset_jobs()
    else:
        dr = args.dataset.expanduser()
        dataset_root = dr.resolve() if dr.is_absolute() else (root / dr).resolve()
        jobs.append((_public_case_id(dataset_root, dataset_root.name), dataset_root))

    index_root = project_root()
    index: List[Dict[str, str]] = []
    for case_id, dataset_root in jobs:
        if not dataset_root.is_dir():
            print(f"Skip missing directory: {dataset_root}", file=sys.stderr)
            continue
        rows, meta = analyze_dataset(
            dataset_root,
            repo_root=root,
            combined_masks_override=combined_override,
        )
        meta["case_id"] = case_id

        prefix = out_dir / case_id
        write_csv(prefix.with_suffix(".csv"), rows, meta["organ_columns"])
        (prefix.with_suffix(".json")).write_text(
            json.dumps({"meta": meta, "slices": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_text_summary(prefix.with_suffix(".txt"), case_id, meta, rows)
        _warn_no_multiclass_masks(case_id, dataset_root, meta)
        index.append(
            {
                "case_id": case_id,
                "csv": _path_for_index(prefix.with_suffix(".csv"), index_root),
                "json": _path_for_index(prefix.with_suffix(".json"), index_root),
                "txt": _path_for_index(prefix.with_suffix(".txt"), index_root),
            }
        )
        print(
            f"{case_id}: {meta['slice_count']} slice PNGs in report "
            f"({meta['slices_with_combined_file_analyzed']} multiclass masks read, "
            f"{meta['missing_combined_file_count']} missing combined file) → {prefix}.csv"
        )

    (out_dir / "index.json").write_text(
        json.dumps({"outputs": index}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
