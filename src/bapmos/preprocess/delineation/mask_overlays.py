"""
QA mask overlays on slice PNGs — **internal datasets only** (simulation, Case 1, Case 2).

PFUS1 overlays: ``python -m bapmos.preprocess.bladder.visualize_samples``.

Reads images and combined multiclass masks from each dataset root; does not
modify source files. Writes RGB overlay PNGs under ``data/prostate/qa_overlays/``
(override with ``--out-root``).

Usage::

    python -m bapmos.preprocess.delineation.mask_overlays --dataset simulation --splits-subdir splits_stratified
    python -m bapmos.preprocess.delineation.mask_overlays --dataset case_1 --splits-subdir splits_stratified
    python -m bapmos.preprocess.delineation.mask_overlays --dataset case_2 --splits-subdir splits_stratified
    python -m bapmos.preprocess.delineation.mask_overlays --all --splits-subdir splits_stratified --seed 42
    python -m bapmos.preprocess.delineation.mask_overlays --dataset case_1 --out-root /tmp/qa_overlays

By default, draws a compact top-left label box (organ pattern and per-structure
area % in mask colors). Slice id, dataset/split, and percentage definitions are
omitted so you can state them in captions or the paper. Use
``--omit-slice-label-box`` to disable. PTV / PTV1 are shown as ``Prostate/PTV``.
Optional ``--include-legend`` adds a bottom legend strip.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bapmos.data.organ_registry import (
    REAL_CLINICAL_ORGANS,
    SIMULATION_THREE_ORGANS,
)
from bapmos.paths import (
    find_combined_masks_dir,
    find_training_images_dir,
    project_root,
)
from bapmos.preprocess.prostate.create_stratified_splits import (
    STRATIFICATION_SEED,
    detect_taxonomy,
    resolve_dataset_arg,
)

REPO_ROOT = project_root()
DEFAULT_QA_ROOT = REPO_ROOT / "data" / "prostate" / "qa_overlays"

DATASET_IDS = ("simulation", "case_1", "case_2")


@dataclass
class SplitReport:
    requested: int = 0
    written: int = 0
    missing_image: List[str] = field(default_factory=list)
    missing_mask: List[str] = field(default_factory=list)
    class_presence_count_by_slice: Counter = field(default_factory=Counter)


@dataclass
class DatasetReport:
    dataset_id: str
    dataset_root: str
    splits_subdir: str
    output_dir: str
    taxonomy: str
    splits: Dict[str, SplitReport] = field(default_factory=dict)
    excluded_written: int = 0
    excluded_missing: List[str] = field(default_factory=list)
    passed: bool = True


def _bgr_to_rgb(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bgr
    return (int(r), int(g), int(b))


def _qa_overlay_display_name(taxonomy_name: str, class_id: int) -> str:
    """Human-facing organ name on QA overlays (PTV / PTV1 → Prostate/PTV)."""
    for o in _organ_defs(taxonomy_name):
        if o.class_id == class_id:
            if o.key in ("ptv", "ptv1"):
                return "Prostate/PTV"
            return o.evaluator_label
    return f"class_{class_id}"


def _organ_defs(taxonomy_name: str) -> Sequence:
    return SIMULATION_THREE_ORGANS if taxonomy_name == "simulation" else REAL_CLINICAL_ORGANS


def _color_map(taxonomy_name: str) -> Dict[int, Tuple[int, int, int]]:
    out: Dict[int, Tuple[int, int, int]] = {}
    for o in _organ_defs(taxonomy_name):
        out[o.class_id] = _bgr_to_rgb(o.color_bgr)
    return out


def _label_map(taxonomy_name: str) -> Dict[int, str]:
    return {o.class_id: _qa_overlay_display_name(taxonomy_name, o.class_id) for o in _organ_defs(taxonomy_name)}


def add_micca_style_slice_label(
    ov_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    dataset_id: str,
    split: str,
    slice_filename: str,
    taxonomy_name: str,
) -> np.ndarray:
    """
    Top-left black label box: multi-organ pattern and per-structure area %
    (full-mask pixels / image pixels).

    ``dataset_id``, ``split``, and ``slice_filename`` are accepted so call sites
    stay stable; they are not drawn (describe in the figure caption or paper).
    """
    _ = (dataset_id, split, slice_filename)
    pil = Image.fromarray(ov_rgb.copy())
    draw = ImageDraw.Draw(pil)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font_title = font_body = ImageFont.load_default()

    h, w = ov_rgb.shape[:2]
    m = mask.astype(np.int32)
    n_pix = float(h * w)
    color_map = _color_map(taxonomy_name)

    present: List[Tuple[int, str, Tuple[int, int, int], float]] = []
    for cid in sorted(color_map.keys()):
        count = int((m == cid).sum())
        pct = 100.0 * count / n_pix if n_pix else 0.0
        if count > 0:
            present.append((cid, _qa_overlay_display_name(taxonomy_name, cid), color_map[cid], pct))

    if present:
        pattern = " + ".join(p[1] for p in present)
    else:
        pattern = "Background"

    line_specs: List[Tuple[str, Tuple[int, int, int], Any]] = [
        (f"Pattern: {pattern}", (255, 255, 255), font_title),
    ]
    for _cid, disp, rgb, pct in present:
        line_specs.append((f"{disp}: {pct:.2f}%", rgb, font_body))

    margin = 8
    line_gap = 3
    max_w = 0
    heights: List[int] = []
    for text, _fill, font in line_specs:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        max_w = max(max_w, tw)
        heights.append(th)

    box_w = min(max(int(max_w + 2 * margin), 120), w - 4)
    box_h = int(sum(heights) + line_gap * (len(heights) - 1) + 2 * margin)
    box_h = min(box_h, h - 4)

    overlay_layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 235))
    pil.paste(overlay_layer, (4, 4), overlay_layer)
    draw = ImageDraw.Draw(pil)

    y = 4 + margin
    x0 = 4 + margin
    for (text, fill, font), th in zip(line_specs, heights):
        draw.text((x0, y), text, fill=fill, font=font)
        y += th + line_gap

    return np.asarray(pil.convert("RGB"))


def _boundary_mask(bin_fg: np.ndarray, k: int = 3) -> np.ndarray:
    if not bin_fg.any():
        return bin_fg.astype(bool)
    u = bin_fg.astype(np.uint8)
    ker = np.ones((k, k), np.uint8)
    er = cv2.erode(u, ker)
    bd = (u - er).clip(0, 1).astype(bool)
    return bd


def build_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    taxonomy_name: str,
    alpha: float,
    boundary_only: bool,
) -> np.ndarray:
    """Return uint8 RGB overlay (H, W, 3)."""
    h, w = image_rgb.shape[:2]
    color_map = _color_map(taxonomy_name)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    m = mask.astype(np.int32)
    for cid, rgb in color_map.items():
        fg = m == cid
        if not fg.any():
            continue
        if boundary_only:
            fg = _boundary_mask(fg)
        for c in range(3):
            overlay[:, :, c] += fg.astype(np.float32) * rgb[c]
    base = image_rgb.astype(np.float32)
    mask_any = np.zeros((h, w), dtype=bool)
    for cid in color_map:
        t = m == cid
        if boundary_only:
            t = _boundary_mask(t)
        mask_any |= t
    a = float(np.clip(alpha, 0.0, 1.0))
    out = base.copy()
    out[mask_any] = (1.0 - a) * base[mask_any] + a * overlay[mask_any]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _add_legend_strip(
    img_rgb: np.ndarray,
    taxonomy_name: str,
    include: bool,
) -> np.ndarray:
    if not include:
        return img_rgb
    h, w = img_rgb.shape[:2]
    lh = min(36 + 18 * len(_organ_defs(taxonomy_name)), h // 3)
    strip = np.ones((lh, w, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(strip)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    color_map = _color_map(taxonomy_name)
    label_map = _label_map(taxonomy_name)
    x = 6
    y = 4
    for cid in sorted(color_map.keys()):
        rgb = color_map[cid]
        draw.rectangle([x, y, x + 12, y + 12], fill=rgb, outline=(0, 0, 0))
        draw.text((x + 16, y), label_map[cid], fill=(0, 0, 0), font=font)
        y += 18
        if y > lh - 16:
            y = 4
            x += min(200, w // 2)
    strip = np.asarray(pil)
    return np.vstack([img_rgb, strip])


def combined_mask_path(mask_dir: Path, stem: str) -> Path:
    return mask_dir / f"{stem}_combined_mask.png"


def load_split_list(splits_dir: Path, split: str) -> List[str]:
    p = splits_dir / f"{split}.txt"
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def process_dataset(
    dataset_id: str,
    dataset_root: Path,
    splits_subdir: str,
    alpha: float,
    boundary_only: bool,
    include_legend: bool,
    include_slice_label_box: bool,
    max_per_split: Optional[int],
    seed: int,
    also_render_excluded: bool,
    *,
    out_root: Path,
) -> DatasetReport:
    rng = random.Random(seed)
    splits_dir = dataset_root / splits_subdir
    out_base = out_root / dataset_id
    out_base.mkdir(parents=True, exist_ok=True)

    try:
        mask_dir = find_combined_masks_dir(dataset_root)
        taxonomy = detect_taxonomy(mask_dir)
        taxonomy_name = taxonomy.name
    except (FileNotFoundError, RuntimeError) as exc:
        rep = DatasetReport(
            dataset_id=dataset_id,
            dataset_root=str(dataset_root),
            splits_subdir=splits_subdir,
            output_dir=str(out_base),
            taxonomy="unknown",
            passed=False,
        )
        print(f"FAIL [{dataset_id}]: cannot resolve masks/taxonomy under {dataset_root}: {exc}", file=sys.stderr)
        return rep

    try:
        images_dir = find_training_images_dir(dataset_root)
    except FileNotFoundError as exc:
        rep = DatasetReport(
            dataset_id=dataset_id,
            dataset_root=str(dataset_root),
            splits_subdir=splits_subdir,
            output_dir=str(out_base),
            taxonomy=taxonomy_name,
            passed=False,
        )
        print(f"FAIL [{dataset_id}]: {exc}", file=sys.stderr)
        return rep

    rep = DatasetReport(
        dataset_id=dataset_id,
        dataset_root=str(dataset_root),
        splits_subdir=splits_subdir,
        output_dir=str(out_base),
        taxonomy=taxonomy_name,
    )

    for split in ("train", "val", "test"):
        names = load_split_list(splits_dir, split)
        sr = SplitReport(requested=len(names))
        if max_per_split is not None and len(names) > max_per_split:
            names = names.copy()
            rng.shuffle(names)
            names = names[:max_per_split]
        rep.splits[split] = sr

        split_out = out_base / split
        split_out.mkdir(parents=True, exist_ok=True)

        for fname in names:
            stem = Path(fname).stem
            img_path = images_dir / Path(fname).name
            mp = combined_mask_path(mask_dir, stem)
            if not img_path.is_file():
                sr.missing_image.append(fname)
                rep.passed = False
                continue
            if not mp.is_file():
                sr.missing_mask.append(fname)
                rep.passed = False
                continue

            bgr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None:
                sr.missing_image.append(fname)
                rep.passed = False
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            mask = np.asarray(Image.open(mp).convert("L"))
            for v in np.unique(mask):
                if v > 0:
                    sr.class_presence_count_by_slice[int(v)] += 1

            ov = build_overlay(rgb, mask, taxonomy_name, alpha, boundary_only)
            if include_slice_label_box:
                ov = add_micca_style_slice_label(
                    ov,
                    mask,
                    dataset_id=dataset_id,
                    split=split,
                    slice_filename=fname,
                    taxonomy_name=taxonomy_name,
                )
            ov = _add_legend_strip(ov, taxonomy_name, include_legend)
            out_path = split_out / Path(fname).name
            Image.fromarray(ov).save(out_path)
            sr.written += 1

    if also_render_excluded:
        ss_path = splits_dir / "split_summary.json"
        extra_dir = out_base / "excluded_background_only"
        extra_dir.mkdir(parents=True, exist_ok=True)
        bg_list: List[str] = []
        if ss_path.is_file():
            try:
                bg_list = json.loads(ss_path.read_text(encoding="utf-8")).get(
                    "background_only_list", []
                )
            except json.JSONDecodeError:
                pass
        for fname in bg_list:
            stem = Path(fname).stem
            img_path = images_dir / Path(fname).name
            mp = combined_mask_path(mask_dir, stem)
            if not img_path.is_file() or not mp.is_file():
                rep.excluded_missing.append(fname)
                rep.passed = False
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None:
                rep.excluded_missing.append(fname)
                rep.passed = False
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            mask = np.asarray(Image.open(mp).convert("L"))
            ov = build_overlay(rgb, mask, taxonomy_name, alpha, boundary_only)
            if include_slice_label_box:
                ov = add_micca_style_slice_label(
                    ov,
                    mask,
                    dataset_id=dataset_id,
                    split="excluded_background_only",
                    slice_filename=fname,
                    taxonomy_name=taxonomy_name,
                )
            ov = _add_legend_strip(ov, taxonomy_name, include_legend)
            Image.fromarray(ov).save(extra_dir / Path(fname).name)
            rep.excluded_written += 1

    return rep


def write_summaries(
    reports: List[DatasetReport],
    seed: int,
    args_ns: argparse.Namespace,
    *,
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "stratification_seed_protocol": STRATIFICATION_SEED,
        "alpha": args_ns.alpha,
        "boundary_only": args_ns.boundary_only,
        "include_legend": args_ns.include_legend,
        "include_slice_label_box": args_ns.include_slice_label_box,
        "max_per_split": args_ns.max_per_split,
        "also_render_excluded": args_ns.also_render_excluded,
        "output_root": str(out_root),
        "summary_json_paths": [str(out_root / "overlay_summary.json")],
        "datasets": [],
        "overall_passed": (len(reports) > 0 and all(r.passed for r in reports)),
    }
    lines: List[str] = []
    lines.append("QA MASK OVERLAY SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append(f"Seed: {seed}  (protocol STRATIFICATION_SEED={STRATIFICATION_SEED})")
    lines.append(f"Overlay output root: {out_root}")
    lines.append(f"Summary files: {out_root / 'overlay_summary.json'}")
    lines.append("")

    for r in reports:
        dsum: Dict[str, Any] = {
            "dataset_id": r.dataset_id,
            "dataset_root": r.dataset_root,
            "splits_subdir": r.splits_subdir,
            "output_dir": r.output_dir,
            "taxonomy": r.taxonomy,
            "passed": r.passed,
            "splits": {},
            "excluded_background_only": {
                "written": r.excluded_written,
                "missing": r.excluded_missing,
            },
        }
        lines.append(f"Dataset: {r.dataset_id}  taxonomy={r.taxonomy}  PASS={r.passed}")
        lines.append(f"  Root: {r.dataset_root}")
        lines.append(f"  Out:  {r.output_dir}")
        for sp, sr in r.splits.items():
            dsum["splits"][sp] = {
                "requested": sr.requested,
                "written": sr.written,
                "missing_image": sr.missing_image,
                "missing_mask": sr.missing_mask,
                "class_presence_count_by_slice": dict(sr.class_presence_count_by_slice),
            }
            lines.append(
                f"  [{sp}] requested={sr.requested} written={sr.written} "
                f"missing_img={len(sr.missing_image)} missing_mask={len(sr.missing_mask)}"
            )
            if sr.missing_image:
                for m in sr.missing_image[:5]:
                    lines.append(f"      missing image: {m}")
            if sr.missing_mask:
                for m in sr.missing_mask[:5]:
                    lines.append(f"      missing mask: {m}")
        if r.excluded_written or r.excluded_missing:
            lines.append(
                f"  [excluded_background_only] written={r.excluded_written} "
                f"missing={len(r.excluded_missing)}"
            )
        lines.append("")
        payload["datasets"].append(dsum)

    jtxt = json.dumps(payload, indent=2) + "\n"
    ttxt = "\n".join(lines) + "\n"
    (out_root / "overlay_summary.json").write_text(jtxt, encoding="utf-8")
    (out_root / "overlay_summary.txt").write_text(ttxt, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.delineation.mask_overlays",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default=None, help="simulation | case_1 | case_2 | path")
    p.add_argument("--all", action="store_true", help="Run all three datasets")
    p.add_argument("--splits-subdir", dest="splits_subdir", type=str, default="splits_stratified")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--boundary-only", dest="boundary_only", action="store_true")
    p.add_argument("--include-legend", dest="include_legend", action="store_true")
    p.add_argument(
        "--omit-slice-label-box",
        dest="include_slice_label_box",
        action="store_false",
        help="Disable the top-left slice label box (on by default).",
    )
    p.set_defaults(include_slice_label_box=True)
    p.add_argument("--max-per-split", dest="max_per_split", type=int, default=None)
    p.add_argument(
        "--also-render-excluded",
        dest="also_render_excluded",
        action="store_true",
        help="Also render background-only slices listed in split_summary.json (e.g. case_2).",
    )
    p.add_argument("--seed", type=int, default=STRATIFICATION_SEED)
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=f"Overlay output root (default: {DEFAULT_QA_ROOT})",
    )
    args = p.parse_args(argv)

    if not args.all and not args.dataset:
        print("Provide --dataset or --all", file=sys.stderr)
        return 2

    out_root = args.out_root or DEFAULT_QA_ROOT
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root = out_root.resolve()

    jobs: List[Tuple[str, Path]] = []
    if args.all:
        for did in DATASET_IDS:
            jobs.append((did, resolve_dataset_arg(did)))
    else:
        dr = resolve_dataset_arg(args.dataset)
        did = args.dataset if args.dataset in DATASET_IDS else dr.name
        jobs.append((did, dr))

    reports: List[DatasetReport] = []
    for dataset_id, dataset_root in jobs:
        if not dataset_root.is_dir():
            print(f"SKIP missing: {dataset_root}", file=sys.stderr)
            continue
        r = process_dataset(
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            splits_subdir=args.splits_subdir,
            alpha=args.alpha,
            boundary_only=args.boundary_only,
            include_legend=args.include_legend,
            include_slice_label_box=args.include_slice_label_box,
            max_per_split=args.max_per_split,
            seed=args.seed,
            also_render_excluded=args.also_render_excluded,
            out_root=out_root,
        )
        reports.append(r)

    write_summaries(reports, args.seed, args, out_root=out_root)

    # Console
    print("\n" + "=" * 72)
    print("QA MASK OVERLAYS")
    print("=" * 72)
    for r in reports:
        print(f"\n{r.dataset_id}  (taxonomy={r.taxonomy})")
        print(f"  Output: {r.output_dir}")
        for sp, sr in r.splits.items():
            print(
                f"  {sp:5s}  written={sr.written}  requested={sr.requested}  "
                f"missing_img={len(sr.missing_image)}  missing_mask={len(sr.missing_mask)}"
            )
        if args.also_render_excluded:
            print(
                f"  excluded_background_only  written={r.excluded_written}  "
                f"missing={len(r.excluded_missing)}"
            )
    overall = bool(reports) and all(r.passed for r in reports)
    print("\n" + "-" * 72)
    print(f"Summary files: {out_root / 'overlay_summary.json'}")
    print(f"               {out_root / 'overlay_summary.txt'}")
    if not reports:
        print("No datasets processed (missing roots?).", file=sys.stderr)
    print(f"RESULT: {'PASS' if overall else 'FAIL'}")
    print("=" * 72 + "\n")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
