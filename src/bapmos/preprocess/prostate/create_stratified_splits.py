"""
Create anatomically stratified train/val/test splits for BAP-MOS datasets.

Canonical split RNG: ``STRATIFICATION_SEED`` (default **42**). Training configs
use ``common.seed: 42`` for PyTorch/DataLoader; that is separate from but
consistent with this stratification seed.

Dataset roots (use ``--dataset`` aliases or explicit paths):

    simulation  →  preprocessing/simulation_data
    case_1      →  preprocessing/real_data/case1
    case_2      →  preprocessing/real_data/case2

Each root should contain slice PNGs under ``images/`` or ``*_dicom_png/``.
Multiclass ``*_combined_mask.png`` must live under ``<dataset_root>/masks/combined_masks/``
for **final** split generation (default ``--canonical-masks-only``). Use
``--allow-mask-fallback`` only during mask-export development (rescues from
``test_output/`` or alternate ``preprocessing/<bundle>/`` trees — never for training).

Then this script writes ``splits_stratified/`` with train/val/test lists.

**Background-only slices** (combined mask has no organ class pixels) are **excluded from the
stratified pool by default** (``--exclude-background``, the default). Use
``--include-background-only`` only if you intentionally want those slices in splits.

Writes under ``<dataset_root>/splits_stratified/`` (does not touch ``splits/``):

    train.txt, val.txt, test.txt
    split_summary.json, split_summary.txt
    stratification_report.json, stratification_report.txt  (duplicate of summary)

Also copies summaries into ``data/prostate/stratified_splits/<case_id>/``.

Usage::

    python -m bapmos.preprocess.prostate.create_stratified_splits --dataset simulation --seed 42
    python -m bapmos.preprocess.prostate.create_stratified_splits --dataset case_1 --seed 42
    python -m bapmos.preprocess.prostate.create_stratified_splits --dataset case_2 --seed 42 --exclude-background
    python -m bapmos.preprocess.prostate.create_stratified_splits --all --seed 42 --exclude-background
    python -m bapmos.preprocess.prostate.create_stratified_splits --all --seed 42   # default: canonical masks only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from bapmos.data.organ_registry import REAL_CLINICAL_ORGANS, SIMULATION_THREE_ORGANS
from bapmos.paths import (
    find_combined_masks_dir,
    find_combined_masks_dir_with_repo_fallback,
    find_training_images_dir,
    project_root,
    real_case_dataset_dir,
    simulation_dataset_dir,
)


@dataclass(frozen=True)
class OrganTaxonomy:
    name: str
    organs: tuple

    @property
    def class_ids(self) -> List[int]:
        return [o.class_id for o in self.organs]

    @property
    def display_names(self) -> List[str]:
        return [o.evaluator_label for o in self.organs]


REAL_CLINICAL = OrganTaxonomy("real_clinical", REAL_CLINICAL_ORGANS)
SIMULATION = OrganTaxonomy("simulation", SIMULATION_THREE_ORGANS)

REPO_ROOT = project_root()

# Project protocol: anatomically stratified splits are generated with this seed
# unless explicitly overridden on the CLI.
STRATIFICATION_SEED = 42

DATASET_ALIASES: Dict[str, Path] = {
    "simulation": simulation_dataset_dir(),
    "case_1": real_case_dataset_dir("case1"),
    "case_2": real_case_dataset_dir("case2"),
}


def resolve_dataset_arg(spec: str) -> Path:
    s = spec.strip()
    if s in DATASET_ALIASES:
        return DATASET_ALIASES[s]
    p = Path(s).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def list_images(dataset_root: Path) -> List[Path]:
    img_dir = find_training_images_dir(dataset_root)
    return sorted(p for p in img_dir.glob("*.png") if p.is_file())


def combined_mask_path(image_path: Path, mask_dir: Path) -> Path:
    return mask_dir / f"{image_path.stem}_combined_mask.png"


def detect_taxonomy(mask_dir: Path, sample: int = 20) -> OrganTaxonomy:
    masks = sorted(mask_dir.glob("*_combined_mask.png"))[:sample]
    if not masks:
        raise FileNotFoundError(f"No combined masks under {mask_dir}")
    max_cid = 0
    for p in masks:
        arr = np.asarray(Image.open(p))
        if arr.size:
            max_cid = max(max_cid, int(arr.max()))
    if max_cid == 0:
        raise RuntimeError(
            f"All sampled masks under {mask_dir} are background-only; cannot detect taxonomy."
        )
    if max_cid >= 4:
        return REAL_CLINICAL
    return SIMULATION


def organ_presence(mask: np.ndarray, taxonomy: OrganTaxonomy) -> Tuple[bool, ...]:
    return tuple(bool((mask == cid).any()) for cid in taxonomy.class_ids)


def pattern_label(pattern: Tuple[bool, ...], taxonomy: OrganTaxonomy) -> str:
    parts = [taxonomy.display_names[i] for i, present in enumerate(pattern) if present]
    return "+".join(parts) if parts else "Background"


def split_bucket(
    items: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Per-bucket split following project rules. Returns (train, val, test, notes).
    """
    notes: List[str] = []
    n = len(items)
    if n == 0:
        return [], [], [], notes

    items = items.copy()
    rng.shuffle(items)

    if n == 1:
        notes.append("bucket_size=1: assigned to train (coverage swaps may move later)")
        return [items[0]], [], [], notes
    if n == 2:
        notes.append("bucket_size=2: 1 train, 1 test (val empty for this pattern)")
        return [items[0]], [], [items[1]], notes
    if n == 3:
        notes.append("bucket_size=3: 1 train, 1 val, 1 test")
        return [items[0]], [items[1]], [items[2]], notes
    if n == 4:
        notes.append("bucket_size=4: 2 train, 1 val, 1 test (train/test each ≥1)")
        return items[0:2], [items[2]], [items[3]], notes
    if n == 5:
        notes.append("bucket_size=5: 3 train, 1 val, 1 test")
        return items[0:3], [items[3]], [items[4]], notes

    # n >= 6: proportional 70/15/15 with at least one per split
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, n - n_train - n_val)
    total = n_train + n_val + n_test
    if total > n:
        adjusted = {"train": n_train, "val": n_val, "test": n_test}
        overflow = total - n
        order = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)
        for k, _ in order:
            take = min(overflow, adjusted[k] - 1)
            adjusted[k] -= take
            overflow -= take
            if overflow <= 0:
                break
        n_train, n_val, n_test = adjusted["train"], adjusted["val"], adjusted["test"]
    elif total < n:
        n_train += n - total

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]
    notes.append(f"bucket_size={n}: proportional split train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test, notes


def _remove_from_bucketed(bucketed: Dict[str, List[str]], fname: str) -> str:
    for pk, lst in bucketed.items():
        if fname in lst:
            lst.remove(fname)
            return pk
    raise KeyError(fname)


def _add_to_bucketed(bucketed: Dict[str, List[str]], pk: str, fname: str) -> None:
    bucketed.setdefault(pk, []).append(fname)


def _flatten(bucketed: Dict[str, List[str]]) -> List[str]:
    return sorted(f for fs in bucketed.values() for f in fs)


def _organ_vector(files: List[str], taxonomy: OrganTaxonomy, ftp: Dict[str, Tuple[bool, ...]]) -> np.ndarray:
    v = np.zeros(len(taxonomy.organs), dtype=bool)
    for f in files:
        v |= np.array(ftp[f], dtype=bool)
    return v


def rebalance_organ_coverage(
    bucketed_train: Dict[str, List[str]],
    bucketed_val: Dict[str, List[str]],
    bucketed_test: Dict[str, List[str]],
    taxonomy: OrganTaxonomy,
    file_to_pattern: Dict[str, Tuple[bool, ...]],
    rng: random.Random,
) -> List[str]:
    """
    Ensure each organ that appears anywhere in train∪val∪test appears in
    train and test (mandatory), and in val when donors allow.

    Val coverage is **not** guaranteed if there are too few donor slices after
    filling train/test; failures are recorded in the returned notes.
    """
    notes: List[str] = []
    B = {"train": bucketed_train, "val": bucketed_val, "test": bucketed_test}

    def all_files() -> List[str]:
        return _flatten(bucketed_train) + _flatten(bucketed_val) + _flatten(bucketed_test)

    def count_organ(split: str, idx: int) -> int:
        c = 0
        for f in _flatten(B[split]):
            if file_to_pattern[f][idx]:
                c += 1
        return c

    union_vec = _organ_vector(all_files(), taxonomy, file_to_pattern)

    def try_fill(target: str, donor_order: Tuple[str, ...]) -> None:
        nonlocal notes
        for idx, organ in enumerate(taxonomy.display_names):
            if not union_vec[idx]:
                continue
            if count_organ(target, idx) > 0:
                continue
            moved = False
            for donor in donor_order:
                candidates = [f for f in _flatten(B[donor]) if file_to_pattern[f][idx]]
                if not candidates:
                    continue
                rng.shuffle(candidates)
                fname = candidates[0]
                pk = _remove_from_bucketed(B[donor], fname)
                _add_to_bucketed(B[target], pk, fname)
                notes.append(
                    f"Coverage: moved {fname} {donor}→{target} so organ {organ!r} appears in {target}"
                )
                moved = True
                break
            if not moved:
                notes.append(
                    f"WARNING: could not place organ {organ!r} into {target} (no donor slice)"
                )

    for _ in range(64):
        before = json.dumps({k: sorted(_flatten(B[k])) for k in B})
        try_fill("test", ("train", "val"))
        try_fill("train", ("val", "test"))
        try_fill("val", ("train", "test"))
        after = json.dumps({k: sorted(_flatten(B[k])) for k in B})
        if before == after:
            break

    return notes


def _pattern_counts(bucketed: Dict[str, List[str]], ftp: Dict[str, Tuple[bool, ...]], tax: OrganTaxonomy) -> Counter:
    c: Counter = Counter()
    for fs in bucketed.values():
        for f in fs:
            c[pattern_label(ftp[f], tax)] += 1
    return c


def _combinations_missing_from_splits(
    bucket_totals: Dict[str, int],
    per_split: Dict[str, Dict[str, int]],
    patterns: List[str],
) -> Dict[str, List[Dict[str, str]]]:
    """per_split[split][pattern] = count."""
    out: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for pat in patterns:
        T = bucket_totals.get(pat, 0)
        for split in ("train", "val", "test"):
            cnt = per_split[split].get(pat, 0)
            if cnt > 0:
                continue
            if T == 0:
                continue
            if T >= 3:
                reason = (
                    f"bucket_total={T}: expected representation in {split} "
                    f"when bucket≥3 (check split logic or coverage swaps)"
                )
            elif T == 2 and split == "val":
                reason = "bucket_total=2: by rule val is empty for this pattern"
            elif T == 1:
                reason = "bucket_total=1: single split assignment (train-only before swaps)"
            elif T == 2 and split in ("train", "test"):
                reason = "bucket_total=2: impossible to cover both train and test with third split empty"
            else:
                reason = f"bucket_total={T}: pattern absent from {split}"
            out[split].append({"pattern": pat, "reason": reason})
    return out


def stratify(
    dataset_root: Path,
    out_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    exclude_background_only: bool,
    *,
    canonical_masks_only: bool = True,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    if canonical_masks_only:
        mask_dir = find_combined_masks_dir(dataset_root)
    else:
        mask_dir = find_combined_masks_dir_with_repo_fallback(
            dataset_root, allow_test_output_fallback=True
        )
    taxonomy = detect_taxonomy(mask_dir)
    images = list_images(dataset_root)

    buckets: Dict[Tuple[bool, ...], List[str]] = defaultdict(list)
    file_to_pattern: Dict[str, Tuple[bool, ...]] = {}
    background_only: List[str] = []
    missing_masks: List[str] = []

    for img_path in images:
        mp = combined_mask_path(img_path, mask_dir)
        if not mp.is_file():
            missing_masks.append(img_path.name)
            continue
        mask = np.asarray(Image.open(mp))
        pat = organ_presence(mask, taxonomy)
        file_to_pattern[img_path.name] = pat
        if not any(pat):
            background_only.append(img_path.name)
            if not exclude_background_only:
                buckets[pat].append(img_path.name)
        else:
            buckets[pat].append(img_path.name)

    for k in buckets:
        buckets[k].sort()

    rare_notes: List[str] = []
    bucketed_train: Dict[str, List[str]] = {}
    bucketed_val: Dict[str, List[str]] = {}
    bucketed_test: Dict[str, List[str]] = {}

    for pat, files in buckets.items():
        tr, va, te, bn = split_bucket(files, train_frac, val_frac, test_frac, rng)
        key = pattern_label(pat, taxonomy)
        rare_notes.extend([f"{key}: {x}" for x in bn])
        bucketed_train[key] = tr
        bucketed_val[key] = va
        bucketed_test[key] = te

    cov_notes = rebalance_organ_coverage(
        bucketed_train, bucketed_val, bucketed_test, taxonomy, file_to_pattern, rng
    )

    train_list = sorted(_flatten(bucketed_train))
    val_list = sorted(_flatten(bucketed_val))
    test_list = sorted(_flatten(bucketed_test))

    if exclude_background_only:
        bg_set = set(background_only)
        for split_name, lst in (("train", train_list), ("val", val_list), ("test", test_list)):
            leaked = sorted(f for f in lst if f in bg_set)
            if leaked:
                raise RuntimeError(
                    f"Stratification internal error: {len(leaked)} background-only slice(s) in "
                    f"{split_name} despite exclude_background_only=True (e.g. {leaked[0]!r})"
                )

    def organ_counts(files: List[str]) -> Dict[str, int]:
        c = {name: 0 for name in taxonomy.display_names}
        for f in files:
            for i, present in enumerate(file_to_pattern[f]):
                if present:
                    c[taxonomy.display_names[i]] += 1
        return c

    bucket_totals: Dict[str, int] = {}
    for pat, files in buckets.items():
        bucket_totals[pattern_label(pat, taxonomy)] = len(files)

    all_patterns = sorted(bucket_totals.keys())
    per_split_patterns = {
        "train": dict(_pattern_counts(bucketed_train, file_to_pattern, taxonomy)),
        "val": dict(_pattern_counts(bucketed_val, file_to_pattern, taxonomy)),
        "test": dict(_pattern_counts(bucketed_test, file_to_pattern, taxonomy)),
    }
    missing_combo = _combinations_missing_from_splits(bucket_totals, per_split_patterns, all_patterns)

    excluded_bg = len(background_only) if exclude_background_only else 0
    stratified_slice_count = len(train_list) + len(val_list) + len(test_list)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "combined_masks_directory": str(mask_dir.resolve()),
        "canonical_masks_only": canonical_masks_only,
        "taxonomy": taxonomy.name,
        "organs": taxonomy.display_names,
        "random_seed": seed,
        "split_ratio": {"train": train_frac, "val": val_frac, "test": test_frac},
        "exclude_background_only": exclude_background_only,
        "totals": {
            "total_available_slices": len(images),
            "missing_masks": len(missing_masks),
            "background_only_slice_count": len(background_only),
            "excluded_background_only_slices": excluded_bg,
            "stratified_slice_count": stratified_slice_count,
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list),
        },
        "organ_counts_by_split": {
            "train": organ_counts(train_list),
            "val": organ_counts(val_list),
            "test": organ_counts(test_list),
        },
        "organ_combination_counts_by_split": per_split_patterns,
        "organ_combination_totals": bucket_totals,
        "combinations_missing_from_split": missing_combo,
        "rare_bucket_handling": rare_notes,
        "coverage_swap_notes": cov_notes,
        "missing_masks_list": sorted(missing_masks),
        "background_only_list": sorted(background_only),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_list) + ("\n" if train_list else ""))
    (out_dir / "val.txt").write_text("\n".join(val_list) + ("\n" if val_list else ""))
    (out_dir / "test.txt").write_text("\n".join(test_list) + ("\n" if test_list else ""))

    json_text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "split_summary.json").write_text(json_text)
    (out_dir / "stratification_report.json").write_text(json_text)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("STRATIFIED SPLIT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append(f"Dataset:   {summary['dataset_root']}")
    lines.append(f"Mask dir:  {summary.get('combined_masks_directory', '')}")
    lines.append(f"Taxonomy:  {summary['taxonomy']} ({', '.join(summary['organs'])})")
    lines.append(f"Seed:      {seed}")
    lines.append(f"Ratio:     train={train_frac}  val={val_frac}  test={test_frac}")
    lines.append(f"Exclude background-only from split: {exclude_background_only}")
    lines.append(f"Canonical masks only: {summary.get('canonical_masks_only', True)}")
    lines.append("")
    t = summary["totals"]
    lines.append(f"Total images on disk:     {t['total_available_slices']}")
    lines.append(f"Missing masks:            {t['missing_masks']}")
    lines.append(f"Background-only slices:   {t['background_only_slice_count']}")
    lines.append(f"Excluded background-only: {t['excluded_background_only_slices']}")
    lines.append(
        f"Stratified pool size:     {t['stratified_slice_count']}  "
        f"(train={t['train']}, val={t['val']}, test={t['test']})"
    )
    lines.append("")
    lines.append("Per-organ slice counts (organ present on slice):")
    lines.append(f"  {'Organ':<14} {'Train':>7} {'Val':>5} {'Test':>5}")
    for name in taxonomy.display_names:
        oc = summary["organ_counts_by_split"]
        lines.append(
            f"  {name:<14} {oc['train'][name]:>7} {oc['val'][name]:>5} {oc['test'][name]:>5}"
        )
    lines.append("")
    lines.append("Organ-combination counts per split:")
    lines.append(f"  {'Pattern':<36} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}")
    for pat in all_patterns:
        tr = per_split_patterns["train"].get(pat, 0)
        va = per_split_patterns["val"].get(pat, 0)
        te = per_split_patterns["test"].get(pat, 0)
        tot = bucket_totals.get(pat, 0)
        lines.append(f"  {pat:<36} {tr:>6} {va:>5} {te:>5} {tot:>6}")
    lines.append("")
    lines.append("Combinations missing from a split (see reason):")
    for split in ("train", "val", "test"):
        items = missing_combo.get(split) or []
        if not items:
            lines.append(f"  {split}: (none)")
            continue
        lines.append(f"  {split}:")
        for it in items:
            lines.append(f"    - {it['pattern']}: {it['reason']}")
    lines.append("")
    lines.append("Rare bucket notes:")
    for n in rare_notes:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("Coverage swaps:")
    for n in cov_notes:
        lines.append(f"  - {n}")
    if missing_masks:
        lines.append("")
        lines.append("Missing masks:")
        for n in missing_masks:
            lines.append(f"  - {n}")
    lines.append("=" * 80)
    txt = "\n".join(lines) + "\n"
    (out_dir / "split_summary.txt").write_text(txt)
    (out_dir / "stratification_report.txt").write_text(txt)

    return summary


def _case_id_for_path(dataset_root: Path) -> str:
    for k, v in DATASET_ALIASES.items():
        if v.resolve() == dataset_root.resolve():
            return k
    return dataset_root.name


def copy_outputs_to_stratified_splits_root(case_id: str, out_dir: Path) -> Path:
    """Mirror split lists + summaries to ``data/prostate/stratified_splits/<case_id>/``."""
    dest = REPO_ROOT / "data" / "prostate" / "stratified_splits" / case_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "train.txt",
        "val.txt",
        "test.txt",
        "split_summary.json",
        "split_summary.txt",
        "stratification_report.json",
        "stratification_report.txt",
    ):
        src = out_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    return dest


def _write_run_aggregate(
    job_specs: List[str],
    out_subdir: str,
    seed: int,
    exclude_background: bool,
) -> None:
    """Write aggregate ``run_summary.{txt,json}`` under ``data/prostate/stratified_splits/``."""
    base = REPO_ROOT / "data" / "prostate" / "stratified_splits"
    base.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    lines = [
        "Stratified splits (--all)",
        f"seed={seed}  exclude_background_only={exclude_background}",
        "",
        "Columns:",
        "  total          = train+val+test (stratified pool size)",
        "  bg_only        = slices whose *_combined_mask.png exists but has NO organ pixels",
        "                   (excluded from pool when exclude_background_only=True)",
        "  missing_mask   = training PNGs with no matching *_combined_mask.png",
        "                   (never enter the stratified pool; not counted as bg_only)",
        "",
        f"{'case':<12} {'train':>6} {'val':>5} {'test':>5} {'total':>6}  "
        f"{'bg_only':>7}  {'excl_bg':>8}  {'miss_mask':>9}",
        "-" * 80,
    ]
    for spec in job_specs:
        dr = resolve_dataset_arg(spec)
        cid = _case_id_for_path(dr)
        p = dr / out_subdir / "split_summary.json"
        if not p.is_file():
            continue
        ss = json.loads(p.read_text())
        t = ss["totals"]
        lines.append(
            f"{cid:<12} {t['train']:>6} {t['val']:>5} {t['test']:>5} "
            f"{t['stratified_slice_count']:>6}  "
            f"{t['background_only_slice_count']:>7}  {t['excluded_background_only_slices']:>8}  "
            f"{t['missing_masks']:>9}"
        )
        rows.append({"case_id": cid, **t, "taxonomy": ss.get("taxonomy")})
    lines.append("")
    (base / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base / "run_summary.json").write_text(
        json.dumps({"seed": seed, "exclude_background_only": exclude_background, "datasets": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.prostate.create_stratified_splits",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="simulation | case_1 | case_2 | or path to dataset root.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run for simulation, case_1, and case_2.",
    )
    p.add_argument("--out_subdir", type=str, default="splits_stratified")
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=STRATIFICATION_SEED)
    p.add_argument(
        "--exclude-background",
        dest="exclude_background",
        action="store_true",
        help="Exclude background-only slices from stratified pool (default).",
    )
    p.add_argument(
        "--include-background-only",
        dest="exclude_background",
        action="store_false",
        help="Include background-only slices in stratified pool.",
    )
    p.set_defaults(exclude_background=True)
    p.add_argument(
        "--canonical-masks-only",
        dest="canonical_masks_only",
        action="store_true",
        default=True,
        help="Require masks under <dataset_root>/masks/ only (default; use for final splits).",
    )
    p.add_argument(
        "--allow-mask-fallback",
        dest="canonical_masks_only",
        action="store_false",
        help="Allow test_output / alternate preprocessing bundle mask discovery (dev/rescue only).",
    )

    args = p.parse_args(argv)
    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise SystemExit("train_frac + val_frac + test_frac must sum to 1")

    jobs: List[str] = []
    if args.all:
        jobs = ["simulation", "case_1", "case_2"]
    else:
        if not args.dataset:
            raise SystemExit("Provide --dataset or use --all")
        jobs = [args.dataset]

    for spec in jobs:
        dataset_root = resolve_dataset_arg(spec)
        if not dataset_root.is_dir():
            print(f"SKIP missing directory: {dataset_root}", file=sys.stderr)
            continue
        out_dir = dataset_root / args.out_subdir
        stratify(
            dataset_root=dataset_root,
            out_dir=out_dir,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            exclude_background_only=args.exclude_background,
            canonical_masks_only=args.canonical_masks_only,
        )
        cid = _case_id_for_path(dataset_root)
        dest = copy_outputs_to_stratified_splits_root(cid, out_dir)
        print(f"Wrote splits and summaries to {out_dir}")
        print(f"  Copied to {dest}")

    if len(jobs) > 1:
        _write_run_aggregate(jobs, args.out_subdir, args.seed, args.exclude_background)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
