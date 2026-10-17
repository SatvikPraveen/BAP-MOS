"""Leave-one-cohort-out (LOCO) splits for the pooled prostate corpus.

The pooled stratified split is slice-level, so neighbouring slices of one volume can
fall in train and test. LOCO holds out one clinical cohort entirely::

    test  = every slice of the held-out cohort (its train + val + test lists)
    train = train ∪ test of the remaining cohorts
    val   = val of the remaining cohorts

Only clinical cohorts are held out; the simulation phantom always stays in training.
Writes ``<pooled>/splits_loco/heldout_<cohort>/{train,val,test}.txt`` plus
``manifest.json`` (counts and sha256 of every list). Trainers use it through
``common.splits_subdir: splits_loco/heldout_<cohort>``; test export then scores the
held-out cohort under ``<output_dir>/<cohort>/``.

Run once, before any LOCO training job::

    python -m bapmos.preprocess.prostate.create_loco_splits
    python -m bapmos.preprocess.prostate.create_loco_splits --check   # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from bapmos.paths import pooled_prostate_dataset_dir
from bapmos.preprocess.prostate.build_pooled import COHORTS, _read_split_lines

LOCO_SUBDIR = "splits_loco"
HELDOUT_COHORTS = ("case1", "case2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_split_lists() -> Dict[str, Dict[str, List[str]]]:
    """Per-cohort train/val/test lists from each cohort's ``splits_stratified/``."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for name, root, _is_sim in COHORTS:
        splits = Path(root) / "splits_stratified"
        out[name] = {
            split: _read_split_lines(splits / f"{split}.txt") for split in ("train", "val", "test")
        }
    return out


def loco_lists(
    cohorts: Dict[str, Dict[str, List[str]]],
    heldout: str,
) -> Dict[str, List[str]]:
    """Train/val/test lists with ``heldout`` as the test cohort."""
    if heldout not in cohorts:
        raise ValueError(f"Unknown cohort {heldout!r}; have {sorted(cohorts)}")
    test = [n for split in ("train", "val", "test") for n in cohorts[heldout][split]]
    train: List[str] = []
    val: List[str] = []
    for name, lists in cohorts.items():
        if name == heldout:
            continue
        train.extend(lists["train"] + lists["test"])
        val.extend(lists["val"])
    overlap = (set(train) | set(val)) & set(test)
    if overlap:
        raise RuntimeError(f"LOCO {heldout}: {len(overlap)} slices in both train/val and test")
    return {"train": train, "val": val, "test": test}


def write_loco_splits(pooled_root: Path, *, check_only: bool = False) -> Dict[str, dict]:
    """Write (or, with ``check_only``, verify) every LOCO split and its manifest."""
    cohorts = cohort_split_lists()
    images = pooled_root / "images"
    manifests: Dict[str, dict] = {}
    for heldout in HELDOUT_COHORTS:
        lists = loco_lists(cohorts, heldout)
        missing = [n for names in lists.values() for n in names if not (images / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"LOCO {heldout}: {len(missing)} listed slices missing under {images} "
                f"(first: {missing[0]})"
            )
        out_dir = pooled_root / LOCO_SUBDIR / f"heldout_{heldout}"
        manifest_path = out_dir / "manifest.json"
        if check_only:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            for split in lists:
                if _sha256(out_dir / f"{split}.txt") != stored["sha256"][split]:
                    raise RuntimeError(f"{out_dir / split}.txt changed since it was written")
            manifests[heldout] = stored
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, names in lists.items():
            (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        manifest = {
            "heldout_cohort": heldout,
            "rule": "test=all heldout slices; train=train+test of others; val=val of others",
            "counts": {split: len(names) for split, names in lists.items()},
            "sha256": {split: _sha256(out_dir / f"{split}.txt") for split in lists},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[heldout] = manifest
    return manifests


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pooled-root", type=str, default=None)
    p.add_argument("--check", action="store_true", help="Verify existing lists; write nothing.")
    args = p.parse_args(argv)
    root = Path(args.pooled_root) if args.pooled_root else pooled_prostate_dataset_dir()
    for heldout, manifest in write_loco_splits(root, check_only=args.check).items():
        print(heldout, manifest["counts"], "ok" if args.check else "written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
