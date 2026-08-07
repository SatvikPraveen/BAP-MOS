"""
Patient-level train / val / test split lists for PFUS1 (no cross-validation folds).

Rules:
  * Every line is ``Pxxx/frame_yyy`` (PNG+JSON pairs under ``raw_root``).
  * **All frames from one patient stay in the same split** (no patient leakage).
  * **~70 / 15 / 15** patient fractions **globally** (largest remainder on ``n_patients``),
    applied **within each frame-count bin** so train/val/test each include low/mid/high
    patients (shuffle **within bin** with **seed 42**, then contiguous bin-internal splits).
  * **Each non-empty frame-count bin must contain at least 3 patients**; otherwise split
    generation fails (strict protocol for representativeness across low/mid/high).

Bins (by number of valid frames per patient):
  * low:  ≤ 110
  * mid:  111–170
  * high: > 170

Label-coverage stats in ``split_summary.*`` are computed from **JSON polygons** at
``min_polygon_area`` (not from exported combined masks).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Sequence, Tuple

import numpy as np

from bapmos.paths import project_root
from bapmos.preprocess.bladder.constants import JSON_LABEL_TO_CLASS_ID, PFUS1_ALL_LABELS


def _root() -> Path:
    return project_root()


BinName = Literal["low", "mid", "high"]

LOW_BIN_MAX_FRAMES = 110
MID_BIN_MAX_FRAMES = 170


def frame_count_bin(n_frames: int) -> BinName:
    if n_frames <= LOW_BIN_MAX_FRAMES:
        return "low"
    if n_frames <= MID_BIN_MAX_FRAMES:
        return "mid"
    return "high"


def list_patients(raw_root: Path) -> List[str]:
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("P"))


def list_samples_for_patient(patient_dir: Path) -> List[str]:
    keys: List[str] = []
    for png in sorted(patient_dir.glob("frame_*.png")):
        stem = png.stem
        if (patient_dir / f"{stem}.json").is_file():
            keys.append(f"{patient_dir.name}/{stem}")
    return keys


def _polygon_area_xy(poly: List[List[float]]) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def json_label_presence(json_path: Path, min_area: float) -> Dict[str, bool]:
    """Per-frame polygon presence (same rule as ``analyze_pfus1_dataset``)."""
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)
    present: Dict[str, bool] = {name: False for name, _ in PFUS1_ALL_LABELS}
    if not isinstance(ann, list):
        return present
    for obj in ann:
        label = obj.get("label")
        pol = obj.get("pol")
        if label not in JSON_LABEL_TO_CLASS_ID:
            continue
        if not isinstance(pol, list) or len(pol) < 3:
            continue
        if _polygon_area_xy(pol) >= min_area:
            present[str(label)] = True
    return present


def _split_patient_counts(n_patients: int) -> Tuple[int, int, int]:
    """
    train / val / test patient counts targeting **70% / 15% / 15%** using the
    largest-remainder method (Hamilton), so counts always sum to ``n_patients``.
    """
    if n_patients < 3:
        raise ValueError(f"Need at least 3 patients; got {n_patients}")
    props = (0.70, 0.15, 0.15)
    exact = [n_patients * p for p in props]
    floors = [int(np.floor(x)) for x in exact]
    rem = [e - float(f) for e, f in zip(exact, floors)]
    left = n_patients - sum(floors)
    idx_order = sorted(range(3), key=lambda i: (-rem[i], i))
    for i in idx_order[:left]:
        floors[i] += 1
    n_train, n_val, n_test = floors[0], floors[1], floors[2]
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"70/15/15 allocation yielded an empty split for n_patients={n_patients} "
            f"→ ({n_train}, {n_val}, {n_test}). Use a larger cohort."
        )
    return n_train, n_val, n_test


def _split_patients_per_bin_stratified(
    bins: Dict[BinName, List[str]],
    rng: np.random.RandomState,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Shuffle within each frame-count bin, then apply the same ~70/15/15 LR rule
    **inside that bin**. Intended to match the global LR totals; ``write_patient_splits``
    validates equality and raises otherwise. Each non-empty bin must have ≥3 patients.
    """
    train_all: List[str] = []
    val_all: List[str] = []
    test_all: List[str] = []
    for b in ("low", "mid", "high"):
        plist = list(bins.get(b, []))
        if not plist:
            continue
        rng.shuffle(plist)
        n_b = len(plist)
        if n_b < 3:
            raise ValueError(
                f"Frame-count bin {b!r} has only {n_b} patient(s); "
                "need at least 3 per bin for independent 70/15/15 splits."
            )
        nt, nv, ne = _split_patient_counts(n_b)
        train_all.extend(plist[:nt])
        val_all.extend(plist[nt : nt + nv])
        test_all.extend(plist[nt + nv :])
    return train_all, val_all, test_all


def _aggregate_label_hits(
    raw_root: Path,
    sample_keys: Sequence[str],
    min_area: float,
) -> Tuple[DefaultDict[str, int], int]:
    """Count frames where each label's polygon meets ``min_area``."""
    hits: DefaultDict[str, int] = defaultdict(int)
    n_ok = 0
    for key in sample_keys:
        patient, stem = key.split("/", 1)
        jp = raw_root / patient / f"{stem}.json"
        if not jp.is_file():
            continue
        pres = json_label_presence(jp, min_area)
        n_ok += 1
        for lab, ok in pres.items():
            if ok:
                hits[lab] += 1
    return hits, n_ok


def write_patient_splits(
    raw_root: Path,
    out_root: Path,
    seed: int,
    min_polygon_area: float,
) -> Dict[str, Any]:
    patients = list_patients(raw_root)
    n_pat = len(patients)
    n_train, n_val, n_test = _split_patient_counts(n_pat)

    patient_to_samples: Dict[str, List[str]] = {}
    for p in patients:
        patient_to_samples[p] = list_samples_for_patient(raw_root / p)
    missing = [p for p, s in patient_to_samples.items() if not s]
    if missing:
        raise RuntimeError(f"Patients with no valid PNG+JSON pairs: {missing}")

    patient_n_frames = {p: len(patient_to_samples[p]) for p in patients}
    bins: Dict[BinName, List[str]] = {"low": [], "mid": [], "high": []}
    for p in patients:
        bins[frame_count_bin(patient_n_frames[p])].append(p)

    rng = np.random.RandomState(seed)
    train_p, val_p, test_p = _split_patients_per_bin_stratified(bins, rng)
    if len(train_p) != n_train or len(val_p) != n_val or len(test_p) != n_test:
        raise RuntimeError(
            "Per-bin split sizes mismatch global targets "
            f"(train/val/test got {len(train_p)}/{len(val_p)}/{len(test_p)}, "
            f"expected {n_train}/{n_val}/{n_test})."
        )

    def expand(patient_list: List[str]) -> List[str]:
        keys: List[str] = []
        for p in sorted(patient_list):
            keys.extend(patient_to_samples[p])
        return keys

    train_keys = expand(train_p)
    val_keys = expand(val_p)
    test_keys = expand(test_p)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train_patients.txt").write_text(
        "\n".join(sorted(train_p)) + "\n", encoding="utf-8"
    )
    (out_root / "val_patients.txt").write_text(
        "\n".join(sorted(val_p)) + "\n", encoding="utf-8"
    )
    (out_root / "test_patients.txt").write_text(
        "\n".join(sorted(test_p)) + "\n", encoding="utf-8"
    )
    (out_root / "train.txt").write_text("\n".join(train_keys) + "\n", encoding="utf-8")
    (out_root / "val.txt").write_text("\n".join(val_keys) + "\n", encoding="utf-8")
    (out_root / "test.txt").write_text("\n".join(test_keys) + "\n", encoding="utf-8")

    def bin_mix(patient_list: List[str]) -> Dict[str, int]:
        mix = {"low": 0, "mid": 0, "high": 0}
        for p in patient_list:
            mix[frame_count_bin(patient_n_frames[p])] += 1
        return mix

    def frame_stats(patient_list: List[str]) -> Dict[str, float]:
        counts = [patient_n_frames[p] for p in patient_list]
        if not counts:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "sum": 0.0}
        return {
            "min": float(min(counts)),
            "max": float(max(counts)),
            "mean": float(sum(counts) / len(counts)),
            "sum": float(sum(counts)),
        }

    splits_payload: Dict[str, Any] = {}
    all_labels = [name for name, _ in PFUS1_ALL_LABELS]
    for name, plist, keys in (
        ("train", train_p, train_keys),
        ("val", val_p, val_keys),
        ("test", test_p, test_keys),
    ):
        hits, _n_scanned = _aggregate_label_hits(raw_root, keys, min_polygon_area)
        label_any_frame = {lab: hits[lab] > 0 for lab in all_labels}
        splits_payload[name] = {
            "patients": sorted(plist),
            "n_patients": len(plist),
            "n_frames": len(keys),
            "patient_frame_count_stats": frame_stats(plist),
            "patients_by_frame_bin": bin_mix(plist),
            "label_frame_hits_ge_min_area": {lab: int(hits[lab]) for lab in all_labels},
            "label_present_in_split_any_frame": label_any_frame,
        }

    total_frames = len(train_keys) + len(val_keys) + len(test_keys)
    expected = sum(len(patient_to_samples[p]) for p in patients)
    overlap_ok = set(train_p).isdisjoint(val_p) and set(train_p).isdisjoint(test_p) and set(val_p).isdisjoint(test_p)
    labels_all_splits = all(
        all(splits_payload[sp]["label_present_in_split_any_frame"].get(lab, False) for sp in ("train", "val", "test"))
        for lab in all_labels
    )

    summary: Dict[str, Any] = {
        "seed": seed,
        "min_polygon_area_px2": min_polygon_area,
        "protocol": "patient_level_train_val_test_per_frame_bin_70_15_15",
        "target_ratios_note": "~70/15/15 patients globally (LR); same rule applied inside each frame-count bin",
        "n_patients": n_pat,
        "patient_split_counts": {"train": n_train, "val": n_val, "test": n_test},
        "frame_count_bins": {
            "low": f"<= {LOW_BIN_MAX_FRAMES} frames",
            "mid": f"{LOW_BIN_MAX_FRAMES + 1}–{MID_BIN_MAX_FRAMES} frames",
            "high": f"> {MID_BIN_MAX_FRAMES} frames",
        },
        "patients_per_bin": {b: sorted(bins[b]) for b in ("low", "mid", "high")},
        "allocation": (
            "Shuffle patients within each frame-count bin (seeded RNG), then apply "
            "the same largest-remainder 70/15/15 split independently inside each bin; "
            "concatenate train / val / test lists across bins (sums match global LR)."
        ),
        "splits": splits_payload,
        "totals": {
            "n_frames_all_splits": total_frames,
            "n_frames_expected_from_raw": expected,
            "frames_match": total_frames == expected,
        },
        "validation": {
            "no_patient_overlap": overlap_ok,
            "seed_is_42": seed == 42,
            "every_label_present_in_train_val_test_any_frame": labels_all_splits,
        },
    }

    (out_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines_txt: List[str] = []
    lines_txt.append("PFUS1 patient-level split summary")
    lines_txt.append("=" * 72)
    lines_txt.append(f"seed: {seed}")
    lines_txt.append(f"n_patients: {n_pat}  →  train {n_train}  val {n_val}  test {n_test}")
    lines_txt.append(
        f"frame bins: low <= {LOW_BIN_MAX_FRAMES}, mid {LOW_BIN_MAX_FRAMES + 1}–{MID_BIN_MAX_FRAMES}, high > {MID_BIN_MAX_FRAMES}"
    )
    lines_txt.append(
        "allocation: per-bin shuffle (seed); 70/15/15 LR inside each bin; concat splits"
    )
    lines_txt.append("")
    for sp in ("train", "val", "test"):
        block = splits_payload[sp]
        lines_txt.append(f"[{sp}] patients={block['n_patients']} frames={block['n_frames']}")
        lines_txt.append(f"  frame-count (per patient): min={block['patient_frame_count_stats']['min']:.0f} "
                         f"max={block['patient_frame_count_stats']['max']:.0f} "
                         f"mean={block['patient_frame_count_stats']['mean']:.1f}")
        lines_txt.append(f"  bin mix (patients): low={block['patients_by_frame_bin']['low']} "
                         f"mid={block['patients_by_frame_bin']['mid']} high={block['patients_by_frame_bin']['high']}")
        lines_txt.append("")
    lines_txt.append("Checks")
    lines_txt.append("-" * 40)
    v = summary["validation"]
    lines_txt.append(f"  [ {'x' if v['no_patient_overlap'] else ' '} ] no patient appears in more than one split")
    lines_txt.append(f"  [ {'x' if summary['totals']['frames_match'] else ' '} ] all frames accounted for ({total_frames} == {expected})")
    lines_txt.append(f"  [ {'x' if v['seed_is_42'] else ' '} ] seed == 42")
    lines_txt.append(f"  [ {'x' if v['every_label_present_in_train_val_test_any_frame'] else ' '} ] each label appears in train, val, and test (any frame)")
    lines_txt.append("")
    (out_root / "split_summary.txt").write_text(
        "\n".join(lines_txt) + "\n", encoding="utf-8"
    )

    print(f"Wrote split under {out_root.resolve()}")
    for ln in lines_txt:
        print(ln)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=_root() / "data/bladder/pfus1_raw",
        help="Directory containing P000, P001, ...",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=_root() / "data/bladder/pfus1/splits_patient_70_15_15_seed42",
        help="Output directory for train/val/test lists and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_polygon_area",
        type=float,
        default=1.0,
        help="Same threshold as analyze_pfus1_dataset for label hit counts.",
    )
    args = parser.parse_args(argv)

    if not args.raw_root.is_dir():
        print(f"ERROR: raw_root not found: {args.raw_root}", file=sys.stderr)
        return 1

    write_patient_splits(
        args.raw_root.resolve(),
        args.out_root.resolve(),
        args.seed,
        args.min_polygon_area,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
