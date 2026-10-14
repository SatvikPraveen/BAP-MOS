"""
Patient-level train / val lists for the PFUS1 outer-loop (TPE) search.

The search runs on a subset carved from the **training patients** of the parent split
(``splits_patient_70_15_15_seed42``), so the parent validation and test patients stay
untouched for the final runs and evaluation.

Defaults reproduce the paper's search split ``splits_hpo_tpe_patient_seed4242_p20``:
  * 20% of the 77 training patients form the search cohort (seed 4242);
  * 20% of that cohort is search validation, the rest search training (seed 4243);
  * all frames of a patient stay on one side (no patient leakage).

Usage (from the repository root)::

    python -m bapmos.preprocess.bladder.create_hpo_splits --data_root data/bladder/pfus1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bapmos.paths import pfus1_bundle_dir
from bapmos.preprocess.bladder.dataset import parse_sample_line

DEFAULT_PARENT_SPLIT = "splits_patient_70_15_15_seed42"
DEFAULT_OUTPUT_SUBDIR = "splits_hpo_tpe_patient_seed4242_p20"
DEFAULT_SEED = 4242
DEFAULT_PATIENT_FRACTION = 0.20
DEFAULT_VAL_PATIENT_FRACTION = 0.20


def _read_frame_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def _group_frames_by_patient(frame_lines: Sequence[str]) -> Dict[str, List[str]]:
    by_patient: Dict[str, List[str]] = defaultdict(list)
    for key in frame_lines:
        patient, _stem = parse_sample_line(key)
        by_patient[patient].append(key)
    for patient in by_patient:
        by_patient[patient].sort()
    return dict(by_patient)


def _select_patients(patients: Sequence[str], *, patient_fraction: float, seed: int) -> List[str]:
    unique = sorted(set(patients))
    if not unique:
        raise ValueError("No patients in source train split.")
    if not 0.0 < patient_fraction <= 1.0:
        raise ValueError(f"patient_fraction must be in (0, 1], got {patient_fraction}")
    if patient_fraction == 1.0:
        return unique
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n = min(len(unique), max(2, int(round(len(unique) * patient_fraction))))
    return sorted(shuffled[:n])


def _split_patients_train_val(
    patients: Sequence[str], *, val_patient_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_patient_fraction < 1.0:
        raise ValueError(f"val_patient_fraction must be in (0, 1), got {val_patient_fraction}")
    unique = sorted(set(patients))
    if len(unique) < 2:
        raise ValueError("Need at least 2 patients to form search train/val.")
    rng = random.Random(seed + 1)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n_val = min(len(unique) - 1, max(1, int(round(len(unique) * val_patient_fraction))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _frames_for_patients(by_patient: Dict[str, List[str]], patients: Sequence[str]) -> List[str]:
    out: List[str] = []
    for patient in sorted(patients):
        out.extend(by_patient[patient])
    return out


def build_hpo_splits(
    *,
    data_root: Path,
    parent_splits_subdir: str = DEFAULT_PARENT_SPLIT,
    output_splits_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    patient_fraction: float = DEFAULT_PATIENT_FRACTION,
    val_patient_fraction: float = DEFAULT_VAL_PATIENT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write search-only ``train.txt`` / ``val.txt`` from the parent split's training patients."""
    train_src = data_root / parent_splits_subdir / "train.txt"
    if not train_src.is_file():
        raise FileNotFoundError(f"Missing source train split: {train_src}")

    by_patient = _group_frames_by_patient(_read_frame_lines(train_src))
    source_patients = sorted(by_patient)
    cohort = _select_patients(source_patients, patient_fraction=patient_fraction, seed=seed)
    train_patients, val_patients = _split_patients_train_val(
        cohort, val_patient_fraction=val_patient_fraction, seed=seed
    )
    train_frames = _frames_for_patients(by_patient, train_patients)
    val_frames = _frames_for_patients(by_patient, val_patients)

    out_dir = data_root / output_splits_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_frames) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_frames) + "\n", encoding="utf-8")
    (out_dir / "train_patients.txt").write_text("\n".join(train_patients) + "\n", encoding="utf-8")
    (out_dir / "val_patients.txt").write_text("\n".join(val_patients) + "\n", encoding="utf-8")

    summary = {
        "protocol": "hpo_tpe_patient_subset_from_train_cohort",
        "parent_splits_subdir": parent_splits_subdir,
        "seed": seed,
        "patient_fraction": patient_fraction,
        "val_patient_fraction": val_patient_fraction,
        "source_train_patients": len(source_patients),
        "source_train_frames": sum(len(v) for v in by_patient.values()),
        "tpe_cohort_patients": len(cohort),
        "tpe_train": {"n_patients": len(train_patients), "n_frames": len(train_frames), "patients": train_patients},
        "tpe_val": {"n_patients": len(val_patients), "n_frames": len(val_frames), "patients": val_patients},
        "parent_val_and_test_untouched": True,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=pfus1_bundle_dir(), help="PFUS1 bundle (default: data/bladder/pfus1).")
    parser.add_argument("--parent_splits_subdir", default=DEFAULT_PARENT_SPLIT)
    parser.add_argument("--output_splits_subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--patient_fraction", type=float, default=DEFAULT_PATIENT_FRACTION,
                        help="Fraction of the parent training patients in the search cohort.")
    parser.add_argument("--val_patient_fraction", type=float, default=DEFAULT_VAL_PATIENT_FRACTION,
                        help="Fraction of the search cohort used for search validation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output split.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = data_root / args.output_splits_subdir
    if out_dir.is_dir() and not args.force:
        sys.exit(f"ERROR: {out_dir} already exists (use --force to regenerate)")
    try:
        out = build_hpo_splits(
            data_root=data_root,
            parent_splits_subdir=args.parent_splits_subdir,
            output_splits_subdir=args.output_splits_subdir,
            patient_fraction=args.patient_fraction,
            val_patient_fraction=args.val_patient_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
    summary = json.loads((out / "split_summary.json").read_text(encoding="utf-8"))
    print(f"Wrote search splits -> {out}")
    print(f"  search train: {summary['tpe_train']['n_patients']} patients, {summary['tpe_train']['n_frames']} frames")
    print(f"  search val:   {summary['tpe_val']['n_patients']} patients, {summary['tpe_val']['n_frames']} frames")


if __name__ == "__main__":
    main()
