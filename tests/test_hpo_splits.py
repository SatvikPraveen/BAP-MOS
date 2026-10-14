"""PFUS1 search splits: patient-disjoint subsets of the parent training patients."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bapmos.preprocess.bladder.create_hpo_splits import build_hpo_splits

PATIENTS = ("P000", "P001", "P002", "P003", "P004")


@pytest.fixture
def mini_pfus1(tmp_path: Path) -> Path:
    root = tmp_path / "pfus1"
    parent = root / "splits_parent"
    parent.mkdir(parents=True)
    lines = [f"{p}/frame_{f:03d}" for p in PATIENTS for f in range(3)]
    (parent / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_search_split_is_patient_disjoint(mini_pfus1: Path) -> None:
    out = build_hpo_splits(
        data_root=mini_pfus1,
        parent_splits_subdir="splits_parent",
        output_splits_subdir="splits_hpo",
        patient_fraction=1.0,
        val_patient_fraction=0.4,
        seed=4242,
    )
    train = set((out / "train_patients.txt").read_text().split())
    val = set((out / "val_patients.txt").read_text().split())
    summary = json.loads((out / "split_summary.json").read_text(encoding="utf-8"))
    assert train.isdisjoint(val)
    assert train | val == set(PATIENTS)
    assert summary["tpe_train"]["n_frames"] + summary["tpe_val"]["n_frames"] == 15
    assert not (out / "test.txt").exists()


def test_patient_fraction_subsamples_the_cohort(mini_pfus1: Path) -> None:
    out = build_hpo_splits(
        data_root=mini_pfus1,
        parent_splits_subdir="splits_parent",
        output_splits_subdir="splits_hpo_small",
        patient_fraction=0.4,
        val_patient_fraction=0.25,
        seed=99,
    )
    summary = json.loads((out / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["tpe_cohort_patients"] == 2


def test_same_seed_gives_the_same_split(mini_pfus1: Path) -> None:
    kwargs = dict(data_root=mini_pfus1, parent_splits_subdir="splits_parent", patient_fraction=0.6, seed=4242)
    a = build_hpo_splits(output_splits_subdir="a", **kwargs)
    b = build_hpo_splits(output_splits_subdir="b", **kwargs)
    assert (a / "train.txt").read_text() == (b / "train.txt").read_text()
    assert (a / "val.txt").read_text() == (b / "val.txt").read_text()
