"""Patient-level train/val splits for nnU-Net (BAP-MOS / U-Net parity).

nnU-Net reads ``splits_final.json`` under the preprocessed dataset folder. By default it
builds a random 5-fold CV. This module writes a single fold from ``case_mapping.json`` +
``data/.../splits_*`` (or explicit ``train``/``val`` sections in the mapping).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from bapmos.external_baselines.nnunet2d.export_nnunet_dataset import dataset_folder


def read_split_sample_keys(data_root: Path, splits_subdir: str, split: str) -> List[str]:
    """Ordered sample keys from ``{data_root}/{splits_subdir}/{split}.txt``."""
    fp = Path(data_root) / splits_subdir / f"{split}.txt"
    if not fp.is_file():
        raise FileNotFoundError(f"Missing split file: {fp}")
    keys: List[str] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    return keys


def partition_case_ids_from_mapping(
    mapping_doc: Dict[str, Any],
    *,
    data_root: Path | None = None,
) -> Tuple[List[str], List[str]]:
    """Return (train_case_ids, val_case_ids) in nnU-Net ``case*****`` form."""
    train_entries = list(mapping_doc.get("train") or [])
    val_entries = list(mapping_doc.get("val") or [])

    if train_entries and val_entries:
        return (
            [str(e["case_id"]) for e in train_entries],
            [str(e["case_id"]) for e in val_entries],
        )

    root = Path(data_root or mapping_doc.get("data_root", ""))
    splits_subdir = str(mapping_doc.get("splits_subdir", ""))
    if not root.is_dir() or not splits_subdir:
        raise ValueError(
            "case_mapping.json lacks train/val sections and data_root/splits_subdir "
            "are missing — cannot derive patient split."
        )

    train_keys = set(read_split_sample_keys(root, splits_subdir, "train"))
    val_keys = set(read_split_sample_keys(root, splits_subdir, "val"))
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError(f"Train/val split overlap ({len(overlap)} sample keys).")

    tr_ids: List[str] = []
    val_ids: List[str] = []
    unknown: List[str] = []
    for entry in mapping_doc.get("train_val") or []:
        sk = str(entry.get("sample_key", ""))
        cid = str(entry["case_id"])
        if sk in train_keys:
            tr_ids.append(cid)
        elif sk in val_keys:
            val_ids.append(cid)
        else:
            unknown.append(sk)

    if unknown:
        raise ValueError(
            f"{len(unknown)} case_mapping sample_key(s) not in train or val splits "
            f"(e.g. {unknown[:3]})."
        )
    if not tr_ids or not val_ids:
        raise ValueError(
            f"Empty patient split after partition: train={len(tr_ids)} val={len(val_ids)}"
        )
    return tr_ids, val_ids


def build_patient_splits_final(
    mapping_doc: Dict[str, Any],
    *,
    data_root: Path | None = None,
    num_folds: int = 1,
) -> List[Dict[str, List[str]]]:
    """nnU-Net ``splits_final.json`` payload (one patient-level fold, replicated if needed)."""
    tr_ids, val_ids = partition_case_ids_from_mapping(mapping_doc, data_root=data_root)
    fold = {"train": tr_ids, "val": val_ids}
    n = max(1, int(num_folds))
    return [dict(fold) for _ in range(n)]


def write_splits_final(
    preprocessed_dataset_folder_base: Path,
    splits: Sequence[Dict[str, List[str]]],
) -> Path:
    """Write ``splits_final.json`` next to ``nnUNetPlans_2d``."""
    out = Path(preprocessed_dataset_folder_base) / "splits_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(list(splits), f, indent=2)
    return out


def resolve_case_mapping_path(
    *,
    dataset_id: int,
    dataset_name: str,
) -> Path | None:
    env = os.environ.get("BAPMOS_NNUNET_CASE_MAPPING", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    raw = os.environ.get("nnUNet_raw", "").strip()
    if not raw:
        return None
    p = dataset_folder(Path(raw), dataset_id, dataset_name) / "case_mapping.json"
    return p if p.is_file() else None


def ensure_patient_splits_final(
    preprocessed_dataset_folder_base: Path,
    case_mapping_path: Path,
    *,
    fold: int = 0,
    num_folds: int = 1,
    data_root: Path | None = None,
    backfill_mapping: bool = True,
) -> Path:
    """Install patient-level ``splits_final.json`` (overwrites nnU-Net default 5-fold CV)."""
    with open(case_mapping_path, encoding="utf-8") as f:
        mapping_doc = json.load(f)

    root = data_root
    if root is None:
        dr = mapping_doc.get("data_root")
        if dr:
            root = Path(dr)

    if backfill_mapping and (not mapping_doc.get("train") or not mapping_doc.get("val")):
        backfill_case_mapping_train_val(case_mapping_path, data_root=root)

    with open(case_mapping_path, encoding="utf-8") as f:
        mapping_doc = json.load(f)

    splits = build_patient_splits_final(
        mapping_doc, data_root=root, num_folds=max(num_folds, fold + 1)
    )
    return write_splits_final(preprocessed_dataset_folder_base, splits)


def backfill_case_mapping_train_val(
    case_mapping_path: Path,
    *,
    data_root: Path | None = None,
) -> Tuple[int, int]:
    """Add ``train``/``val`` sections to an existing export (no image re-copy)."""
    with open(case_mapping_path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("train") and doc.get("val"):
        return len(doc["train"]), len(doc["val"])

    root = data_root
    if root is None:
        root = Path(doc.get("data_root", ""))
    tr_ids, val_ids = partition_case_ids_from_mapping(doc, data_root=root)
    by_id = {str(e["case_id"]): e for e in doc.get("train_val") or []}
    doc["train"] = [by_id[cid] for cid in tr_ids]
    doc["val"] = [by_id[cid] for cid in val_ids]
    doc["num_train"] = len(doc["train"])
    doc["num_val"] = len(doc["val"])
    with open(case_mapping_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return len(doc["train"]), len(doc["val"])
