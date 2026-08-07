"""
Export a BAP-MOS ``data/`` bundle to nnU-Net v2 **raw** dataset layout.

This does **not** run nnU-Net training inside this repo; install ``nnunetv2`` in a
separate environment (see ``requirements_nnunet.txt``), set ``nnUNet_raw``, then::

    nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
    nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerBapMosProtocol

``DATASET_ID`` is the integer you pass with ``--dataset_id`` (e.g. ``505`` maps to
``Dataset505_<name>`` for pooled prostate; PFUS1 uses ``503``).

**Training export (default):** only ``train`` + ``val`` splits go into ``imagesTr/``
and ``labelsTr/``. Do **not** merge the ``test`` split into training folds for paper
runs—doing so causes label leakage. For debugging only you can pass
``--splits train,val,test`` explicitly.

Images are written as single-channel grayscale PNGs under ``imagesTr/`` (must match
``channel_names`` / ``NaturalImage2DIO`` — do not save 3-channel BGR for 1-channel datasets);
labels are uint8 multiclass PNGs under ``labelsTr/`` with the same class IDs as
``*_combined_mask.png`` in this project.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

from bapmos.paths import (
    find_combined_masks_dir,
    find_training_images_dir,
    is_pfus1_advanced_training_root,
    is_pfus1_family_training_root,
    pfus1_advanced_image_root,
    pfus1_image_root,
)
from bapmos.pfus1.dataset_pfus1 import parse_sample_line
from bapmos.training_taxonomy import (
    default_splits_subdir,
    get_baseline_taxonomy_profile,
    resolve_training_data_root_path,
)


def _read_split_lines(data_root: Path, splits_subdir: str, split: str) -> List[str]:
    fp = data_root / splits_subdir / f"{split}.txt"
    if not fp.is_file():
        raise FileNotFoundError(f"Missing split file: {fp}")
    keys: List[str] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    return keys


def _collect_filenames(data_root: Path, splits_subdir: str, splits: List[str]) -> List[str]:
    """Union of split keys (sorted). Prefer per-split export for patient-level nnU-Net."""
    names: Set[str] = set()
    for sp in splits:
        names.update(_read_split_lines(data_root, splits_subdir, sp))
    return sorted(names)


def _labels_json_dict(taxonomy) -> dict:
    out = {"background": 0}
    for key, cid in sorted(taxonomy.organ_to_class.items(), key=lambda kv: kv[1]):
        out[str(key)] = int(cid)
    return out


def dataset_folder(output_parent: Path, dataset_id: int, dataset_name: str) -> Path:
    return output_parent / f"Dataset{dataset_id:03d}_{dataset_name}"


def _resolve_sample_paths(
    sample_key: str,
    *,
    pfus1_mode: bool,
    images_dir: Optional[Path],
    img_root: Optional[Path],
    masks_dir: Path,
) -> Tuple[Path, Path, str]:
    if pfus1_mode:
        patient, stem = parse_sample_line(sample_key)
        img_id = f"{patient}_{stem}"
        src_img = img_root / patient / f"{stem}.png"
        src_mask = masks_dir / f"{img_id}_combined_mask.png"
    else:
        img_name = sample_key if sample_key.endswith(".png") else f"{sample_key}.png"
        img_id = img_name.replace(".png", "")
        src_img = images_dir / img_name
        src_mask = masks_dir / f"{img_id}_combined_mask.png"
    return src_img, src_mask, img_id


def _write_grayscale_image(src_img: Path, dest: Path) -> None:
    """Write 1-channel PNG for nnU-Net ``NaturalImage2DIO`` (1 modality in dataset.json)."""
    gray = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Failed to read image: {src_img}")
    cv2.imwrite(str(dest), gray)


def export_test_images_ts(
    *,
    data_root: Path,
    ds_folder: Path,
    splits_subdir: str,
    pfus1_mode: bool,
    images_dir: Optional[Path],
    img_root: Optional[Path],
) -> List[Dict[str, Any]]:
    """Export ``test`` split to ``imagesTs/`` (no labels). Returns test case-mapping entries."""
    test_keys = _collect_filenames(data_root, splits_subdir, ["test"])
    images_ts = ds_folder / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for idx, sample_key in enumerate(test_keys):
        case_str = f"case{idx:05d}"
        src_img, _, img_id = _resolve_sample_paths(
            sample_key,
            pfus1_mode=pfus1_mode,
            images_dir=images_dir,
            img_root=img_root,
            masks_dir=find_combined_masks_dir(data_root),
        )
        if not src_img.is_file():
            raise FileNotFoundError(f"Missing test image: {src_img}")
        _write_grayscale_image(src_img, images_ts / f"{case_str}_0000.png")
        entries.append(
            {
                "case_id": case_str,
                "sample_key": sample_key,
                "image_id": img_id,
                "nnunet_image": f"{case_str}_0000.png",
            }
        )
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description="Export data/ bundle to nnUNet_raw")
    p.add_argument("--data_root", type=str, required=True, help="e.g. data/prostate/pooled")
    p.add_argument(
        "--output_parent",
        type=str,
        required=True,
        help="Folder that will contain DatasetXXX_Name (e.g. path to nnUNet_raw).",
    )
    p.add_argument(
        "--dataset_id",
        type=int,
        default=505,
        help="nnU-Net dataset id (pooled prostate default 505 -> Dataset505_*; PFUS1 typically 503)",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="BapMosExport",
        help="Suffix folder name (alphanumeric + underscore).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help=(
            "Comma-separated splits merged into nnU-Net's training folds (imagesTr/labelsTr). "
            "Default train,val avoids test leakage. Use e.g. 'test' alone for an imagesTs-only "
            "export, or 'train,val,test' only for non-benchmark debugging."
        ),
    )
    p.add_argument(
        "--splits_subdir",
        type=str,
        default=None,
        help="Split folder (default: taxonomy-specific, e.g. splits_patient_70_15_15_seed42 for PFUS1).",
    )
    p.add_argument(
        "--test_images_only",
        action="store_true",
        help=(
            "Only export the test split to imagesTs/ and update case_mapping.json (test section). "
            "Requires an existing Dataset folder from a prior train+val export."
        ),
    )
    args = p.parse_args()

    data_root = resolve_training_data_root_path(args.data_root)
    tax = get_baseline_taxonomy_profile(data_root)
    splits_subdir = args.splits_subdir or default_splits_subdir(data_root)
    masks_dir = find_combined_masks_dir(data_root)
    pfus1_mode = is_pfus1_family_training_root(data_root)
    if pfus1_mode:
        images_dir = None
        img_root = (
            pfus1_advanced_image_root()
            if is_pfus1_advanced_training_root(data_root)
            else pfus1_image_root()
        )
    else:
        images_dir = find_training_images_dir(data_root)
        img_root = None

    out_parent = Path(args.output_parent).expanduser().resolve()
    ds_folder = dataset_folder(out_parent, args.dataset_id, args.dataset_name)

    if args.test_images_only:
        if not ds_folder.is_dir():
            raise FileNotFoundError(
                f"--test_images_only requires existing dataset folder: {ds_folder}. "
                "Run a train+val export first."
            )
        test_entries = export_test_images_ts(
            data_root=data_root,
            ds_folder=ds_folder,
            splits_subdir=splits_subdir,
            pfus1_mode=pfus1_mode,
            images_dir=images_dir,
            img_root=img_root,
        )
        mapping_path = ds_folder / "case_mapping.json"
        mapping: Dict[str, Any] = {}
        if mapping_path.is_file():
            with open(mapping_path) as f:
                mapping = json.load(f)
        mapping["test"] = test_entries
        mapping["num_test"] = len(test_entries)
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Wrote {len(test_entries)} test cases to {ds_folder / 'imagesTs'}")
        print(f"Updated case mapping: {mapping_path}")
        return

    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    if ds_folder.exists():
        shutil.rmtree(ds_folder)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    train_val_entries: List[Dict[str, Any]] = []
    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []
    case_idx = 0
    for sp in split_list:
        for sample_key in _read_split_lines(data_root, splits_subdir, sp):
            src_img, src_mask, img_id = _resolve_sample_paths(
                sample_key,
                pfus1_mode=pfus1_mode,
                images_dir=images_dir,
                img_root=img_root,
                masks_dir=masks_dir,
            )
            if not src_img.is_file():
                raise FileNotFoundError(f"Missing image: {src_img}")
            if not src_mask.is_file():
                raise FileNotFoundError(f"Missing mask: {src_mask}")

            case_str = f"case{case_idx:05d}"
            case_idx += 1
            _write_grayscale_image(src_img, images_tr / f"{case_str}_0000.png")

            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Failed to read mask: {src_mask}")
            cv2.imwrite(str(labels_tr / f"{case_str}.png"), mask)
            entry = {
                "case_id": case_str,
                "sample_key": sample_key,
                "split": sp,
                "image_id": img_id,
                "nnunet_image": f"{case_str}_0000.png",
                "nnunet_label": f"{case_str}.png",
            }
            train_val_entries.append(entry)
            if sp == "train":
                train_entries.append(entry)
            elif sp == "val":
                val_entries.append(entry)

    sample_keys = [e["sample_key"] for e in train_val_entries]

    mapping_doc = {
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "data_root": str(data_root),
        "splits_subdir": splits_subdir,
        "train_val_splits": split_list,
        "train_val": train_val_entries,
        "train": train_entries,
        "val": val_entries,
        "num_train_val": len(train_val_entries),
        "num_train": len(train_entries),
        "num_val": len(val_entries),
    }
    with open(ds_folder / "case_mapping.json", "w") as f:
        json.dump(mapping_doc, f, indent=2)

    labels = _labels_json_dict(tax)
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": labels,
        "numTraining": len(sample_keys),
        "file_ending": ".png",
        "name": f"BAP-MOS {args.dataset_name} ({tax.taxonomy_name})",
        "description": f"Exported from {data_root} splits={split_list}",
        "reference": "BAP-MOS nnU-Net export",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "licence": "See source DICOM / cohort agreements.",
        "converted_by": "bapmos.external_baselines.nnunet2d.export_nnunet_dataset",
    }
    with open(ds_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Wrote nnU-Net v2 raw dataset: {ds_folder}")
    print(f"  cases: {len(sample_keys)}  labels: {list(labels.keys())}")
    print(f"  case_mapping.json: {ds_folder / 'case_mapping.json'}")
    print("\nNext (in nnUNet env), from project or any shell:")
    print(f"  export nnUNet_raw={out_parent}")
    print(
        f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity"
    )
    print(
        f"  nnUNetv2_train {args.dataset_id} 2d 0 -tr nnUNetTrainerBapMosProtocol"
    )
    print(
        "  # Protocol-constrained nnU-Net (stock Dice+CE loss; flat LR / patient splits via env)."
    )
    print(f"  # raw folder: {ds_folder.name}")


if __name__ == "__main__":
    main()
