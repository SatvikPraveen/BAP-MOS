#!/usr/bin/env python3
"""Build pooled prostate training corpus under ``data/prostate/pooled``.

Policy (Option 1):
  train = sim_train ∪ case1_train ∪ case2_train
  val   = sim_val ∪ case1_val ∪ case2_val
  test  = unchanged per-site lists under ``site_tests/<site>/test.txt``

Simulation slices are resampled to clinical spacing and masks remapped to the
clinical four-organ taxonomy (sim rectum/bladder/ptv1 → clinical rectum/bladder/ptv).
Clinical cohorts are linked (symlink) into the pool; simulation assets are materialized.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

from bapmos.paths import (
    find_combined_masks_dir,
    find_training_images_dir,
    pooled_prostate_dataset_dir,
    project_root,
    real_case_dataset_dir,
    simulation_dataset_dir,
)
from bapmos.training_taxonomy import (
    PIXEL_SPACING_CLINICAL_MM,
    PIXEL_SPACING_SIMULATION_MM,
)

SPACING_SIM_MM = PIXEL_SPACING_SIMULATION_MM[0]
SPACING_CLIN_MM = PIXEL_SPACING_CLINICAL_MM[0]
RESAMPLE_SCALE = SPACING_SIM_MM / SPACING_CLIN_MM

from bapmos.preprocess.prostate.spacing import (
    SPACING_CONTRACT_FILE,
    build_spacing_contract_dict,
)
SIM_CLASS_TO_CLINICAL = np.array([0, 3, 1, 2], dtype=np.uint8)

COHORTS: Tuple[Tuple[str, Path, bool], ...] = (
    ("simulation", simulation_dataset_dir(), True),
    ("case1", real_case_dataset_dir("case1"), False),
    ("case2", real_case_dataset_dir("case2"), False),
)


def _read_split_lines(split_file: Path) -> List[str]:
    lines: List[str] = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line.split()[0])
    return lines


def _mask_stem(image_name: str) -> str:
    stem = Path(image_name).stem
    if stem.endswith("_combined_mask"):
        return stem[: -len("_combined_mask")]
    return stem


def _combined_mask_path(mask_dir: Path, image_name: str) -> Path:
    stem = _mask_stem(image_name)
    return mask_dir / f"{stem}_combined_mask.png"


def _collect_all_filenames(cohort_root: Path) -> Set[str]:
    splits = cohort_root / "splits_stratified"
    names: Set[str] = set()
    for split in ("train", "val", "test"):
        fp = splits / f"{split}.txt"
        if fp.is_file():
            names.update(_read_split_lines(fp))
    return names


def _remap_sim_mask(mask: np.ndarray) -> np.ndarray:
    return SIM_CLASS_TO_CLINICAL[mask.astype(np.int64)].astype(mask.dtype)


def _resample_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * RESAMPLE_SCALE)))
    new_h = max(1, int(round(h * RESAMPLE_SCALE)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _resample_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    new_w = max(1, int(round(w * RESAMPLE_SCALE)))
    new_h = max(1, int(round(h * RESAMPLE_SCALE)))
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _link_or_copy(src: Path, dst: Path, *, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _materialize_sim_sample(
    image_name: str,
    src_images: Path,
    src_masks: Path,
    dst_images: Path,
    dst_masks: Path,
) -> None:
    src_img = src_images / image_name
    src_msk = _combined_mask_path(src_masks, image_name)
    dst_img = dst_images / image_name
    dst_msk = dst_masks / f"{_mask_stem(image_name)}_combined_mask.png"
    if not src_img.is_file():
        raise FileNotFoundError(src_img)
    if not src_msk.is_file():
        raise FileNotFoundError(src_msk)

    img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_img}")
    msk = cv2.imread(str(src_msk), cv2.IMREAD_UNCHANGED)
    if msk is None:
        raise RuntimeError(f"Failed to read mask: {src_msk}")
    if msk.ndim == 3:
        msk = msk[:, :, 0]

    img_out = _resample_image(img)
    msk_out = _remap_sim_mask(_resample_mask(msk))

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_msk.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_img), img_out):
        raise RuntimeError(f"Failed to write {dst_img}")
    if not cv2.imwrite(str(dst_msk), msk_out):
        raise RuntimeError(f"Failed to write {dst_msk}")


def _validate_sim_resample(
    *,
    sample_name: str,
    src_images: Path,
    dst_images: Path,
) -> Dict[str, Any]:
    """Confirm one simulation slice was shrunk to clinical spacing in the pool."""
    src = cv2.imread(str(src_images / sample_name), cv2.IMREAD_UNCHANGED)
    dst = cv2.imread(str(dst_images / sample_name), cv2.IMREAD_UNCHANGED)
    if src is None or dst is None:
        raise RuntimeError(f"Resample validation failed for {sample_name!r}")

    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    expected_w = max(1, int(round(sw * RESAMPLE_SCALE)))
    expected_h = max(1, int(round(sh * RESAMPLE_SCALE)))

    if abs(dw - expected_w) > 1 or abs(dh - expected_h) > 1:
        raise RuntimeError(
            f"Simulation resample mismatch for {sample_name}: "
            f"src={sw}x{sh} dst={dw}x{dh} expected~{expected_w}x{expected_h} "
            f"(scale={RESAMPLE_SCALE:.6f})"
        )

    native_mm = SPACING_SIM_MM
    effective_mm = SPACING_CLIN_MM
    physical_w_native = sw * native_mm
    physical_w_pooled = dw * effective_mm
    rel_err = abs(physical_w_native - physical_w_pooled) / max(physical_w_native, 1e-6)

    return {
        "sample": sample_name,
        "src_pixels": [sw, sh],
        "dst_pixels": [dw, dh],
        "expected_pixels": [expected_w, expected_h],
        "native_spacing_mm": native_mm,
        "pool_spacing_mm": effective_mm,
        "physical_width_mm_native": round(physical_w_native, 4),
        "physical_width_mm_pooled": round(physical_w_pooled, 4),
        "physical_width_relative_error": round(rel_err, 6),
    }


def _concat_splits(
    out_splits: Path,
    cohort_splits: Dict[str, Dict[str, List[str]]],
    *,
    which: str,
) -> List[str]:
    merged: List[str] = []
    for splits in cohort_splits.values():
        for name in splits[which]:
            merged.append(name)
    out_file = out_splits / f"{which}.txt"
    out_file.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    return merged


def build_pooled(
    *,
    use_symlinks: bool = True,
    force: bool = False,
    out_root: Path | None = None,
) -> Path:
    out_root = Path(out_root) if out_root is not None else pooled_prostate_dataset_dir()
    if force and out_root.exists():
        shutil.rmtree(out_root)

    out_images = out_root / "images"
    out_masks = out_root / "masks" / "combined_masks"
    out_splits = out_root / "splits_stratified"
    site_tests_root = out_root / "site_tests"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)
    site_tests_root.mkdir(parents=True, exist_ok=True)

    cohort_splits: Dict[str, Dict[str, List[str]]] = {}
    counts: Dict[str, int] = {}
    sim_src_images: Path | None = None
    sim_sample_for_validation: str | None = None

    for cohort_name, cohort_root, is_sim in COHORTS:
        splits_dir = cohort_root / "splits_stratified"
        cohort_splits[cohort_name] = {
            split: _read_split_lines(splits_dir / f"{split}.txt")
            for split in ("train", "val", "test")
        }

        src_images = find_training_images_dir(cohort_root)
        src_masks = find_combined_masks_dir(cohort_root)
        all_names = _collect_all_filenames(cohort_root)
        counts[cohort_name] = len(all_names)
        if is_sim:
            sim_src_images = src_images
            sim_sample_for_validation = sorted(all_names)[0]

        for image_name in sorted(all_names):
            if is_sim:
                _materialize_sim_sample(
                    image_name, src_images, src_masks, out_images, out_masks
                )
            else:
                src_img = src_images / image_name
                src_msk = _combined_mask_path(src_masks, image_name)
                _link_or_copy(src_img, out_images / image_name, use_symlinks=use_symlinks)
                _link_or_copy(
                    src_msk,
                    out_masks / src_msk.name,
                    use_symlinks=use_symlinks,
                )

        site_dir = site_tests_root / cohort_name
        site_dir.mkdir(parents=True, exist_ok=True)
        test_lines = cohort_splits[cohort_name]["test"]
        (site_dir / "test.txt").write_text(
            "\n".join(test_lines) + ("\n" if test_lines else ""),
            encoding="utf-8",
        )

    train_n = _concat_splits(out_splits, cohort_splits, which="train")
    val_n = _concat_splits(out_splits, cohort_splits, which="val")

    resample_check: Dict[str, Any] | None = None
    if sim_src_images is not None and sim_sample_for_validation is not None:
        resample_check = _validate_sim_resample(
            sample_name=sim_sample_for_validation,
            src_images=sim_src_images,
            dst_images=out_images,
        )

    spacing_contract = build_spacing_contract_dict()
    if resample_check is not None:
        spacing_contract["resample_validation"] = resample_check
    (out_root / SPACING_CONTRACT_FILE).write_text(
        json.dumps(spacing_contract, indent=2), encoding="utf-8"
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "train=union(train); val=union(val); test=per-site unchanged",
        "spacing_mm": SPACING_CLIN_MM,
        "native_simulation_spacing_mm": SPACING_SIM_MM,
        "simulation_resampled_to_clinical_spacing": True,
        "spacing_contract": SPACING_CONTRACT_FILE,
        "simulation_resample_scale": RESAMPLE_SCALE,
        "sim_to_clinical_class_map": SIM_CLASS_TO_CLINICAL.tolist(),
        "cohort_unique_slice_counts": counts,
        "train_slices": len(train_n),
        "val_slices": len(val_n),
        "site_test_slices": {
            c: len(s["test"]) for c, s in cohort_splits.items()
        },
        "clinical_materialization": "symlink" if use_symlinks else "copy",
    }
    (out_splits / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    manifest = {
        **summary,
        "data_root": str(out_root.resolve()),
        "repo_root": str(project_root().resolve()),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Spacing: simulation native {SPACING_SIM_MM} mm/px → pool {SPACING_CLIN_MM} mm/px "
        f"(scale {RESAMPLE_SCALE:.6f})"
    )
    if resample_check is not None:
        print(f"Resample check: {resample_check['sample']} physical width error "
              f"{resample_check['physical_width_relative_error']:.4%}")
    return out_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.prostate",
        description="Build pooled prostate dataset under data/prostate/pooled/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output pooled root (default: <BAPMOS>/data/prostate/pooled)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy clinical files instead of symlinking (default: symlink)",
    )
    p.add_argument("--force", action="store_true", help="Delete existing pool before rebuild")
    p.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    args = p.parse_args(argv)
    target = args.out or pooled_prostate_dataset_dir()
    print("Prostate pooled preprocess")
    print(f"  out: {Path(target).resolve()}")
    print("  steps: resample simulation → clinical spacing; link case1/case2; write splits + spacing_contract.json")
    if args.dry_run:
        print("  dry-run: no files written")
        return 0
    out = build_pooled(
        use_symlinks=not args.copy,
        force=args.force,
        out_root=target,
    )
    print(f"Pooled dataset ready: {out}")
    print(f"Manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
