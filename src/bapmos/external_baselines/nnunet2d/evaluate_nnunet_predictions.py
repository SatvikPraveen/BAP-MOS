"""
Evaluate nnU-Net 2D multiclass predictions with the same ``MetricsEvaluator`` as U-Net / BAP-MOS.

Expects:
- ``case_mapping.json`` from ``export_nnunet_dataset`` (``test`` section populated via ``--test_images_only``)
- nnU-Net prediction PNGs under ``--predictions_dir`` (one label map per case, uint8 class ids)
- Ground-truth combined masks under the PFUS1 (or case) ``data/`` bundle

Writes under ``--output_dir`` (default: ``.../test_results/``):
  summary_metrics.csv, per_slice_metrics.csv, per_organ_metrics.csv, failure_analysis.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from bapmos.external_baselines.baseline_training_protocol import (
    PFUS1_BASELINE_EVAL_SIZE,
    resolve_boundary_eval_size,
)
from bapmos.external_baselines.common import resize_masks_nearest
from bapmos.external_baselines.nnunet2d.export_nnunet_dataset import dataset_folder
from bapmos.paths import (
    dataset_bundle_tag,
    find_combined_masks_dir,
    is_pfus1_family_training_root,
    method_test_output_dir,
    project_root,
    write_method_evaluation_meta,
)
from bapmos.pfus1.dataset_pfus1 import parse_sample_line
from bapmos.training_taxonomy import (
    get_baseline_taxonomy_profile,
    resolve_training_data_root_path,
)
from bapmos.baseline_validation_metrics import organ_balanced_validation_summary
from bapmos.evaluation.baseline_epoch_monitoring import (
    build_test_wandb_log,
    per_organ_wandb_dict,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator


def _find_prediction_png(predictions_dir: Path, case_id: str) -> Path:
    candidates = [
        predictions_dir / f"{case_id}.png",
        predictions_dir / f"{case_id}_0000.png",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No prediction for {case_id} under {predictions_dir} (tried: {[p.name for p in candidates]})"
    )


def _load_split_entries(case_mapping_path: Path, split: str) -> List[Dict[str, Any]]:
    split = split.strip().lower()
    if split not in ("test", "val", "train"):
        raise ValueError(f"split must be 'test', 'val', or 'train', got {split!r}")
    with open(case_mapping_path) as f:
        doc = json.load(f)
    entries = doc.get(split)
    if not entries:
        raise ValueError(
            f"{case_mapping_path} has no {split!r} entries. "
            "For test: run export with --test_images_only after train+val export. "
            "For train: use --case_ids or ensure train section exists in case_mapping.json."
        )
    return entries


def case_entries_for_ids(case_mapping_path: Path, case_ids: List[str]) -> List[Dict[str, Any]]:
    """Resolve nnU-Net case IDs to mapping entries (train_val pool or explicit split sections)."""
    with open(case_mapping_path) as f:
        doc = json.load(f)
    by_case: Dict[str, Dict[str, Any]] = {}
    for section in ("train_val", "train", "val", "test"):
        for entry in doc.get(section) or []:
            by_case[str(entry["case_id"])] = entry
    missing = [cid for cid in case_ids if cid not in by_case]
    if missing:
        raise KeyError(
            f"{len(missing)} case_id(s) missing from {case_mapping_path}: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    return [by_case[cid] for cid in case_ids]


def evaluate_nnunet_predictions_for_entries(
    *,
    predictions_dir: Path,
    entries: List[Dict[str, Any]],
    data_root: Path,
    output_dir: Path,
    split_label: str = "val",
    eval_size: Optional[int] = PFUS1_BASELINE_EVAL_SIZE,
) -> tuple[MetricsEvaluator, Dict[str, Any]]:
    """Evaluate nnU-Net PNG predictions for an explicit list of case_mapping entries."""
    data_root = resolve_training_data_root_path(str(data_root))
    tax = get_baseline_taxonomy_profile(data_root)
    masks_dir = find_combined_masks_dir(data_root)
    class_mapping = {int(k): v for k, v in tax.multiclass_eval_mapping.items()}
    metric_eval_size = resolve_boundary_eval_size(eval_size)

    evaluator = MetricsEvaluator(
        pixel_spacing=tuple(tax.pixel_spacing_mm),
        organs=list(tax.evaluator_organ_labels),
    )

    predictions_dir = predictions_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    n_eval = 0
    for entry in entries:
        case_id = entry["case_id"]
        image_id = entry.get("image_id") or entry["sample_key"]
        pred_path = _find_prediction_png(predictions_dir, case_id)
        gt_path = _gt_mask_path(data_root, entry, masks_dir)
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT mask: {gt_path}")

        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            raise IOError(f"Failed to read pred={pred_path} or gt={gt_path}")
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        pred = _align_pred_to_gt(pred.astype(np.uint8), gt.astype(np.uint8))
        gt = gt.astype(np.uint8)
        if metric_eval_size is not None:
            pred, gt = resize_masks_nearest(pred, gt, int(metric_eval_size))

        evaluator.evaluate_multiclass_slice(
            pred, gt, slice_idx=0, image_id=image_id, class_mapping=class_mapping
        )
        n_eval += 1

    evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
    evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
    evaluator.export_failure_analysis_csv(metrics_dir / "failure_analysis.csv", top_n=20)
    export_per_organ_metrics_csv(
        evaluator, list(tax.evaluator_organ_labels), metrics_dir / "per_organ_metrics.csv"
    )

    split = split_label.strip().lower()
    meta = {
        "baseline": "nnunetv2_2d",
        "predictions_dir": str(predictions_dir),
        "data_root": str(data_root),
        "taxonomy": tax.taxonomy_name,
        "distance_unit": "px" if tax.taxonomy_name == "pfus1_eight_organ" else "mm",
        "pixel_spacing": list(tax.pixel_spacing_mm),
        "split": split,
        f"n_{split}_cases": n_eval,
        "eval_size": int(metric_eval_size) if metric_eval_size is not None else 0,
    }
    with open(output_dir / "evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary = organ_balanced_validation_summary(evaluator)
    split_results = {
        f"{split}_msd_mm": summary.get("val_msd"),
        f"{split}_hd95_mm": summary.get("val_hd95"),
        f"{split}_dice_organ_balanced": summary.get("val_dice"),
        f"n_{split}_cases": n_eval,
        "taxonomy": tax.taxonomy_name,
        "split": split,
    }
    if split == "test":
        split_results.update(
            {
                "test_msd_mm": split_results[f"{split}_msd_mm"],
                "test_hd95_mm": split_results[f"{split}_hd95_mm"],
                "test_dice_organ_balanced": split_results[f"{split}_dice_organ_balanced"],
                "n_test_cases": n_eval,
            }
        )
    results_path = output_dir / (
        "test_results.json" if split == "test" else f"{split}_results.json"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(split_results, f, indent=2)

    print(f"Evaluated {n_eval} {split} cases → {output_dir}")
    return evaluator, split_results


def _gt_mask_path(data_root: Path, entry: Dict[str, Any], masks_dir: Path) -> Path:
    sample_key = entry["sample_key"]
    if is_pfus1_family_training_root(data_root):
        patient, stem = parse_sample_line(sample_key)
        img_id = f"{patient}_{stem}"
    else:
        img_id = entry.get("image_id") or sample_key.replace(".png", "")
    return masks_dir / f"{img_id}_combined_mask.png"


def _align_pred_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    return cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)


# Public aliases for shared use by ``run_test_inference`` (stable names).
find_prediction_png = _find_prediction_png
load_split_entries = _load_split_entries
gt_mask_path = _gt_mask_path
align_pred_to_gt = _align_pred_to_gt


def export_per_organ_metrics_csv(evaluator: MetricsEvaluator, organs: List[str], output_path: Path) -> None:
    rows = []
    for organ in organs:
        agg = evaluator.aggregate_metrics(organ)
        if agg is not None:
            rows.append(agg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Per-organ metrics saved to: {output_path}")


def evaluate_nnunet_split_predictions(
    *,
    predictions_dir: Path,
    case_mapping_path: Path,
    data_root: Path,
    output_dir: Path,
    split: str = "test",
    eval_size: Optional[int] = PFUS1_BASELINE_EVAL_SIZE,
) -> tuple[MetricsEvaluator, Dict[str, Any]]:
    split_entries = _load_split_entries(case_mapping_path, split)
    return evaluate_nnunet_predictions_for_entries(
        predictions_dir=predictions_dir,
        entries=split_entries,
        data_root=data_root,
        output_dir=output_dir,
        split_label=split,
        eval_size=eval_size,
    )


def evaluate_nnunet_test_predictions(
    *,
    predictions_dir: Path,
    case_mapping_path: Path,
    data_root: Path,
    output_dir: Path,
    eval_size: Optional[int] = PFUS1_BASELINE_EVAL_SIZE,
) -> tuple[MetricsEvaluator, Dict[str, Any]]:
    """Backward-compatible wrapper for held-out test evaluation."""
    return evaluate_nnunet_split_predictions(
        predictions_dir=predictions_dir,
        case_mapping_path=case_mapping_path,
        data_root=data_root,
        output_dir=output_dir,
        split="test",
        eval_size=eval_size,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate nnU-Net 2D predictions on held-out test split")
    p.add_argument("--predictions_dir", type=str, required=True, help="Folder of nnUNetv2_predict PNG outputs")
    p.add_argument(
        "--case_mapping",
        type=str,
        default=None,
        help="Path to case_mapping.json (default: under nnUNet_raw dataset folder)",
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--nnunet_raw_parent",
        type=str,
        default=None,
        help="Parent of DatasetXXX_* when --case_mapping omitted",
    )
    p.add_argument("--dataset_id", type=int, default=503)
    p.add_argument("--dataset_name", type=str, default="BapMosPfus1TrainVal")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Default: output/<dataset>/nn_unet/ for case1/case2/simulation; "
            "else runs/pfus1/ExternalBaselines/nnunet2d/<run_name>/test_results"
        ),
    )
    p.add_argument(
        "--method-slug",
        type=str,
        default="nn_unet",
        help="Flat folder under output/<dataset>/ when using PTV layout (default: nn_unet).",
    )
    p.add_argument("--run_name", type=str, default="nnunet_pfus1_2d_fold0")
    p.add_argument(
        "--run_root",
        type=str,
        default="runs/pfus1/ExternalBaselines",
        help="Parent for nnunet2d/<run_name>/test_results",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("test", "val"),
        default="test",
        help="Case-mapping split to evaluate (default: test).",
    )
    p.add_argument(
        "--eval-size",
        type=int,
        default=PFUS1_BASELINE_EVAL_SIZE,
        help=(
            "Square side for nearest-neighbor pred/GT resize before metrics "
            f"(default {PFUS1_BASELINE_EVAL_SIZE}, U-Net/SAM/BAP-MOS parity). "
            "Use 0 for native resolution."
        ),
    )
    p.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log split MSD/HD95/dice + per-organ keys to W&B (requires wandb login)",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project when --log-wandb (default: env nnUNet_wandb_project or nnunet_test)",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: --run_name + _test_eval)",
    )
    args = p.parse_args()

    pr = project_root()
    if args.case_mapping:
        mapping_path = Path(args.case_mapping).expanduser()
        if not mapping_path.is_absolute():
            mapping_path = (pr / mapping_path).resolve()
    else:
        raw_parent = args.nnunet_raw_parent or "exports/nnUNet_raw/pfus1"
        raw_p = Path(raw_parent)
        if not raw_p.is_absolute():
            raw_p = (pr / raw_p).resolve()
        mapping_path = dataset_folder(raw_p, args.dataset_id, args.dataset_name) / "case_mapping.json"

    data_root_resolved = resolve_training_data_root_path(str(args.data_root))
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (pr / out_dir).resolve()
    elif dataset_bundle_tag(data_root_resolved) in ("case_1", "case_2", "simulation"):
        out_dir = method_test_output_dir(data_root_resolved, args.method_slug, repo_root=pr)
    else:
        rr = Path(args.run_root)
        base = rr if rr.is_absolute() else (pr / rr)
        out_dir = (base / "nnunet2d" / args.run_name / "test_results").resolve()

    pred_dir = Path(args.predictions_dir).expanduser()
    if not pred_dir.is_absolute():
        pred_dir = (pr / pred_dir).resolve()

    split = args.split
    evaluator, split_results = evaluate_nnunet_split_predictions(
        predictions_dir=pred_dir,
        case_mapping_path=mapping_path,
        data_root=Path(data_root_resolved),
        output_dir=out_dir,
        split=split,
        eval_size=args.eval_size,
    )
    write_method_evaluation_meta(
        out_dir,
        checkpoint=str(pred_dir),
        data_root=data_root_resolved,
        method_slug=args.method_slug,
        split=split,
        extra={"baseline": "nnunetv2_2d", "run_name": args.run_name},
    )
    organ_labels = list(
        get_baseline_taxonomy_profile(
            resolve_training_data_root_path(str(args.data_root))
        ).evaluator_organ_labels
    )

    if args.log_wandb:
        try:
            import os

            import wandb
        except ImportError as exc:
            raise RuntimeError("Install wandb to use --log-wandb") from exc
        project = (
            args.wandb_project
            or os.environ.get("nnUNet_wandb_project")
            or "nnunet_test"
        )
        run_name = args.wandb_run_name or f"{args.run_name}_{split}_eval"
        wandb.init(project=project, name=run_name, dir=str(out_dir.parent))
        if split == "test":
            wandb_result = {
                "test_msd_mm": split_results.get("test_msd_mm"),
                "test_hd95_mm": split_results.get("test_hd95_mm"),
                "test_dice_organ_balanced": split_results.get("test_dice_organ_balanced"),
            }
            log = build_test_wandb_log(wandb_result, evaluator, organ_labels)
        else:
            log = {}
            for key, wandb_key in (
                ("val_msd_mm", "val/msd_mm"),
                ("val_hd95_mm", "val/hd95_mm"),
                ("val_dice_organ_balanced", "val/dice_organ_balanced"),
            ):
                v = split_results.get(key)
                if v is not None:
                    log[wandb_key] = float(v)
            log.update(per_organ_wandb_dict(evaluator, organ_labels, "val"))
        if log:
            wandb.log(log)
            wandb.summary.update({k.replace("/", "_"): v for k, v in log.items()})
        wandb.finish()
        print(f"W&B {split} metrics logged to project={project} run={run_name}")


if __name__ == "__main__":
    main()
