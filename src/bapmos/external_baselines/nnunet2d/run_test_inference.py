"""Re-export nnU-Net test predictions into the inference_output bundle (metrics + viz).

Pooled prostate: one folder per ``site_tests/<site>/`` under ``output_dir`` (same
layout as other methods). Pass ``--splits-subdir site_tests/<site>`` to export a
single site after a per-site ``nnUNetv2_predict``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.external_baselines.baseline_training_protocol import PFUS1_BASELINE_EVAL_SIZE
from bapmos.external_baselines.nnunet2d.evaluate_nnunet_predictions import (
    align_pred_to_gt,
    find_prediction_png,
    load_split_entries,
)
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.inference_output.site_export import export_stratified_sites, resolve_splits_subdir
from bapmos.paths import resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile


def _image_id_to_case_id(case_mapping_path: Path, split: str = "test") -> Dict[str, str]:
    entries = load_split_entries(case_mapping_path, split)
    out: Dict[str, str] = {}
    for entry in entries:
        image_id = entry.get("image_id") or entry["sample_key"].replace(".png", "")
        fn_key = entry["sample_key"] if entry["sample_key"].endswith(".png") else f"{entry['sample_key']}.png"
        out[fn_key] = entry["case_id"]
        out[image_id] = entry["case_id"]
    return out


def run_nnunet_test_export(
    *,
    predictions_dir: Path,
    case_mapping_path: Path,
    data_root: Path,
    output_dir: Path,
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    export_viz_kwargs: Optional[Dict[str, Any]] = None,
    evaluation_meta_extra: Optional[Dict[str, Any]] = None,
    eval_size: Optional[int] = PFUS1_BASELINE_EVAL_SIZE,
    force_inference_output: bool = False,
    experiment_seed: Optional[int] = None,
    run_name: Optional[str] = None,
) -> None:
    data_root = Path(resolve_under_project(data_root))
    predictions_dir = Path(resolve_under_project(predictions_dir))
    output_dir = Path(resolve_under_project(output_dir))
    case_mapping_path = Path(resolve_under_project(case_mapping_path))

    require_primary_inference_seed_run(
        experiment_seed=experiment_seed,
        run_name=run_name,
        force=force_inference_output,
    )

    splits = resolve_splits_subdir(None, splits_subdir, data_root)
    id_to_case = _image_id_to_case_id(case_mapping_path, split="test")
    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)

    def predict_fn(image_rgb, mask_multi, fn: str):
        del image_rgb
        case_id = id_to_case.get(fn) or id_to_case.get(Path(fn).stem)
        if case_id is None:
            raise KeyError(f"No nnU-Net case_id for test filename {fn!r}")
        pred_path = find_prediction_png(predictions_dir, case_id)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise IOError(f"Failed to read prediction: {pred_path}")
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        if hasattr(mask_multi, "numpy"):
            mask_multi = mask_multi.numpy()
        gt = mask_multi.astype(np.uint8)
        return align_pred_to_gt(pred.astype(np.uint8), gt)

    viz_kwargs = dict(
        export_viz_kwargs
        or export_viz_kwargs_for_data_root(
            str(data_root), per_patient_max=viz_per_patient_max
        )
    )
    # Ensure first-N image+PDF protocol even when caller overrides other viz kwargs.
    viz_kwargs.setdefault("viz_pdf_max", 10)
    viz_kwargs.setdefault("save_prediction_images_max", viz_kwargs["viz_pdf_max"])
    viz_kwargs.setdefault("viz_selection", "all")

    base_extra = dict(evaluation_meta_extra or {})
    base_extra.setdefault("family", "nnunet")
    base_extra.setdefault("predictions_dir", str(predictions_dir))
    base_extra.setdefault("case_mapping", str(case_mapping_path))
    if eval_size is not None:
        base_extra.setdefault("eval_size", int(eval_size))

    export_stratified_sites(
        data_root=data_root,
        output_dir=output_dir,
        splits_subdir=splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=unit,
        checkpoint=str(predictions_dir),
        method_slug="nnunet",
        eval_size=eval_size,
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {**base_extra, "test_site": label},
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export nnU-Net test split to inference_output/ "
            "(first-10 pred+panel PNG/PDF @ 350 dpi; pooled prostate → per-site folders)."
        )
    )
    p.add_argument("--predictions_dir", type=str, required=True)
    p.add_argument("--case_mapping", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits-subdir", type=str, default=None)
    p.add_argument("--viz-per-patient-max", type=int, default=None, help="Deprecated (ignored).")
    p.add_argument("--viz-pdf-max", type=int, default=None)
    p.add_argument(
        "--eval-size",
        type=int,
        default=PFUS1_BASELINE_EVAL_SIZE,
        help=(
            "Square side for nearest-neighbor pred/GT resize before metrics "
            f"(default {PFUS1_BASELINE_EVAL_SIZE}, U-Net/SAM parity). Use 0 for native resolution."
        ),
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--experiment-seed", type=int, default=42)
    p.add_argument("--force-inference-output", action="store_true")
    args = p.parse_args()

    eval_size = None if args.eval_size in (None, 0) else int(args.eval_size)
    viz = export_viz_kwargs_for_data_root(
        args.data_root, viz_pdf_max=args.viz_pdf_max
    )

    run_nnunet_test_export(
        predictions_dir=Path(args.predictions_dir),
        case_mapping_path=Path(args.case_mapping),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        splits_subdir=args.splits_subdir,
        viz_per_patient_max=args.viz_per_patient_max,
        export_viz_kwargs=viz,
        eval_size=0 if eval_size is None else eval_size,
        force_inference_output=bool(args.force_inference_output),
        experiment_seed=args.experiment_seed,
        run_name=args.run_name,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
