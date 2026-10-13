"""Write stratified test predictions, TP/FP/FN visualizations, and metrics."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

from bapmos.evaluation.test_panels import (
    multiclass_mask_to_bgr,
    save_overlay_pred_pair_png_pdf,
)
from bapmos.evaluation.difference_v1 import save_difference_v1_panel
from bapmos.pdf_export import INFERENCE_PANEL_PDF_DPI
from bapmos.evaluation.viz_index import write_visualization_index_csv
from bapmos.evaluation.viz_selection import (
    SliceVizRecord,
    aggregate_slice_metrics_for_image,
    select_slices_for_visualization,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.metrics.evaluation_grid import EVALUATION_GRID, to_reference_grid
from bapmos.training_taxonomy import BaselineTaxonomyProfile


# ``predict_fn(image_rgb, mask_multi, filename)`` → multiclass label map or None. The map may
# be on the model's own grid; the bundle upsamples it to the GT grid. ``mask_multi`` is passed
# only so prompt-based methods can build prompts -- never as a label for the prediction.
PredictFn = Callable[[Any, Any, str], Optional[np.ndarray]]


def _batch_filename_stem(filename: str) -> str:
    """Filesystem-safe stem for prediction / visualization artifact names.

    ``batch['filename']`` is typically a PNG basename (prostate) or an
    extensionless sample id (PFUS1 ``Pxxx_frame_yyy``). Path separators are
    collapsed so outputs never nest under ``predictions/``.
    """
    s = str(filename).strip().replace("\\", "/")
    if "/" in s:
        s = s.replace("/", "_")
    return Path(s).stem


def _csv_metric_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def export_stratified_test_bundle(
    *,
    test_loader,
    predict_fn: PredictFn,
    output_dir: Path,
    taxonomy: BaselineTaxonomyProfile,
    distance_unit: str,
    save_predictions: bool = True,
    save_prediction_images_max: Optional[int] = None,
    save_visualizations: bool = True,
    save_viz_pdf: bool = True,
    viz_selection: str = "all",
    viz_max: Optional[int] = None,
    viz_pdf_max: Optional[int] = None,
    viz_pdf_dpi: int = INFERENCE_PANEL_PDF_DPI,
    viz_per_patient_max: Optional[int] = None,
    viz_seed: int = 42,
    evaluation_meta_extra: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str | Path] = None,
    data_root: Optional[str | Path] = None,
    method_slug: Optional[str] = None,
    record_e2e_timing: Optional[bool] = None,
) -> MetricsEvaluator:
    """
    Run test split inference and write the standard inference_output bundle::

        predictions/multiclass/*_pred_ids.png   # sibling of visualizations/
        predictions/*_pred.png
        visualizations/*_viz.png|.pdf           # matplotlib panels
        visualizations/difference/*_diff.png|.pdf  # difference_v1
        metrics/summary_metrics.csv
        metrics/per_slice_metrics.csv

    Only slices with at least one foreground GT pixel are evaluated and
    considered for image export (pure-background masks are skipped). Artifact
    basenames use :func:`_batch_filename_stem` so ``.png`` is not duplicated
    and path separators cannot nest under ``predictions/``.

    ``predict_fn(image_rgb, mask_multi, filename)`` must return a multiclass
    label map on any grid (or ``None`` to skip). The GT ``mask_multi`` is
    provided only for prompt construction, not as a supervision signal.

    Every method is scored on the native GT grid with the taxonomy's physical
    pixel spacing (:mod:`bapmos.metrics.evaluation_grid`): a coarser prediction is
    nearest-neighbour upsampled to the GT shape, and the GT is never resampled.

    Difference panels use ``difference_v1`` (neutral grey canvas), not the older
    ultrasound-underlay difference blend. Overlay / difference PNG and PDF are
    separate matplotlib ``savefig`` calls from in-memory RGB at ``viz_pdf_dpi``
    (default 350).

    When ``viz_pdf_max`` is set (default protocol: 10), only the first N
    evaluated slices in loader order receive panel PNG **and** PDF. When
    ``save_prediction_images_max`` is set, colored / id prediction PNGs are
    limited to that same first-N budget. Metrics still cover every evaluated
    slice.

    Non-``simple_viz`` selection modes iterate the loader twice (metrics pass,
    then selected-slice visualization) and re-run ``predict_fn``; acceptable when
    inference is cheap relative to I/O.
    """
    from bapmos.paths import write_method_evaluation_meta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = output_dir / "predictions"
    pred_ids_dir = pred_dir / "multiclass"
    viz_dir = output_dir / "visualizations"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_ids_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        viz_dir.mkdir(parents=True, exist_ok=True)

    simple_viz = (
        save_visualizations
        and viz_selection == "all"
        and viz_max is None
        and viz_per_patient_max is None
    )
    evaluator = MetricsEvaluator(
        pixel_spacing=tuple(taxonomy.pixel_spacing_mm),
        organs=list(taxonomy.evaluator_organ_labels),
    )
    viz_records: List[SliceVizRecord] = []
    idx_rows: List[Dict[str, Any]] = []
    class_mapping = {int(k): v for k, v in taxonomy.multiclass_eval_mapping.items()}
    viz_png_count = 0
    pred_img_count = 0
    if record_e2e_timing is None:
        record_e2e_timing = os.environ.get("BAPMOS_RECORD_E2E_TIMING", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    e2e_rows: List[Dict[str, Any]] = []
    wall0 = time.perf_counter()

    for batch in test_loader:
        images = batch["image"]
        masks = batch["mask"]
        filenames = batch["filename"]
        for i in range(len(images)):
            image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
            mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
            fn = str(filenames[i])
            stem = _batch_filename_stem(fn)
            mask_arr = np.asarray(mask_multi)
            # Skip empty / pure-background GT slices (no foreground for metrics / viz).
            if mask_arr.size == 0 or int(mask_arr.max()) == 0:
                continue

            if record_e2e_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_slice0 = time.perf_counter()

            pred = predict_fn(image_rgb, mask_multi, fn)
            if pred is None:
                continue

            gt_u8 = mask_multi.astype(np.uint8)
            pred_u8 = to_reference_grid(pred, gt_u8)

            evaluator.evaluate_multiclass_slice(
                pred_u8,
                gt_u8,
                slice_idx=0,
                image_id=stem,
                class_mapping=class_mapping,
            )
            if record_e2e_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t_slice0) * 1000.0
                e2e_rows.append(
                    {
                        "stem": stem,
                        "ms_predict_and_metrics": round(ms, 4),
                        "timed_scope": "export_loop_predict_and_metrics",
                    }
                )
            rec = aggregate_slice_metrics_for_image(evaluator, stem)

            write_pred_images = save_predictions and (
                save_prediction_images_max is None
                or pred_img_count < int(save_prediction_images_max)
            )
            if write_pred_images:
                cv2.imwrite(str(pred_ids_dir / f"{stem}_pred_ids.png"), pred_u8)
                cv2.imwrite(
                    str(pred_dir / f"{stem}_pred.png"),
                    multiclass_mask_to_bgr(pred_u8, taxonomy.organ_definitions),
                )
                pred_img_count += 1

            if save_visualizations and rec is not None:
                if simple_viz:
                    # Cap PNG + PDF to the same first-N budget (protocol default 10).
                    if viz_pdf_max is not None and viz_png_count >= int(viz_pdf_max):
                        continue
                    write_pdf = bool(save_viz_pdf)
                    try:
                        png_path, pdf_path = save_overlay_pred_pair_png_pdf(
                            image_rgb=image_rgb,
                            pred_mask=pred_u8,
                            sample_stem=stem,
                            output_dir=viz_dir,
                            organ_definitions=taxonomy.organ_definitions,
                            taxonomy_name=taxonomy.taxonomy_name,
                            save_pdf=write_pdf,
                            pdf_dpi=viz_pdf_dpi,
                        )
                        fg_ids = sorted(
                            {int(o.class_id) for o in taxonomy.organ_definitions}
                        )
                        d_png, d_pdf = save_difference_v1_panel(
                            gt_u8,
                            pred_u8,
                            output_path=viz_dir / "difference" / f"{stem}_diff.png",
                            foreground_class_ids=fg_ids,
                            image_hw=tuple(gt_u8.shape[:2]),
                            save_pdf=write_pdf,
                            pdf_dpi=viz_pdf_dpi,
                        )
                        viz_png_count += 1
                        row = {
                            "sample_id": rec.sample_id,
                            "split": "test",
                            "mean_dice": _csv_metric_float(rec.mean_dice),
                            "mean_msd": _csv_metric_float(rec.mean_msd),
                            "mean_hd95": _csv_metric_float(rec.mean_hd95),
                            "distance_unit": distance_unit,
                            "viz_png_relative": str(png_path.relative_to(output_dir)),
                            "diff_png_relative": (
                                str(d_png.relative_to(output_dir)) if d_png is not None else ""
                            ),
                            "visualization_selection_mode": viz_selection,
                        }
                        if pdf_path is not None:
                            row["viz_pdf_relative"] = str(pdf_path.relative_to(output_dir))
                        if d_pdf is not None:
                            row["diff_pdf_relative"] = str(d_pdf.relative_to(output_dir))
                        idx_rows.append(row)
                    except Exception as exc:
                        raise RuntimeError(
                            f"visualization failed for {stem} (refusing silent skip)"
                        ) from exc
                else:
                    viz_records.append(rec)

    if save_visualizations and not simple_viz:
        picked = select_slices_for_visualization(
            viz_records,
            viz_selection,
            viz_max,
            viz_seed,
            per_patient_max=viz_per_patient_max,
        )
        selected_stems = {r.sample_stem for r in picked}
        by_stem = {r.sample_stem: r for r in picked}
        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]
            for i in range(len(images)):
                fn = str(filenames[i])
                stem = _batch_filename_stem(fn)
                if stem not in selected_stems:
                    continue
                rec = by_stem.get(stem)
                if rec is None:
                    continue
                # Same PNG+PDF cap as the simple path when viz_pdf_max is set.
                if viz_pdf_max is not None and viz_png_count >= int(viz_pdf_max):
                    continue
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                mask_arr = np.asarray(mask_multi)
                if mask_arr.size == 0 or int(mask_arr.max()) == 0:
                    continue
                pred = predict_fn(image_rgb, mask_multi, fn)
                if pred is None:
                    continue
                gt_u8 = mask_multi.astype(np.uint8)
                pred_u8 = to_reference_grid(pred, gt_u8)
                write_pdf = bool(save_viz_pdf)
                try:
                    png_path, pdf_path = save_overlay_pred_pair_png_pdf(
                        image_rgb=image_rgb,
                        pred_mask=pred_u8,
                        sample_stem=stem,
                        output_dir=viz_dir,
                        organ_definitions=taxonomy.organ_definitions,
                        taxonomy_name=taxonomy.taxonomy_name,
                        save_pdf=write_pdf,
                        pdf_dpi=viz_pdf_dpi,
                    )
                    fg_ids = sorted(
                        {int(o.class_id) for o in taxonomy.organ_definitions}
                    )
                    d_png, d_pdf = save_difference_v1_panel(
                        gt_u8,
                        pred_u8,
                        output_path=viz_dir / "difference" / f"{stem}_diff.png",
                        foreground_class_ids=fg_ids,
                        image_hw=tuple(gt_u8.shape[:2]),
                        save_pdf=write_pdf,
                        pdf_dpi=viz_pdf_dpi,
                    )
                    viz_png_count += 1
                    row = {
                        "sample_id": rec.sample_id,
                        "split": "test",
                        "mean_dice": _csv_metric_float(rec.mean_dice),
                        "mean_msd": _csv_metric_float(rec.mean_msd),
                        "mean_hd95": _csv_metric_float(rec.mean_hd95),
                        "distance_unit": distance_unit,
                        "viz_png_relative": str(png_path.relative_to(output_dir)),
                        "diff_png_relative": (
                            str(d_png.relative_to(output_dir)) if d_png is not None else ""
                        ),
                        "visualization_selection_mode": viz_selection,
                    }
                    if pdf_path is not None:
                        row["viz_pdf_relative"] = str(pdf_path.relative_to(output_dir))
                    if d_pdf is not None:
                        row["diff_pdf_relative"] = str(d_pdf.relative_to(output_dir))
                    idx_rows.append(row)
                except Exception as exc:
                    raise RuntimeError(
                        f"visualization failed for {stem} (refusing silent skip)"
                    ) from exc

    if save_visualizations and idx_rows:
        write_visualization_index_csv(output_dir, idx_rows)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
    evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
    evaluator.export_failure_analysis_csv(metrics_dir / "failure_analysis.csv", top_n=20)

    wall_s = time.perf_counter() - wall0
    if record_e2e_timing and e2e_rows:
        e2e_csv = metrics_dir / "e2e_per_slice.csv"
        with e2e_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["stem", "ms_predict_and_metrics", "timed_scope"]
            )
            w.writeheader()
            for row in e2e_rows:
                w.writerow(row)
        vals = [float(r["ms_predict_and_metrics"]) for r in e2e_rows]
        mean_ms = float(sum(vals) / len(vals))
        summary = {
            "timed_scope": "export_loop_predict_and_metrics",
            "n_slices": len(vals),
            "ms_per_slice_mean": round(mean_ms, 4),
            "ms_per_slice_std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "wall_clock_s": round(wall_s, 3),
            "wall_ms_per_slice": round((wall_s * 1000.0) / len(vals), 4),
            "note": (
                "Per-slice = predict_fn + native-grid metric eval (CUDA sync); not "
                "comparable with exports that scored metrics on a 256x256 grid. "
                "wall_* includes dataloader iteration; viz/pred PNG writes excluded when capped at 0."
            ),
        }
        (metrics_dir / "e2e_timing.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    if checkpoint is not None and data_root is not None and method_slug:
        meta_extra = dict(evaluation_meta_extra or {})
        meta_extra["evaluation_grid"] = EVALUATION_GRID
        meta_extra["pixel_spacing_mm"] = list(taxonomy.pixel_spacing_mm)
        if record_e2e_timing and e2e_rows:
            meta_extra["e2e_timing"] = "metrics/e2e_timing.json"
        write_method_evaluation_meta(
            output_dir,
            checkpoint=checkpoint,
            data_root=data_root,
            method_slug=method_slug,
            split="test",
            extra=meta_extra,
        )

    return evaluator
