"""Optional qualitative panels after single-organ baseline test evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from bapmos.metrics.baseline_validation_metrics import (
    append_single_organ_validation_slice,
    multiclass_pred_uint8_from_logits,
)
from bapmos.evaluation.test_panels import distance_unit_short, save_multiclass_test_panel_png_pdf
from bapmos.evaluation.viz_index import write_visualization_index_csv
from bapmos.evaluation.viz_selection import (
    SliceVizRecord,
    aggregate_slice_metrics_for_image,
    select_slices_for_visualization,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator
from bapmos.data.organ_registry import REAL_CLINICAL_ORGANS, OrganDefinition


def _organ_def_tuple(organ_key: str) -> Tuple[OrganDefinition, ...]:
    for o in REAL_CLINICAL_ORGANS:
        if o.key == organ_key:
            return (o,)
    raise ValueError(f"Unknown clinical organ key for viz: {organ_key!r}")


def _resize_mask(pred_low: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(pred_low.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _fmt_metric(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.6f}"


def run_singleorgan_baseline_test_visualizations(
    trainer: Any,
    test_loader,
    run_dir: Path,
    viz: Dict[str, Any],
) -> None:
    """Panels under ``run_dir / test_results / visualizations`` (same layout as multi-organ baseline)."""
    selection = viz.get("selection", "all")
    max_n = viz.get("max")
    seed = int(viz.get("seed", 42))

    test_results_dir = run_dir / "test_results"
    viz_dir = test_results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    organ_key = trainer.organ
    organ_defs = _organ_def_tuple(organ_key)
    taxonomy_name = trainer._taxonomy.taxonomy_name
    unit = distance_unit_short(taxonomy_name)

    evaluator = MetricsEvaluator(
        pixel_spacing=trainer._taxonomy.pixel_spacing_mm,
        organs=[trainer._eval_organ_label],
    )

    simple_viz = selection == "all" and max_n is None
    records: List[SliceVizRecord] = []
    idx_rows: List[Dict[str, Any]] = []

    trainer.model.eval()

    for batch in test_loader:
        images = batch["image"]
        masks = batch["mask"]
        filenames = batch["filename"]

        for i in range(len(images)):
            image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
            mask_bin = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
            fn = filenames[i]

            if mask_bin.max() == 0:
                continue

            with torch.no_grad():
                out = trainer._forward_one(image_rgb, mask_bin, fn, is_train=False)
            if out is None:
                continue
            logits, gt = out

            append_single_organ_validation_slice(
                evaluator,
                pred_logits=logits,
                gt_binary=gt,
                organ_label=trainer._eval_organ_label,
                image_id=str(fn),
            )
            rec = aggregate_slice_metrics_for_image(evaluator, str(fn))
            if rec is None:
                continue

            pred_lr = multiclass_pred_uint8_from_logits(logits)
            pred_full = _resize_mask(pred_lr, mask_bin.shape[:2])
            gt_full = (mask_bin > 0).astype(np.uint8)

            if simple_viz:
                png_path, pdf_path = save_multiclass_test_panel_png_pdf(
                    image_rgb=image_rgb,
                    gt_mask=gt_full,
                    pred_mask=pred_full,
                    sample_stem=rec.sample_stem,
                    output_dir=viz_dir,
                    organ_definitions=organ_defs,
                    taxonomy_name=taxonomy_name,
                )
                idx_rows.append(
                    {
                        "sample_id": rec.sample_id,
                        "split": "test",
                        "mean_dice": _fmt_metric(rec.mean_dice),
                        "mean_msd": _fmt_metric(rec.mean_msd),
                        "mean_hd95": _fmt_metric(rec.mean_hd95),
                        "distance_unit": unit,
                        "panel_png_relative": str(png_path.relative_to(test_results_dir)),
                        "panel_pdf_relative": str(pdf_path.relative_to(test_results_dir)),
                        "visualization_selection_mode": selection,
                    }
                )
            else:
                records.append(rec)

    if not simple_viz:
        picked = select_slices_for_visualization(records, selection, max_n, seed)
        selected_stems = {r.sample_stem for r in picked}
        by_stem = {r.sample_stem: r for r in picked}

        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"]
            filenames = batch["filename"]

            for i in range(len(images)):
                image_rgb = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
                mask_bin = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                stem = Path(str(fn)).stem

                if stem not in selected_stems:
                    continue
                if mask_bin.max() == 0:
                    continue

                with torch.no_grad():
                    out = trainer._forward_one(image_rgb, mask_bin, fn, is_train=False)
                if out is None:
                    continue
                logits, _gt = out
                pred_lr = multiclass_pred_uint8_from_logits(logits)
                pred_full = _resize_mask(pred_lr, mask_bin.shape[:2])
                gt_full = (mask_bin > 0).astype(np.uint8)
                rec = by_stem.get(stem)
                if rec is None:
                    continue

                png_path, pdf_path = save_multiclass_test_panel_png_pdf(
                    image_rgb=image_rgb,
                    gt_mask=gt_full,
                    pred_mask=pred_full,
                    sample_stem=rec.sample_stem,
                    output_dir=viz_dir,
                    organ_definitions=organ_defs,
                    taxonomy_name=taxonomy_name,
                )
                idx_rows.append(
                    {
                        "sample_id": rec.sample_id,
                        "split": "test",
                        "mean_dice": _fmt_metric(rec.mean_dice),
                        "mean_msd": _fmt_metric(rec.mean_msd),
                        "mean_hd95": _fmt_metric(rec.mean_hd95),
                        "distance_unit": unit,
                        "panel_png_relative": str(png_path.relative_to(test_results_dir)),
                        "panel_pdf_relative": str(pdf_path.relative_to(test_results_dir)),
                        "visualization_selection_mode": selection,
                    }
                )

    if idx_rows:
        write_visualization_index_csv(test_results_dir, idx_rows)
