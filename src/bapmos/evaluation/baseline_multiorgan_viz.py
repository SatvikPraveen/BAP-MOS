"""Optional qualitative panels after multi-organ baseline test evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

from bapmos.metrics.baseline_validation_metrics import (
    append_multiclass_validation_slice,
    gt_uint8_from_tensor,
    multiclass_pred_uint8_from_logits,
)
from bapmos.evaluation.test_panels import (
    distance_unit_short,
    save_multiclass_test_panel_png_pdf,
    save_overlay_pred_pair_png_pdf,
)
from bapmos.evaluation.difference_v1 import save_difference_v1_panel
from bapmos.evaluation.viz_index import write_visualization_index_csv
from bapmos.evaluation.viz_selection import (
    SliceVizRecord,
    aggregate_slice_metrics_for_image,
    select_slices_for_visualization,
)
from bapmos.legacy.optimization.metrics import MetricsEvaluator


def _resize_multiclass(pred_low: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(pred_low.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )


def _fmt_metric(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.6f}"


def _export_evaluator_metrics(evaluator: MetricsEvaluator, metrics_dir: Path, organ_labels: List[str]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.export_per_slice_csv(metrics_dir / "per_slice_metrics.csv")
    evaluator.export_summary_csv(metrics_dir / "summary_metrics.csv")
    evaluator.export_failure_analysis_csv(metrics_dir / "failure_analysis.csv", top_n=20)
    rows = []
    for organ in organ_labels:
        agg = evaluator.aggregate_metrics(organ_name=organ)
        if agg is not None:
            rows.append(agg)
    if rows:
        pd.DataFrame(rows).to_csv(metrics_dir / "per_organ_metrics.csv", index=False)


def run_multiorgan_baseline_test_visualizations(
    trainer: Any,
    test_loader,
    run_dir: Path,
    viz: Dict[str, Any],
    *,
    output_root: Optional[Path] = None,
    save_viz_pdf: bool = True,
) -> Path:
    """
    Test metrics + TP/FP/FN visualizations under canonical ``output/<dataset>/<method>/``.

    When ``output_root`` is set, writes::

        <output_root>/metrics/*.csv
        <output_root>/visualizations/difference/*_diff.png|.pdf
        <output_root>/visualization_index.csv

    Otherwise uses legacy ``run_dir / test_results /`` (backward compatible).
    """
    selection = viz.get("selection", "all")
    max_n = viz.get("max")
    seed = int(viz.get("seed", 42))

    root = Path(output_root) if output_root is not None else Path(run_dir) / "test_results"
    root.mkdir(parents=True, exist_ok=True)
    viz_dir = root / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "metrics"

    taxonomy = trainer._taxonomy
    unit = distance_unit_short(taxonomy.taxonomy_name)
    organ_labels = list(taxonomy.evaluator_organ_labels)

    evaluator = MetricsEvaluator(
        pixel_spacing=taxonomy.pixel_spacing_mm,
        organs=organ_labels,
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
            mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
            fn = filenames[i]

            if mask_multi.max() == 0:
                continue

            with torch.no_grad():
                out = trainer._forward_one(image_rgb, mask_multi, fn, is_train=False)
            if out is None:
                continue
            logits, gt = out

            pred_u8_lr = multiclass_pred_uint8_from_logits(logits)
            gt_u8_lr = gt_uint8_from_tensor(gt)
            append_multiclass_validation_slice(
                evaluator,
                pred_classes=pred_u8_lr,
                gt_classes=gt_u8_lr,
                image_id=str(fn),
                class_mapping=taxonomy.multiclass_eval_mapping,
            )
            rec = aggregate_slice_metrics_for_image(evaluator, str(fn))
            if rec is None:
                continue

            pred_full = _resize_multiclass(pred_u8_lr, mask_multi.shape[:2])

            if simple_viz:
                stem = rec.sample_stem
                try:
                    png_path, pdf_path = save_overlay_pred_pair_png_pdf(
                        image_rgb=image_rgb,
                        pred_mask=pred_full,
                        sample_stem=stem,
                        output_dir=viz_dir,
                        organ_definitions=taxonomy.organ_definitions,
                        taxonomy_name=taxonomy.taxonomy_name,
                        save_pdf=save_viz_pdf,
                    )
                    fg_ids = sorted({int(o.class_id) for o in taxonomy.organ_definitions})
                    d_png, d_pdf = save_difference_v1_panel(
                        mask_multi.astype(np.uint8),
                        pred_full,
                        output_path=viz_dir / "difference" / f"{stem}_diff.png",
                        foreground_class_ids=fg_ids,
                        image_hw=tuple(mask_multi.shape[:2]),
                        save_pdf=save_viz_pdf,
                    )
                    row = {
                        "sample_id": rec.sample_id,
                        "split": "test",
                        "mean_dice": _fmt_metric(rec.mean_dice),
                        "mean_msd": _fmt_metric(rec.mean_msd),
                        "mean_hd95": _fmt_metric(rec.mean_hd95),
                        "distance_unit": unit,
                        "viz_png_relative": str(png_path.relative_to(root)),
                        "diff_png_relative": (
                            str(d_png.relative_to(root)) if d_png is not None else ""
                        ),
                        "visualization_selection_mode": selection,
                    }
                    if pdf_path is not None:
                        row["viz_pdf_relative"] = str(pdf_path.relative_to(root))
                    if d_pdf is not None:
                        row["diff_pdf_relative"] = str(d_pdf.relative_to(root))
                    idx_rows.append(row)
                except Exception as exc:
                    raise RuntimeError(
                        f"visualization failed for {fn} (refusing silent skip)"
                    ) from exc
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
                mask_multi = masks[i].numpy() if torch.is_tensor(masks[i]) else masks[i]
                fn = filenames[i]
                stem = Path(str(fn)).stem

                if stem not in selected_stems:
                    continue
                if mask_multi.max() == 0:
                    continue

                with torch.no_grad():
                    out = trainer._forward_one(image_rgb, mask_multi, fn, is_train=False)
                if out is None:
                    continue
                logits, _gt = out
                pred_u8_lr = multiclass_pred_uint8_from_logits(logits)
                pred_full = _resize_multiclass(pred_u8_lr, mask_multi.shape[:2])
                rec = by_stem.get(stem)
                if rec is None:
                    continue

                png_path, pdf_path = save_multiclass_test_panel_png_pdf(
                    image_rgb=image_rgb,
                    gt_mask=mask_multi.astype(np.uint8),
                    pred_mask=pred_full,
                    sample_stem=rec.sample_stem,
                    output_dir=viz_dir,
                    organ_definitions=taxonomy.organ_definitions,
                    taxonomy_name=taxonomy.taxonomy_name,
                )
                fg_ids = sorted({int(o.class_id) for o in taxonomy.organ_definitions})
                d_png, d_pdf = save_difference_v1_panel(
                    mask_multi.astype(np.uint8),
                    pred_full,
                    output_path=viz_dir / "difference" / f"{rec.sample_stem}_diff.png",
                    foreground_class_ids=fg_ids,
                    image_hw=tuple(mask_multi.shape[:2]),
                    save_pdf=save_viz_pdf,
                )
                row = {
                    "sample_id": rec.sample_id,
                    "split": "test",
                    "mean_dice": _fmt_metric(rec.mean_dice),
                    "mean_msd": _fmt_metric(rec.mean_msd),
                    "mean_hd95": _fmt_metric(rec.mean_hd95),
                    "distance_unit": unit,
                    "panel_png_relative": str(png_path.relative_to(root)),
                    "panel_pdf_relative": str(pdf_path.relative_to(root)),
                    "diff_png_relative": (
                        str(d_png.relative_to(root)) if d_png is not None else ""
                    ),
                    "visualization_selection_mode": selection,
                }
                if d_pdf is not None:
                    row["diff_pdf_relative"] = str(d_pdf.relative_to(root))
                idx_rows.append(row)

    _export_evaluator_metrics(evaluator, metrics_dir, organ_labels)

    if idx_rows:
        write_visualization_index_csv(root, idx_rows)

    return root
