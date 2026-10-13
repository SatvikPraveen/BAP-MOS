"""Every method is scored on the native reference grid with physical spacing."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bapmos.data.organ_registry import OrganDefinition
from bapmos.metrics.evaluation_grid import to_reference_grid
from bapmos.train.training_taxonomy import BaselineTaxonomyProfile

SPACING_MM = 0.159072
# Non-square native frame, so a square evaluation grid would distort each axis differently.
NATIVE_HW = (400, 600)
MODEL_GRID = 256
SHIFT_PX = 12


def _reference_and_shifted_prediction() -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros(NATIVE_HW, dtype=np.uint8)
    gt[100:300, 150:450] = 1
    pred = np.zeros_like(gt)
    pred[100:300, 150 + SHIFT_PX : 450 + SHIFT_PX] = 1
    return gt, pred


def _on_model_grid(native: np.ndarray) -> np.ndarray:
    return cv2.resize(native, (MODEL_GRID, MODEL_GRID), interpolation=cv2.INTER_NEAREST)


def _one_organ_taxonomy() -> BaselineTaxonomyProfile:
    return BaselineTaxonomyProfile(
        taxonomy_name="test_one_organ",
        is_simulation=False,
        num_classes=2,
        organ_keys=("organ",),
        organ_to_class={"organ": 1},
        multiclass_eval_mapping={1: "Organ"},
        evaluator_organ_labels=("Organ",),
        pixel_spacing_mm=(SPACING_MM, SPACING_MM),
        data_root_resolved=Path("."),
        organ_definitions=(OrganDefinition("organ", 1, "Organ", (0, 0, 255)),),
    )


def test_to_reference_grid_upsamples_prediction_to_reference_shape():
    gt, pred = _reference_and_shifted_prediction()
    aligned = to_reference_grid(_on_model_grid(pred), gt)
    assert aligned.shape == gt.shape
    assert aligned.dtype == np.uint8
    assert set(np.unique(aligned)) == {0, 1}


def test_to_reference_grid_leaves_native_prediction_unchanged():
    gt, pred = _reference_and_shifted_prediction()
    np.testing.assert_array_equal(to_reference_grid(pred, gt), pred)


def test_to_reference_grid_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        to_reference_grid(np.zeros((4, 4, 3)), np.zeros((4, 4)))


def test_export_reports_distances_at_native_scale(tmp_path):
    """A model-grid prediction is scored as far from the reference as it really is."""
    from bapmos.inference_output.export_bundle import export_stratified_test_bundle

    gt, pred = _reference_and_shifted_prediction()
    loader = [
        {
            "image": [np.zeros((*NATIVE_HW, 3), dtype=np.uint8)],
            "mask": [gt],
            "filename": ["slice_000.png"],
        }
    ]
    export_stratified_test_bundle(
        test_loader=loader,
        predict_fn=lambda image, mask, name: _on_model_grid(pred),
        output_dir=tmp_path,
        taxonomy=_one_organ_taxonomy(),
        distance_unit="mm",
        save_predictions=False,
        save_visualizations=False,
    )

    with (tmp_path / "metrics" / "per_slice_metrics.csv").open(encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    hd95_px = float(row["hd95_mm"]) / SPACING_MM
    # The model grid quantises the shift by at most one model pixel per side (~2.3 native px).
    # Scoring on a shared 256x256 grid reports about 6 px instead (half the true distance).
    assert hd95_px == pytest.approx(SHIFT_PX, abs=3)


@pytest.mark.parametrize(
    "module",
    [
        "bapmos.inference_output",
        "bapmos.method.run_test_inference",
        "bapmos.external_baselines.unet.run_test_inference",
        "bapmos.external_baselines.nnunet2d.run_test_inference",
        "bapmos.external_baselines.nnunet2d.evaluate_nnunet_predictions",
        "bapmos.external_baselines.medsam_init.run_test_inference",
        "bapmos.multiorgan.run_test_inference",
        "bapmos.legacy.optimization.run_test_inference",
    ],
)
def test_test_export_entry_points_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("bad", [np.full((4, 4), -1), np.full((4, 4), 0.5), np.full((4, 4), 300)])
def test_to_reference_grid_rejects_invalid_class_ids(bad):
    with pytest.raises(ValueError, match="class ids"):
        to_reference_grid(bad, np.zeros((4, 4), dtype=np.uint8))


def test_trainer_metrics_score_against_the_native_reference():
    """Validation / test helpers used by every trainer lift the prediction to the reference."""
    from bapmos.legacy.optimization.metrics import MetricsEvaluator
    from bapmos.metrics.baseline_validation_metrics import (
        append_multiclass_validation_slice,
    )

    gt, pred = _reference_and_shifted_prediction()
    evaluator = MetricsEvaluator(pixel_spacing=(SPACING_MM, SPACING_MM), organs=["Organ"])
    append_multiclass_validation_slice(
        evaluator,
        pred_classes=_on_model_grid(pred),
        gt_classes=gt,
        image_id="slice_000",
        class_mapping={1: "Organ"},
    )
    (row,) = [m for m in evaluator.per_slice_metrics if m["organ"] == "Organ"]
    assert row["hd95_mm"] / SPACING_MM == pytest.approx(SHIFT_PX, abs=3)


def test_export_records_the_evaluation_grid(tmp_path):
    from bapmos.inference_output.export_bundle import export_stratified_test_bundle
    from bapmos.metrics.evaluation_grid import EVALUATION_GRID

    gt, pred = _reference_and_shifted_prediction()
    checkpoint = tmp_path / "best_checkpoint.pth"
    checkpoint.touch()
    export_stratified_test_bundle(
        test_loader=[
            {"image": [np.zeros((*NATIVE_HW, 3), np.uint8)], "mask": [gt], "filename": ["s.png"]}
        ],
        predict_fn=lambda image, mask, name: _on_model_grid(pred),
        output_dir=tmp_path / "export",
        taxonomy=_one_organ_taxonomy(),
        distance_unit="mm",
        save_predictions=False,
        save_visualizations=False,
        checkpoint=checkpoint,
        data_root=tmp_path,
        method_slug="test_method",
    )
    meta = json.loads((tmp_path / "export" / "evaluation_meta.json").read_text(encoding="utf-8"))
    assert meta["evaluation_grid"] == EVALUATION_GRID
    assert meta["pixel_spacing_mm"] == [SPACING_MM, SPACING_MM]


def test_ingest_reads_only_native_grid_metrics(tmp_path):
    from bapmos.metrics.evaluation_grid import EVALUATION_GRID
    from bapmos.results.recompute_native_metrics import _native_metrics_dir

    def export(name, meta):
        d = tmp_path / name
        (d / "metrics").mkdir(parents=True)
        (d / "metrics" / "summary_metrics.csv").write_text("organ\n", encoding="utf-8")
        (d / "evaluation_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return d

    native = export("native", {"evaluation_grid": EVALUATION_GRID})
    resized = export("resized", {"eval_size": 256})
    assert _native_metrics_dir(native) == native / "metrics"
    assert _native_metrics_dir(resized) is None

    redone = resized / "metrics_native_redone"
    redone.mkdir()
    (redone / "summary_metrics.csv").write_text("organ\n", encoding="utf-8")
    assert _native_metrics_dir(resized) == redone
