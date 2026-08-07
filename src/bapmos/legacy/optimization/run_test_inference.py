"""Stratified test export for legacy optimization checkpoints (ratios, policies).

Uses ``OptimizationInference`` with the standard ``export_stratified_test_bundle``
layout (per-site under pooled prostate). Primary seed-42 only unless
``--force-inference-output``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.external_baselines.baseline_training_protocol import PFUS1_BASELINE_EVAL_SIZE
from bapmos.inference_output.export_bundle import export_stratified_test_bundle
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.legacy.optimization.inference import OptimizationInference
from bapmos.method.data_adapter import iter_test_split_datasets
from bapmos.multiorgan.dataset_multi_organ import multi_organ_collate_fn
from bapmos.paths import is_pooled_prostate_training_root, resolve_under_project
from bapmos.train.training_taxonomy import default_splits_subdir, get_baseline_taxonomy_profile


def run_legacy_optimization_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    method_slug: str,
    splits_subdir: str | None = None,
    viz_pdf_max: int | None = None,
    force_inference_output: bool = False,
    device: str = "cuda",
    eval_size: int | None = PFUS1_BASELINE_EVAL_SIZE,
) -> None:
    checkpoint = resolve_under_project(checkpoint)
    run_dir = checkpoint.parent
    inference = OptimizationInference(checkpoint, device=device)
    cfg = inference.config
    require_primary_inference_seed_run(
        cfg=cfg,
        run_name=str(cfg.get("run_name") or run_dir.name),
        force=force_inference_output,
    )

    data_root = cfg["data_root"]
    splits = cfg.get("splits_subdir") or splits_subdir or default_splits_subdir(data_root)
    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)
    run_name = str(cfg.get("run_name") or run_dir.name)
    viz_kwargs = export_viz_kwargs_for_data_root(data_root, viz_pdf_max=viz_pdf_max)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def predict_fn(image_rgb, mask_multi, fn: str):
        return inference.predict_one(image_rgb, mask_multi, fn)

    site_labels: list[str] = []
    for label, test_ds in iter_test_split_datasets(data_root, splits_subdir=splits):
        site_labels.append(label)
        site_out = (
            output_dir / label
            if is_pooled_prostate_training_root(data_root)
            else output_dir
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=multi_organ_collate_fn,
        )
        export_stratified_test_bundle(
            test_loader=test_loader,
            predict_fn=predict_fn,
            output_dir=site_out,
            taxonomy=taxonomy,
            distance_unit=unit,
            checkpoint=checkpoint,
            data_root=data_root,
            method_slug=method_slug,
            eval_size=eval_size,
            evaluation_meta_extra={
                "family": "legacy_optimization",
                "prompt_strategy": inference.prompt_strategy,
                "run_name": run_name,
                "test_site": label,
            },
            **viz_kwargs,
        )

    if is_pooled_prostate_training_root(data_root):
        (output_dir / "sites.json").write_text(
            json.dumps({"sites": site_labels, "run_name": run_name}, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export legacy optimization test split under inference_output/ "
            "(primary seed-42 only unless --force-inference-output)."
        )
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--method-slug", type=str, required=True, dest="method_slug")
    p.add_argument("--splits-subdir", type=str, default=None)
    p.add_argument("--viz-pdf-max", type=int, default=None)
    p.add_argument("--force-inference-output", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--eval-size",
        type=int,
        default=PFUS1_BASELINE_EVAL_SIZE,
        help=(
            "Square side for nearest-neighbor pred/GT resize before MSD/HD95 "
            f"(default {PFUS1_BASELINE_EVAL_SIZE}). Use 0 for native resolution."
        ),
    )
    args = p.parse_args()

    run_legacy_optimization_test_export(
        Path(args.checkpoint),
        resolve_under_project(args.output_dir),
        method_slug=str(args.method_slug),
        splits_subdir=args.splits_subdir,
        viz_pdf_max=args.viz_pdf_max,
        force_inference_output=bool(args.force_inference_output),
        device=args.device,
        eval_size=args.eval_size,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
