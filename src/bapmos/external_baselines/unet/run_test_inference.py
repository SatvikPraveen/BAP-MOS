"""Stratified test inference export for SMP U-Net multiclass baselines.

Pooled prostate: one folder per ``site_tests/<site>/`` under ``output_dir``.
Primary seed-42 only unless ``--force-inference-output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bapmos.external_baselines.baseline_training_protocol import PFUS1_BASELINE_EVAL_SIZE
from bapmos.external_baselines.common import imagenet_normalize_bchw, resize_for_baseline
from bapmos.external_baselines.unet.train_multiclass import (
    UNetMultiOrganTrainer,
    multiclass_pred_uint8_from_logits,
)
from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.inference_output.site_export import (
    export_stratified_sites,
    load_run_config,
    resolve_splits_subdir,
    resolve_torch_device,
)
from bapmos.paths import resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile


def run_unet_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    viz_pdf_max: int | None = None,
    save_visualizations: bool | None = None,
    force_inference_output: bool = False,
    device: str = "cuda",
    eval_size: int | None = PFUS1_BASELINE_EVAL_SIZE,
) -> None:
    checkpoint = resolve_under_project(checkpoint)
    run_dir = checkpoint.parent
    cfg = load_run_config(run_dir, checkpoint)

    require_primary_inference_seed_run(
        cfg=cfg,
        run_name=str(cfg.get("run_name") or run_dir.name),
        force=force_inference_output,
    )

    data_root = cfg["data_root"]
    splits = resolve_splits_subdir(cfg.get("splits_subdir"), splits_subdir, data_root)

    trainer = UNetMultiOrganTrainer(cfg)
    requested = resolve_torch_device(device)
    if trainer.device != requested:
        trainer.device = requested
        trainer.model.to(requested)
    try:
        ckpt = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=trainer.device)
    if "model_state" not in ckpt:
        raise KeyError(f"No model_state in {checkpoint}")
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.model.eval()

    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)
    run_name = str(cfg.get("run_name") or run_dir.name)

    @torch.no_grad()
    def predict_fn(image_rgb, mask_multi, fn: str):
        del fn
        x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
        xb = imagenet_normalize_bchw(x.to(trainer.device))
        logits = trainer.model(xb)
        pred_lr = multiclass_pred_uint8_from_logits(logits)
        if hasattr(mask_multi, "numpy"):
            mask_multi = mask_multi.numpy()
        h, w = mask_multi.shape[:2]
        return cv2.resize(pred_lr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(
            np.uint8
        )

    viz_kwargs = export_viz_kwargs_for_data_root(
        data_root,
        per_patient_max=viz_per_patient_max,
        viz_pdf_max=viz_pdf_max,
    )
    if save_visualizations is not None:
        viz_kwargs = dict(viz_kwargs)
        viz_kwargs["save_visualizations"] = save_visualizations
        if not save_visualizations:
            viz_kwargs["save_viz_pdf"] = False

    export_stratified_sites(
        data_root=data_root,
        output_dir=Path(output_dir),
        splits_subdir=splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=unit,
        checkpoint=checkpoint,
        method_slug="u_net",
        eval_size=eval_size,
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "unet",
            "run_name": run_name,
            "test_site": label,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export U-Net test split to inference_output/ "
            "(primary seed-42 only unless --force-inference-output; "
            "first-10 pred+panel PNG/PDF @ 350 dpi). "
            "Pooled prostate writes one subfolder per site_tests site."
        )
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits-subdir", type=str, default=None)
    p.add_argument("--viz-per-patient-max", type=int, default=None, help="Deprecated (ignored).")
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

    run_unet_test_export(
        Path(args.checkpoint),
        resolve_under_project(args.output_dir),
        splits_subdir=args.splits_subdir,
        viz_per_patient_max=args.viz_per_patient_max,
        viz_pdf_max=args.viz_pdf_max,
        force_inference_output=bool(args.force_inference_output),
        device=args.device,
        eval_size=args.eval_size,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
