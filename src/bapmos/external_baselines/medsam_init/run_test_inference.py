"""Stratified test inference export for MedSAM-init box decoder baselines.

Uses the same ``SAMMultiOrganTrainer`` box path as training (no ``legacy`` import).
Pooled prostate: one folder per ``site_tests/<site>/`` under ``output_dir``.
Primary seed-42 only unless ``--force-inference-output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.external_baselines.baseline_training_protocol import PFUS1_BASELINE_EVAL_SIZE
from bapmos.external_baselines.medsam_init.weight_loader import apply_medsam_encoder_init
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.inference_output.site_export import (
    export_stratified_sites,
    load_run_config,
    resolve_splits_subdir,
    resolve_torch_device,
)
from bapmos.metrics.baseline_validation_metrics import multiclass_pred_uint8_from_logits
from bapmos.multiorgan.train_sam_multiorgan_decoder_box import SAMMultiOrganTrainer
from bapmos.paths import resolve_model_checkpoint, resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile


def _resize_pred(pred_lr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(pred_lr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )


def run_medsam_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    viz_pdf_max: int | None = None,
    device: str = "cuda",
    force_inference_output: bool = False,
    eval_size: int | None = PFUS1_BASELINE_EVAL_SIZE,
) -> None:
    checkpoint = resolve_under_project(checkpoint)
    run_dir = checkpoint.parent
    cfg = load_run_config(run_dir, checkpoint)
    data_root = cfg.get("data_root")
    if not data_root:
        raise ValueError(f"checkpoint config missing data_root: {checkpoint}")
    run_name = str(cfg.get("run_name") or run_dir.name)
    require_primary_inference_seed_run(
        cfg=cfg,
        run_name=run_name,
        force=force_inference_output,
    )
    splits = resolve_splits_subdir(cfg.get("splits_subdir"), splits_subdir, data_root)

    # Resolve weight paths relative to the checkout (portable configs).
    cfg = dict(cfg)
    if cfg.get("sam_checkpoint"):
        cfg["sam_checkpoint"] = str(resolve_model_checkpoint(cfg["sam_checkpoint"]))
    if cfg.get("medsam_checkpoint"):
        cfg["medsam_checkpoint"] = str(resolve_model_checkpoint(cfg["medsam_checkpoint"]))

    trainer = SAMMultiOrganTrainer(cfg)
    apply_medsam_encoder_init(trainer.model, cfg)
    requested = resolve_torch_device(device)
    if trainer.device != requested:
        trainer.device = requested
        trainer.model.to(requested)

    try:
        ckpt = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=trainer.device)
    md = ckpt.get("model_state", {}).get("mask_decoder")
    if md is None:
        raise KeyError(f"No model_state.mask_decoder in {checkpoint}")
    trainer.model.mask_decoder.load_state_dict(md)
    trainer.model.eval()

    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)
    viz_kwargs = export_viz_kwargs_for_data_root(
        data_root,
        per_patient_max=viz_per_patient_max,
        viz_pdf_max=viz_pdf_max,
    )

    @torch.no_grad()
    def predict_fn(image_rgb, mask_multi, fn: str):
        out = trainer._forward_one(image_rgb, mask_multi, fn, is_train=False)
        if out is None:
            return None
        logits, _gt = out
        pred_lr = multiclass_pred_uint8_from_logits(logits)
        if hasattr(mask_multi, "numpy"):
            mask_multi = mask_multi.numpy()
        return _resize_pred(pred_lr, mask_multi.shape[:2])

    export_stratified_sites(
        data_root=data_root,
        output_dir=Path(output_dir),
        splits_subdir=splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=unit,
        checkpoint=checkpoint,
        method_slug="medsam",
        eval_size=eval_size,
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "medsam",
            "run_name": run_name,
            "test_site": label,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export MedSAM-init test split to inference_output/ "
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

    run_medsam_test_export(
        Path(args.checkpoint),
        resolve_under_project(args.output_dir),
        splits_subdir=args.splits_subdir,
        viz_per_patient_max=args.viz_per_patient_max,
        viz_pdf_max=args.viz_pdf_max,
        device=args.device,
        force_inference_output=bool(args.force_inference_output),
        eval_size=args.eval_size,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
