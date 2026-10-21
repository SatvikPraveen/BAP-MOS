"""Stratified test inference export for MedSAM-init box decoder baselines.

Uses the same ``SAMMultiOrganTrainer`` box path as training (no ``legacy`` import).
Pooled prostate: one folder per ``site_tests/<site>/`` under ``output_dir``.
Primary seed-42 only unless ``--force-inference-output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bapmos.inference_output.prompt_log import write_prompt_log
from bapmos.evaluation.test_panels import distance_unit_short
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
from bapmos.method.run_test_inference import load_prompt_mask
from bapmos.multiorgan.train_sam_multiorgan_decoder_box import SAMMultiOrganTrainer
from bapmos.paths import resolve_model_checkpoint, resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile


def run_medsam_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    viz_pdf_max: int | None = None,
    device: str = "cuda",
    force_inference_output: bool = False,
    prompt_noise_px: int = 0,
    prompt_noise_mode: str = "shift_paired",
    prompt_noise_seed: int = 0,
    prompt_mask_dir: Path | None = None,
) -> None:
    """Export stratified test-split predictions and panels for a MedSAM-init box checkpoint.

    Rebuilds ``SAMMultiOrganTrainer`` from the run's saved config, re-applies the MedSAM
    encoder weights, loads ``model_state.mask_decoder`` from ``checkpoint`` and writes
    per-site results under ``output_dir`` via ``export_stratified_sites``. Refuses
    non-primary-seed runs unless ``force_inference_output``.
    """
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
    trainer.test_prompt_noise_px = int(prompt_noise_px)
    trainer.test_prompt_noise_mode = str(prompt_noise_mode)
    trainer.test_prompt_noise_seed = int(prompt_noise_seed)
    trainer.test_prompt_log = {} if prompt_noise_px > 0 else None
    if prompt_mask_dir is not None:
        prompt_mask_dir = resolve_under_project(prompt_mask_dir)
        if not prompt_mask_dir.is_dir():
            raise FileNotFoundError(f"--prompt-mask-dir not found: {prompt_mask_dir}")

    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)
    viz_kwargs = export_viz_kwargs_for_data_root(
        data_root,
        per_patient_max=viz_per_patient_max,
        viz_pdf_max=viz_pdf_max,
    )

    @torch.no_grad()
    def predict_fn(image_rgb, mask_multi, fn: str):
        """Return the 256x256 label map for one test slice, or None if no organ yields a prompt."""
        # Same box-prompt forward pass as validation; None means the slice has no prompt.
        prompt_mask = mask_multi
        if prompt_mask_dir is not None:
            prompt_mask = load_prompt_mask(prompt_mask_dir, fn, np.asarray(mask_multi).shape)
        out = trainer._forward_one(image_rgb, prompt_mask, fn, is_train=False)
        if out is None:
            return None
        logits, _gt = out
        pred_lr = multiclass_pred_uint8_from_logits(logits)
        return pred_lr

    export_stratified_sites(
        data_root=data_root,
        output_dir=Path(output_dir),
        splits_subdir=splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=unit,
        checkpoint=checkpoint,
        method_slug="medsam",
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "medsam",
            "run_name": run_name,
            "test_site": label,
            "prompt_noise_px": int(prompt_noise_px),
            "prompt_noise_mode": str(prompt_noise_mode) if prompt_noise_px > 0 else None,
            "prompt_source": "external_mask" if prompt_mask_dir is not None else "gt_mask",
            "prompt_mask_dir": str(prompt_mask_dir) if prompt_mask_dir is not None else None,
        },
    )
    if trainer.test_prompt_log is not None:
        write_prompt_log(trainer.test_prompt_log, Path(output_dir), {"noise_px": int(prompt_noise_px), "mode": prompt_noise_mode, "seed": int(prompt_noise_seed), "checkpoint": str(checkpoint)})


def main() -> None:
    """CLI entry point for :func:`run_medsam_test_export`."""
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
    p.add_argument("--prompt-noise-px", type=int, default=0, help="Paired test-time box perturbation size (px).")
    p.add_argument("--prompt-noise-mode", choices=("shift_paired", "loose_paired"), default="shift_paired")
    p.add_argument("--prompt-noise-seed", type=int, default=0)
    p.add_argument(
        "--prompt-mask-dir",
        type=str,
        default=None,
        help="Build box prompts from another model's predicted label maps (*_pred_ids.png) "
        "instead of the GT mask; scoring still uses the GT mask.",
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
        prompt_noise_px=int(args.prompt_noise_px),
        prompt_noise_mode=args.prompt_noise_mode,
        prompt_noise_seed=int(args.prompt_noise_seed),
        prompt_mask_dir=Path(args.prompt_mask_dir) if args.prompt_mask_dir else None,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
