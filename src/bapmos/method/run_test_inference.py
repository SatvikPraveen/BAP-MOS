"""Stratified test-split inference export for ``bapmos.method`` checkpoints.

Decoder outputs are at 256×256; ``export_stratified_test_bundle`` brings them to the
native GT grid, where every method is scored (``bapmos.metrics.evaluation_grid``).

Pooled prostate uses ``site_tests/<site>/test.txt`` (no global ``test.txt``).
Each site is exported under ``<output_dir>/<site>/``.

Prompt-source options (robustness / fully automatic evaluation):

- ``--prompt-noise-px n``: perturb every box edge and point by U{-n..n} px
  (``prompt_geometry.perturb_slice_prompts``; same offsets for every method).
- ``--prompt-noise-mode loose``: instead move every box edge outward by U{0..n} px
  (a box drawn larger than the organ; points unchanged).
- ``--prompt-mask-dir DIR``: build prompts from another model's predicted label maps
  (``DIR/<stem>_pred_ids.png`` on the native grid, e.g. a U-Net export's
  ``predictions/multiclass/``) instead of the GT mask. Scoring still uses the GT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bapmos.inference_output.prompt_log import write_prompt_log
from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.inference_output.site_export import (
    export_stratified_sites,
    load_run_config,
    resolve_splits_subdir,
    resolve_torch_device,
)
from bapmos.method.bap_mos_trainer import BAPMOSTrainer
from bapmos.method.data_adapter import infer_organ_to_class
from bapmos.paths import resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile


def run_bapmos_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    viz_pdf_max: int | None = None,
    force_inference_output: bool = False,
    prompt_noise_px: int = 0,
    prompt_noise_seed: int = 0,
    prompt_mask_dir: Path | None = None,
    prompt_noise_mode: str = "uniform",
) -> None:
    """Export stratified test predictions, metrics and panels for one trained checkpoint.

    Rebuilds the trainer from the run's saved config, loads the checkpoint's decoder and
    bandit state, prompts each organ with its highest-Q arm, and writes the export bundle
    under ``output_dir`` (one subfolder per site for pooled prostate). Refuses
    non-primary-seed runs unless ``force_inference_output`` is set.
    """
    checkpoint = resolve_under_project(checkpoint)
    run_dir = checkpoint.parent
    cfg = load_run_config(run_dir, checkpoint)
    common = cfg.get("common", cfg)
    run_name = str(
        cfg.get("run_name")
        or cfg.get("experiment_name")
        or common.get("run_name")
        or run_dir.name
    )
    require_primary_inference_seed_run(
        cfg=cfg,
        run_name=run_name,
        force=force_inference_output,
    )
    data_root = common["data_root"]
    common_splits = resolve_splits_subdir(
        common.get("splits_subdir"),
        splits_subdir,
        data_root,
    )

    trainer = BAPMOSTrainer(cfg)
    requested = resolve_torch_device(device)
    if trainer.device != requested:
        trainer.device = requested
        trainer.model.to(requested)

    organ_keys = list(cfg.get("bandit", {}).get("organs", []))
    organ_to_class = infer_organ_to_class(resolve_under_project(data_root), organ_keys)
    try:
        num_classes = int(get_baseline_taxonomy_profile(str(data_root)).num_classes)
    except ValueError:
        num_classes = max(organ_to_class.values()) + 1 if organ_to_class else 5
    trainer.set_data_taxonomy(organ_to_class, num_classes, data_root=data_root)

    try:
        ckpt = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=trainer.device)
    trainer.model.mask_decoder.load_state_dict(ckpt["mask_decoder"])
    if ckpt.get("bandit_state"):
        trainer.policy.load_state(ckpt["bandit_state"])

    test_arms = trainer.policy.get_best_arms_per_organ()
    trainer.test_prompt_noise_px = int(prompt_noise_px)
    trainer.test_prompt_noise_seed = int(prompt_noise_seed)
    trainer.test_prompt_noise_mode = str(prompt_noise_mode)
    trainer.test_prompt_log = {} if prompt_noise_px > 0 else None
    if prompt_mask_dir is not None:
        prompt_mask_dir = resolve_under_project(prompt_mask_dir)
        if not prompt_mask_dir.is_dir():
            raise FileNotFoundError(f"--prompt-mask-dir not found: {prompt_mask_dir}")
    taxonomy = get_baseline_taxonomy_profile(data_root)
    unit = distance_unit_short(taxonomy.taxonomy_name)

    emb_dir: Path | None = None
    if bool(common.get("use_cached_image_embeddings", False)):
        raw_emb = common.get("image_embedding_dir")
        if raw_emb:
            emb_dir = resolve_under_project(str(raw_emb))
            if not emb_dir.is_dir():
                emb_dir = None

    @torch.no_grad()
    def predict_fn(image_rgb, mask_multi, fn: str):
        """Return the 256x256 label map for one slice using each organ's highest-Q prompt."""
        trainer.model.eval()
        emb_path = None
        if emb_dir is not None:
            cand = emb_dir / f"{fn}.pt"
            if cand.is_file():
                emb_path = str(cand)
        prompt_mask = mask_multi
        if prompt_mask_dir is not None:
            prompt_mask = load_prompt_mask(prompt_mask_dir, fn, np.asarray(mask_multi).shape)
        out = trainer._forward_sample(
            image_rgb,
            prompt_mask,
            fn,
            test_arms,
            train=False,
            embedding_path=emb_path,
        )
        if out is None:
            return None
        logits, _gt = out
        pred_lr = (
            torch.argmax(torch.softmax(logits, dim=1), dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return pred_lr

    ablation_id = str(cfg.get("ablation", {}).get("id", "bapmos"))
    experiment = str(cfg.get("experiment_name", run_dir.name))
    method_slug = f"bapmos_{ablation_id}_{experiment}"
    viz_kwargs = export_viz_kwargs_for_data_root(
        data_root,
        per_patient_max=viz_per_patient_max,
        viz_pdf_max=viz_pdf_max,
    )
    export_stratified_sites(
        data_root=data_root,
        output_dir=Path(output_dir),
        splits_subdir=common_splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=unit,
        checkpoint=checkpoint,
        method_slug=method_slug,
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "bapmos",
            "ablation_id": ablation_id,
            "experiment_name": experiment,
            "test_prompt_arms": dict(test_arms),
            "test_site": label,
            "prompt_source": "external_mask" if prompt_mask_dir is not None else "gt_mask",
            "prompt_mask_dir": str(prompt_mask_dir) if prompt_mask_dir is not None else None,
            "prompt_noise_px": int(prompt_noise_px),
            "prompt_noise_seed": int(prompt_noise_seed),
            "prompt_noise_mode": str(prompt_noise_mode),
        },
    )
    if trainer.test_prompt_log is not None:
        write_prompt_log(
            trainer.test_prompt_log,
            Path(output_dir),
            {
                "noise_px": int(prompt_noise_px),
                "mode": str(prompt_noise_mode),
                "seed": int(prompt_noise_seed),
                "checkpoint": str(checkpoint),
            },
        )


def load_prompt_mask(mask_dir: Path, filename: str, shape: tuple) -> np.ndarray:
    """Predicted label map ``<stem>_pred_ids.png`` used as the prompt source for one slice.

    ``mask_dir`` is either a ``predictions/multiclass`` folder or an export root whose
    sites hold ``<site>/predictions/multiclass/``. Raises when the file is missing, found
    more than once, or its shape differs from the GT grid, so a missing prediction is
    never silently replaced by the GT mask.
    """
    from bapmos.inference_output.export_bundle import _batch_filename_stem

    name = f"{_batch_filename_stem(filename)}_pred_ids.png"
    hits = [Path(mask_dir) / name] if (Path(mask_dir) / name).is_file() else sorted(
        Path(mask_dir).glob(f"*/predictions/multiclass/{name}")
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Prompt mask for {filename}: expected one {name} under {mask_dir}, found {len(hits)}"
        )
    path = hits[0]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Prompt mask unreadable for {filename}: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if tuple(mask.shape) != tuple(shape[:2]):
        raise ValueError(
            f"Prompt mask {path} has shape {mask.shape}, expected GT grid {shape[:2]}"
        )
    return mask.astype(np.uint8)


def main() -> None:
    """CLI wrapper around :func:`run_bapmos_test_export`."""
    p = argparse.ArgumentParser(
        description=(
            "Export bapmos.method test split under inference_output/ "
            "(primary seed-42 run only; first-10 pred+panel PNG/PDF @ 350 dpi). "
            "Pooled prostate writes one subfolder per site_tests site."
        )
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits-subdir", type=str, default=None)
    p.add_argument(
        "--viz-per-patient-max",
        type=int,
        default=None,
        help="Deprecated (ignored); protocol exports first-N images + PDF.",
    )
    p.add_argument(
        "--viz-pdf-max",
        type=int,
        default=None,
        help=(
            "Max prediction PNGs + overlay/difference PNG/PDF panels "
            "(default 10; same first-N stems across methods)."
        ),
    )
    p.add_argument(
        "--force-inference-output",
        action="store_true",
        help="Allow export for non-primary (non-seed-42) runs.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--prompt-noise-px",
        type=int,
        default=0,
        help="Perturb each box edge / point by U{-n..n} px at test time (default 0).",
    )
    p.add_argument("--prompt-noise-seed", type=int, default=0)
    p.add_argument(
        "--prompt-noise-mode",
        choices=("uniform", "loose", "shift_paired", "loose_paired"),
        default="uniform",
        help="uniform: edges and points U{-n..n}; loose: box edges outward U{0..n}; *_paired: the same, with offsets keyed by (slice, class id) so every model family gets identical boxes.",
    )
    p.add_argument(
        "--prompt-mask-dir",
        type=str,
        default=None,
        help="Build prompts from <dir>/<stem>_pred_ids.png instead of the GT mask.",
    )
    args = p.parse_args()

    run_bapmos_test_export(
        Path(args.checkpoint),
        resolve_under_project(args.output_dir),
        device=args.device,
        splits_subdir=args.splits_subdir,
        viz_per_patient_max=args.viz_per_patient_max,
        viz_pdf_max=args.viz_pdf_max,
        force_inference_output=bool(args.force_inference_output),
        prompt_noise_px=int(args.prompt_noise_px),
        prompt_noise_seed=int(args.prompt_noise_seed),
        prompt_mask_dir=Path(args.prompt_mask_dir) if args.prompt_mask_dir else None,
        prompt_noise_mode=args.prompt_noise_mode,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
