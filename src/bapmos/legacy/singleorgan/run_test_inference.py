"""Stratified test export for single-organ SAM checkpoints (Table I single-organ rows).

Same export path and scoring as the multi-organ runs (``export_stratified_sites``: native
GT grid, pooled prostate one bundle per ``site_tests/<site>``). The binary prediction is
written into the organ's class of the multi-class label map, so ``summary_metrics.csv``
holds the trained organ's Dice / HD95 / MSD; the other organs have empty predictions and
carry no information. Prompts come from the trained organ's GT mask, as in training.

    python -m bapmos.legacy.singleorgan.run_test_inference \\
      --checkpoint runs/.../best_checkpoint.pth --output_dir inference_output/... \\
      --force-inference-output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.inference_output.pfus1_defaults import export_viz_kwargs_for_data_root
from bapmos.inference_output.protocol import require_primary_inference_seed_run
from bapmos.inference_output.site_export import export_stratified_sites, resolve_splits_subdir
from bapmos.paths import resolve_under_project
from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

from .inference_singleorgan import SingleOrganInference


def _as_numpy_hwc(image) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def run_singleorgan_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    splits_subdir: str | None = None,
    viz_pdf_max: int | None = None,
    force_inference_output: bool = False,
    device: str | None = None,
) -> None:
    """Export the test split of one single-organ checkpoint under ``output_dir``."""
    checkpoint = resolve_under_project(checkpoint)
    engine = SingleOrganInference(checkpoint, device=device)
    cfg = engine.config
    run_name = str(cfg.get("run_name") or checkpoint.parent.name)
    require_primary_inference_seed_run(cfg=cfg, run_name=run_name, force=force_inference_output)

    data_root = cfg["data_root"]
    taxonomy = get_baseline_taxonomy_profile(data_root)
    class_id = int(taxonomy.organ_to_class[engine.organ])
    splits = resolve_splits_subdir(cfg.get("splits_subdir"), splits_subdir, data_root)

    @torch.no_grad()
    def predict_fn(image_rgb, mask_multi, fn: str):
        """Native-grid label map: the organ's class where the binary prediction is 1."""
        mask_multi = np.asarray(mask_multi.numpy() if torch.is_tensor(mask_multi) else mask_multi)
        organ_gt = (mask_multi == class_id).astype(np.uint8)
        if organ_gt.sum() == 0:
            return np.zeros(mask_multi.shape[:2], dtype=np.uint8)
        pred = engine.predict_logits(_as_numpy_hwc(image_rgb), organ_gt, Path(str(fn)).stem)
        if pred is None:
            return None
        return np.where(pred > 0, class_id, 0).astype(np.uint8)

    export_stratified_sites(
        data_root=data_root,
        output_dir=Path(output_dir),
        splits_subdir=splits,
        predict_fn=predict_fn,
        taxonomy=taxonomy,
        distance_unit=distance_unit_short(taxonomy.taxonomy_name),
        checkpoint=checkpoint,
        method_slug=f"sam_single_{engine.organ}_{engine.prompt_type}",
        viz_kwargs=export_viz_kwargs_for_data_root(data_root, viz_pdf_max=viz_pdf_max),
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "singleorgan_sam",
            "organ": engine.organ,
            "prompt_type": engine.prompt_type,
            "finetune_scope": cfg.get("finetune_scope", "decoder"),
            "run_name": run_name,
            "test_site": label,
        },
    )


def main() -> None:
    """CLI wrapper around :func:`run_singleorgan_test_export`."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits-subdir", type=str, default=None)
    p.add_argument("--viz-pdf-max", type=int, default=None)
    p.add_argument("--force-inference-output", action="store_true")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    run_singleorgan_test_export(
        Path(args.checkpoint),
        resolve_under_project(args.output_dir),
        splits_subdir=args.splits_subdir,
        viz_pdf_max=args.viz_pdf_max,
        force_inference_output=bool(args.force_inference_output),
        device=args.device,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
