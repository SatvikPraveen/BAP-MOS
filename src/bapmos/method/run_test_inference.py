"""Stratified test-split inference export for ``bapmos.method`` checkpoints.

Loss / decoder outputs are at 256×256; exported label maps are nearest-neighbor
resized to native GT resolution for visualization. Boundary MSD/HD95 default to
``PFUS1_BASELINE_EVAL_SIZE`` (256) for cross-method parity.

Pooled prostate uses ``site_tests/<site>/test.txt`` (no global ``test.txt``).
Each site is exported under ``<output_dir>/<site>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bapmos.evaluation.test_panels import distance_unit_short
from bapmos.external_baselines.baseline_training_protocol import PFUS1_BASELINE_EVAL_SIZE
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


def _resize_pred(pred_lr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample of class map to native ``(H, W)`` for viz / export."""
    h, w = hw
    return cv2.resize(pred_lr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )


def run_bapmos_test_export(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    splits_subdir: str | None = None,
    viz_per_patient_max: int | None = None,
    viz_pdf_max: int | None = None,
    force_inference_output: bool = False,
    eval_size: int | None = PFUS1_BASELINE_EVAL_SIZE,
) -> None:
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
        trainer.model.eval()
        emb_path = None
        if emb_dir is not None:
            cand = emb_dir / f"{fn}.pt"
            if cand.is_file():
                emb_path = str(cand)
        out = trainer._forward_sample(
            image_rgb,
            mask_multi,
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
        if hasattr(mask_multi, "numpy"):
            mask_multi = mask_multi.numpy()
        return _resize_pred(pred_lr, mask_multi.shape[:2])

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
        eval_size=eval_size,
        viz_kwargs=viz_kwargs,
        run_name=run_name,
        evaluation_meta_extra=lambda label: {
            "family": "bapmos",
            "ablation_id": ablation_id,
            "experiment_name": experiment,
            "test_prompt_arms": dict(test_arms),
            "test_site": label,
        },
    )


def main() -> None:
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
        "--eval-size",
        type=int,
        default=PFUS1_BASELINE_EVAL_SIZE,
        help=(
            "Square side for nearest-neighbor pred/GT resize before MSD/HD95 "
            f"(default {PFUS1_BASELINE_EVAL_SIZE}). Use 0 for native resolution."
        ),
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
        eval_size=args.eval_size,
    )
    print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
