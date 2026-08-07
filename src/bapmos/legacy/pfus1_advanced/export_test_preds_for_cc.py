"""Export test multiclass PNGs to diagnostics/ only (for bladder CC)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from bapmos.external_baselines.unet.train_multiclass import (
    resize_for_baseline,
    imagenet_normalize_bchw,
    multiclass_pred_uint8_from_logits,
)
from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset, multi_organ_collate_fn
from bapmos.paths import pfus1_bundle_dir, project_root
from bapmos.training_taxonomy import get_baseline_taxonomy_profile


def export_method_test_preds(
    *,
    module: str,
    run_name: str,
    run_root: Path,
    export_dir: Path,
    data_root: str | None = None,
) -> List[Tuple[Path, str]]:
    """
    Run test inference and write ``{image_id}_pred.png`` under ``export_dir``.

    Returns list of (path, image_id) for slices with any foreground.
    """
    if data_root is None:
        data_root = str(pfus1_bundle_dir())
    root = project_root()
    run_dir = run_root / (
        "unet" if module == "unet" else "medsam_init"
    ) / run_name
    ckpt_path = run_dir / "best_checkpoint.pth"
    if not ckpt_path.is_file():
        ckpt_path = run_dir / "last_checkpoint.pth"
    if not ckpt_path.is_file():
        return []

    export_dir.mkdir(parents=True, exist_ok=True)
    tax = get_baseline_taxonomy_profile(data_root)
    test_ds = MultiOrganDataset(
        data_root,
        split="test",
        splits_subdir="splits_patient_70_15_15_seed42",
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=multi_organ_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if module == "unet":
        from bapmos.external_baselines.unet.train_multiclass import UNetMultiOrganTrainer
        import json

        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = {
                "data_root": data_root,
                "run_dir": str(run_dir),
                "run_name": run_name,
                "encoder": "resnet34",
                "num_classes": tax.num_classes,
                "lr": 1e-4,
                "max_epochs": 100,
                "patience": 20,
                "batch_size": 128,
                "seed": 42,
            }
        trainer = UNetMultiOrganTrainer(cfg)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.eval()

        paths: List[Tuple[Path, str]] = []
        for batch in test_loader:
            image_rgb = batch["image"][0]
            mask_multi = batch["mask"][0]
            if isinstance(mask_multi, torch.Tensor):
                mask_multi = mask_multi.numpy()
            if isinstance(image_rgb, torch.Tensor):
                image_rgb = image_rgb.numpy()
            image_id = str(batch["filename"][0])
            if mask_multi.max() == 0:
                continue
            x, y = resize_for_baseline(image_rgb, mask_multi, size=256)
            xb = imagenet_normalize_bchw(x.to(device))
            with torch.no_grad():
                logits = trainer.model(xb)
            pred_u8 = multiclass_pred_uint8_from_logits(logits)[0]  # H,W at 256
            # Resize pred to native GT size for fair CC vs combined masks
            h, w = mask_multi.shape[:2]
            pred_native = cv2.resize(
                pred_u8.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            out_path = export_dir / f"{image_id}_pred.png"
            cv2.imwrite(str(out_path), pred_native)
            paths.append((out_path, image_id))
        return paths

    if module == "medsam_init":
        # MedSAM path: box-only multiclass — skip if too heavy; return empty
        return []

    return []
