"""Precompute frozen SAM image embeddings for the pooled prostate corpus (shared, once).

The BAP-MOS image encoder is frozen and prostate training applies no image augmentation
(only prompt jitter), so every run encodes identical images to identical embeddings.
This writes them once, through the trainer's own ``FrozenSAMMultiOrgan.encode_image`` on
the images exactly as ``MultiOrganDataset`` loads them, stored in float32::

    <out_dir>/<filename>.pt   {"image_embedding", "original_size", "checkpoint"}
    <out_dir>/manifest.json   counts, checkpoint and the on-the-fly check below

Before writing the manifest, ``--verify N`` re-encodes N slices on the fly and records
the largest absolute difference to the stored embedding; the job fails if it exceeds
``--tol``. Runs read the cache through ``common.use_cached_image_embeddings``.

    python -m bapmos.preprocess.prostate.precompute_sam_embeddings \\
      --out-dir data/prostate/pooled/sam_vit_b_embeddings --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from bapmos.method.bap_mos_trainer import BAPMOSTrainer
from bapmos.method.sam_model import FrozenSAMMultiOrgan
from bapmos.multiorgan.dataset_multi_organ import MultiOrganDataset
from bapmos.paths import pooled_prostate_dataset_dir


def pooled_split_dirs(root: Path) -> List[str]:
    """Every split folder under the pooled root that lists slices (relative paths)."""
    subs = ["splits_stratified"]
    subs += [f"site_tests/{p.parent.name}" for p in sorted((root / "site_tests").glob("*/test.txt"))]
    subs += [str(p.parent.relative_to(root)) for p in sorted(root.glob("splits_loco/*/test.txt"))]
    return subs


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sam-checkpoint", default="models/sam_base/sam_vit_b_01ec64.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify", type=int, default=8, help="Slices re-encoded to check the cache")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args(argv)

    root = pooled_prostate_dataset_dir()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = FrozenSAMMultiOrgan(args.sam_checkpoint, init="sam").to(device).eval()
    ckpt_name = Path(args.sam_checkpoint).name

    seen: dict = {}
    for sub in pooled_split_dirs(root):
        for split in ("train", "val", "test"):
            if not (root / sub / f"{split}.txt").is_file():
                continue
            ds = MultiOrganDataset(str(root), split=split, splits_subdir=sub)
            for i in range(len(ds)):
                sample = ds[i]
                name = str(sample["filename"])
                if name in seen:
                    continue
                # Same conversion as BAPMOSTrainer._forward_sample before encoding.
                image = BAPMOSTrainer._as_numpy_hwc(sample["image"])
                with torch.no_grad():
                    emb, original_size = model.encode_image(image, device)
                torch.save(
                    {
                        "image_embedding": emb.detach().cpu().float(),
                        "original_size": tuple(int(v) for v in original_size),
                        "checkpoint": ckpt_name,
                    },
                    out_dir / f"{name}.pt",
                )
                seen[name] = image

    # Re-encode a few slices on the fly and compare with what was stored.
    max_diff = 0.0
    for name in sorted(seen)[: max(0, args.verify)]:
        with torch.no_grad():
            emb, _ = model.encode_image(seen[name], device)
        stored = torch.load(out_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        max_diff = max(max_diff, float((emb.cpu().float() - stored["image_embedding"]).abs().max()))
    if max_diff > args.tol:
        raise SystemExit(f"cached vs on-the-fly embeddings differ by {max_diff:.3g} > {args.tol}")

    manifest = {
        "slices": len(seen),
        "checkpoint": ckpt_name,
        "dtype": "float32",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "verify_slices": min(len(seen), max(0, args.verify)),
        "verify_max_abs_diff": max_diff,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
