"""
Precompute frozen SAM image embeddings for PFUS1 frames (decoder-only training speedup).

Each output ``.pt`` uses ``ResizeLongestSide``, ``sam.preprocess``, and ``image_encoder``,
matching :class:`segment_anything.Sam` and the optimization trainer's image path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from bapmos.paths import pfus1_image_root, project_root, resolve_model_checkpoint, resolve_under_project
from bapmos.preprocess.bladder.dataset import parse_sample_line

logger = logging.getLogger(__name__)


def _read_split_lines(split_file: Path) -> List[str]:
    lines: List[str] = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def _collect_jobs(
    data_root: Path,
    splits_subdir: str,
    split_names: Sequence[str],
    limit: int,
) -> List[Tuple[str, str, Path, str]]:
    """
    Returns list of (split_name, sample_id_slash, image_path, stem_filename).

    ``stem_filename`` is ``P000_frame_000`` used for flat ``{stem}.pt`` output.
    Images are always read from ``pfus1_image_root()`` (raw tree); ``data_root``
    only supplies the split list files.
    """
    splits_dir = data_root / splits_subdir
    img_root = pfus1_image_root()
    jobs: List[Tuple[str, str, Path, str]] = []
    for split_name in split_names:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        for line in _read_split_lines(split_file):
            patient, stem = parse_sample_line(line)
            sample_id = f"{patient}/{stem}"
            img_path = img_root / patient / f"{stem}.png"
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found for split line {line!r}: {img_path}")
            flat = f"{patient}_{stem}"
            jobs.append((split_name, sample_id, img_path, flat))
            if limit > 0 and len(jobs) >= limit:
                return jobs
    return jobs


def _save_payload(
    path: Path,
    payload: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_existing_meta(
    path: Path,
) -> Tuple[str, Tuple[int, ...], Tuple[int, int]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    emb = obj["image_embedding"]
    dtype_str = str(emb.dtype).replace("torch.", "")
    inp = obj.get("input_size")
    if inp is None:
        ih, iw = -1, -1
    else:
        ih, iw = int(inp[0]), int(inp[1])
    return dtype_str, tuple(emb.shape), (ih, iw)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Precompute frozen SAM ViT image embeddings for PFUS1 split lines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Images are read from the raw PFUS1 tree (pfus1_image_root); --data_root only\n"
            "needs the split list files under splits_subdir.\n\n"
            "Example (full cache):\n"
            "  python -m bapmos.preprocess.bladder.precompute_sam_embeddings \\\n"
            "    --data_root data/bladder/pfus1 \\\n"
            "    --splits_subdir splits_patient_70_15_15_seed42 \\\n"
            "    --sam_checkpoint models/sam_base/sam_vit_b_01ec64.pth \\\n"
            "    --model_type vit_b \\\n"
            "    --output_dir data/bladder/pfus1/sam_embeddings/sam_vit_b \\\n"
            "    --splits train,val,test --device cuda --dtype float16\n\n"
            "Smoke (20 frames from val):\n"
            "  python -m bapmos.preprocess.bladder.precompute_sam_embeddings \\\n"
            "    --splits val --limit 20 \\\n"
            "    --output_dir data/bladder/pfus1/sam_embeddings/sam_vit_b_smoke \\\n"
            "    --dtype float16\n"
        ),
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="data/bladder/pfus1",
        help="PFUS1 bundle root containing splits_subdir (not the raw image tree).",
    )
    p.add_argument(
        "--splits_subdir",
        type=str,
        default="splits_patient_70_15_15_seed42",
    )
    p.add_argument(
        "--sam_checkpoint",
        type=str,
        default="models/sam_base/sam_vit_b_01ec64.pth",
        help="Weights file for the image encoder (Meta SAM or MedSAM .pth).",
    )
    p.add_argument(
        "--init",
        type=str,
        default="meta",
        choices=("meta", "sam", "medsam"),
        help="meta/sam: load --sam_checkpoint as Meta SAM. "
        "medsam: load Meta architecture then overlay MedSAM weights from --sam_checkpoint.",
    )
    p.add_argument(
        "--architecture_checkpoint",
        type=str,
        default="models/sam_base/sam_vit_b_01ec64.pth",
        help="Meta SAM Vit-B architecture file (only used when --init medsam).",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=("vit_b", "vit_l", "vit_h"),
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names (must match ``{split}.txt`` files).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Reserved for future batched encoder; only 1 is supported.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="Storage dtype for image_embedding on disk.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of frames to process in this run (0 = no cap).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing ``.pt`` files.",
    )
    args = p.parse_args(argv)

    if args.batch_size != 1:
        logger.warning("Only batch_size=1 is implemented; using 1.")
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_names:
        raise SystemExit("No splits given.")

    data_root = resolve_under_project(args.data_root)
    ckpt_path = resolve_model_checkpoint(args.sam_checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SAM/MedSAM checkpoint not found: {ckpt_path}")
    out_dir = resolve_under_project(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(data_root, args.splits_subdir, split_names, args.limit)
    if not jobs:
        raise SystemExit("No samples to process (empty splits or limit).")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU.")
        device = torch.device("cpu")

    if device.type == "cpu":
        # Some CPU stacks hit conv "could not create a primitive" with MKLDNN; this matches
        # a known-good smoke on shared login nodes (still slow for large PFUS1 — prefer GPU/Slurm).
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(1)
        logger.info("CPU precompute: MKLDNN off, torch intra-op threads=1 for SAM image_encoder.")

    init_key = str(args.init).lower().strip()
    if init_key == "medsam":
        meta_ckpt = resolve_model_checkpoint(args.architecture_checkpoint)
        if not meta_ckpt.is_file():
            raise FileNotFoundError(
                f"Meta SAM architecture checkpoint not found for MedSAM init: {meta_ckpt}"
            )
        from bapmos.external_baselines.medsam_init.weight_loader import (
            load_medsam_weights_into_sam_vit_b,
        )

        sam = sam_model_registry[args.model_type](checkpoint=str(meta_ckpt))
        load_medsam_weights_into_sam_vit_b(sam, str(ckpt_path))
        # Packs must report the MedSAM basename so trainers with
        # common.sam_checkpoint=.../medsam_vit_b.pth accept the cache.
        ckpt_basename = ckpt_path.name
        logger.info(
            "MedSAM image_encoder init: architecture=%s | weights=%s",
            meta_ckpt.name,
            ckpt_basename,
        )
    elif init_key in ("meta", "sam"):
        sam = sam_model_registry[args.model_type](checkpoint=str(ckpt_path))
        ckpt_basename = ckpt_path.name
    else:
        raise SystemExit(f"Unknown --init {args.init!r}")

    sam.to(device)
    sam.eval()
    for par in sam.parameters():
        par.requires_grad_(False)

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    store_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    manifest_rows: List[Dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    n_written = 0
    n_skipped = 0

    for split_name, sample_id, img_path, flat in jobs:
        split_counts[split_name] += 1
        out_pt = out_dir / f"{flat}.pt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

        if out_pt.is_file() and not args.overwrite:
            _, emb_shape, in_hw = _load_existing_meta(out_pt)
            n_skipped += 1
            patient, stem = parse_sample_line(sample_id)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": patient,
                    "frame_id": stem,
                    "split": split_name,
                    "image_path": str(img_path),
                    "embedding_path": str(out_pt),
                    "original_h": original_size[0],
                    "original_w": original_size[1],
                    "input_h": in_hw[0],
                    "input_w": in_hw[1],
                    "embedding_shape": str(emb_shape),
                    "checkpoint": ckpt_basename,
                    "model_type": args.model_type,
                }
            )
            continue

        input_image = transform.apply_image(image_rgb)
        input_size = (int(input_image.shape[0]), int(input_image.shape[1]))
        x = torch.as_tensor(input_image, device=device).float()
        x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
        x = sam.preprocess(x)

        with torch.no_grad():
            embedding = sam.image_encoder(x)

        emb_cpu = embedding.detach().cpu()
        if store_dtype == torch.float16:
            emb_cpu = emb_cpu.half()
        else:
            emb_cpu = emb_cpu.float()

        payload = {
            "image_embedding": emb_cpu,
            "original_size": original_size,
            "input_size": input_size,
            "sample_id": sample_id,
            "image_path": str(img_path),
            "checkpoint": ckpt_basename,
            "model_type": args.model_type,
        }
        _save_payload(out_pt, payload)
        n_written += 1

        patient, stem = parse_sample_line(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient,
                "frame_id": stem,
                "split": split_name,
                "image_path": str(img_path),
                "embedding_path": str(out_pt),
                "original_h": original_size[0],
                "original_w": original_size[1],
                "input_h": input_size[0],
                "input_w": input_size[1],
                "embedding_shape": str(tuple(emb_cpu.shape)),
                "checkpoint": ckpt_basename,
                "model_type": args.model_type,
            }
        )

    csv_path = out_dir / "manifest.csv"
    fieldnames = [
        "sample_id",
        "patient_id",
        "frame_id",
        "split",
        "image_path",
        "embedding_path",
        "original_h",
        "original_w",
        "input_h",
        "input_w",
        "embedding_shape",
        "checkpoint",
        "model_type",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    split_totals = dict(split_counts)
    meta_json = {
        "model_type": args.model_type,
        "sam_checkpoint": str(ckpt_path),
        "num_embeddings": len(manifest_rows),
        "splits": split_totals,
        "embedding_dtype": args.dtype,
        "image_encoder_frozen": True,
        "num_written_this_run": n_written,
        "num_skipped_existing": n_skipped,
        "data_root": str(data_root),
        "splits_subdir": args.splits_subdir,
        "output_dir": str(out_dir),
        "project_root": str(project_root()),
        "raw_image_root": str(pfus1_image_root()),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    logger.info(
        "Done: wrote %d embeddings, skipped %d existing, manifest=%s",
        n_written,
        n_skipped,
        csv_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
