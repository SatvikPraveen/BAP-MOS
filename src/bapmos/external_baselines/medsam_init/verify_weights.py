"""CLI: verify MedSAM ViT-B weight overlap against Meta SAM ViT-B architecture."""

from __future__ import annotations

import argparse
import sys

from bapmos.external_baselines.medsam_init.weight_loader import (
    DEFAULT_MAX_MISSING_KEYS,
    DEFAULT_MIN_OVERLAP_RATIO,
    format_report_json,
    verify_medsam_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Verify MedSAM checkpoint key overlap on SAM ViT-B before training. "
            "Prints overlap ratio and missing/unexpected key counts."
        )
    )
    p.add_argument(
        "--sam-checkpoint",
        type=str,
        default="models/sam_base/sam_vit_b_01ec64.pth",
        help="Meta SAM ViT-B architecture checkpoint.",
    )
    p.add_argument(
        "--medsam-checkpoint",
        type=str,
        default="models/medsam/medsam_vit_b.pth",
        help="MedSAM ViT-B weights to overlay (Zenodo medsam_vit_b.pth).",
    )
    p.add_argument(
        "--max-missing-keys",
        type=int,
        default=DEFAULT_MAX_MISSING_KEYS,
        help=f"Fail when more than this many model keys are missing (default {DEFAULT_MAX_MISSING_KEYS}).",
    )
    p.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=DEFAULT_MIN_OVERLAP_RATIO,
        help=f"Fail when matched/model ratio is below this (default {DEFAULT_MIN_OVERLAP_RATIO}).",
    )
    args = p.parse_args(argv)

    try:
        report = verify_medsam_checkpoint(
            args.sam_checkpoint,
            args.medsam_checkpoint,
            max_missing_keys=args.max_missing_keys,
            min_overlap_ratio=args.min_overlap_ratio,
        )
    except Exception as exc:
        print(f"VERIFY FAILED: {exc}", file=sys.stderr)
        return 1

    print(format_report_json(report))
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
