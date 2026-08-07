"""Public bladder preprocess CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.bladder",
        description="Prepare PFUS1 bladder data under data/bladder/pfus1/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/bladder/pfus1"),
        help="Output PFUS1 bundle root (masks + splits)",
    )
    p.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Raw PFUS1 root with Pxxx/frame_*.png + .json (default: data/bladder/pfus1_raw)",
    )
    p.add_argument(
        "--step",
        choices=("all", "convert", "splits", "verify"),
        default="all",
        help="Which preprocess step to run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps without writing",
    )
    args = p.parse_args(argv)

    from bapmos.paths import pfus1_image_root, project_root
    from bapmos.preprocess.bladder.convert_json_polygons_to_masks import (
        main as convert_main,
    )
    from bapmos.preprocess.bladder.create_splits import main as splits_main
    from bapmos.preprocess.bladder.verify_label_registry import main as verify_main

    raw_path = Path(args.raw) if args.raw is not None else Path(pfus1_image_root())
    raw = raw_path if raw_path.is_absolute() else project_root() / raw_path
    raw = raw.resolve()
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out = out.resolve()
    masks_root = out / "masks"
    splits_root = out / "splits_patient_70_15_15_seed42"

    step_names: list[str] = []
    if args.step in ("all", "convert"):
        step_names.append("convert_json_polygons_to_masks")
    if args.step in ("all", "splits"):
        step_names.append("create_splits")
    if args.step in ("all", "verify"):
        step_names.append("verify_label_registry")

    print("Bladder PFUS1 preprocess")
    print(f"  raw: {raw}")
    print(f"  out: {out}")
    print(f"  step: {args.step}")
    print(f"  plan: {' → '.join(step_names)}")
    if args.dry_run:
        print("  dry-run: no files written")
        return 0

    needs_raw = args.step in ("all", "convert", "splits")
    if needs_raw and not raw.is_dir():
        raise FileNotFoundError(f"Raw PFUS1 root not found: {raw}")

    if args.step in ("all", "convert"):
        print("  run: convert_json_polygons_to_masks")
        rc = convert_main(["--raw_root", str(raw), "--masks_root", str(masks_root)])
        if rc:
            return int(rc)
    if args.step in ("all", "splits"):
        print("  run: create_splits")
        rc = splits_main(["--raw_root", str(raw), "--out_root", str(splits_root)])
        if rc:
            return int(rc)
    if args.step in ("all", "verify"):
        print("  run: verify_label_registry")
        rc = verify_main()
        if rc:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
