"""Delineation / mask-overlap analysis CLI."""

from __future__ import annotations

import argparse


_COMMAND_HELP = {
    "overlays": "QA RGB overlays on prostate site split slices",
    "summarize": "RTSTRUCT/RTPLAN metadata linked to exported masks",
    "organ-presence": "Per-slice organ presence audit from combined masks",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bapmos.preprocess.delineation",
        description="Mask overlays and RTSTRUCT/RTPLAN delineation summaries",
    )
    p.add_argument(
        "command",
        choices=("overlays", "summarize", "organ-presence", "help"),
        nargs="?",
        default="help",
        help="Which analysis to run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned step without running",
    )
    args, rest = p.parse_known_args(argv)

    modules = {
        "overlays": "bapmos.preprocess.delineation.mask_overlays",
        "summarize": "bapmos.preprocess.delineation.summarize_rtstruct_rtplan_masks",
        "organ-presence": "bapmos.preprocess.delineation.report_slice_organ_presence",
    }
    if args.command == "help":
        print("Delineation analysis commands:")
        for key, mod in modules.items():
            print(f"  {key:16s}  {_COMMAND_HELP[key]}")
            print(f"  {'':16s}  python -m {mod}")
        print("Also: python -m bapmos.preprocess.bladder.visualize_samples  (PFUS1 overlays)")
        return 0

    if args.dry_run:
        print(f"dry-run: python -m {modules[args.command]} {' '.join(rest)}".rstrip())
        return 0

    if args.command == "overlays":
        from bapmos.preprocess.delineation.mask_overlays import main as run

        return int(run(rest))
    if args.command == "summarize":
        from bapmos.preprocess.delineation.summarize_rtstruct_rtplan_masks import main as run

        return int(run(rest))
    from bapmos.preprocess.delineation.report_slice_organ_presence import main as run

    return int(run(rest))


if __name__ == "__main__":
    raise SystemExit(main())
