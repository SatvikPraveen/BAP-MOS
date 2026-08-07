"""Print shell exports for one infer-only diagnostic (used by Slurm)."""

from __future__ import annotations

import sys

from bapmos.legacy.pfus1_advanced.infer_only_diagnostics import discover_infer_only_diagnostics
from bapmos.paths import project_root


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m bapmos.legacy.pfus1_advanced.infer_only_diag_env <diag_id>", file=sys.stderr)
        return 2
    diag_id = sys.argv[1]
    root = project_root()
    for d in discover_infer_only_diagnostics():
        if d.diag_id != diag_id:
            continue
        ckpt = d.checkpoint.resolve()
        out = d.output_dir.resolve()
        print(f"export CHECKPOINT={ckpt}")
        print(f"export DATA_ROOT={root / d.data_root}")
        print(f"export OUTPUT_DIR={out}")
        print(f"export SPLITS_SUBDIR={d.splits_subdir}")
        if d.prompt_config is not None:
            print(f"export PROMPT_CONFIG={d.prompt_config.resolve()}")
        else:
            print("unset PROMPT_CONFIG")
        return 0
    print(f"unknown diag_id: {diag_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
