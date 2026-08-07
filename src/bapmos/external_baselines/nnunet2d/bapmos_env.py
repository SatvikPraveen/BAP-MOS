"""Shared nnU-Net + BAP-MOS env resolution (data root, checkpoint preflight)."""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence


def resolve_bapmos_data_root() -> Optional[str]:
    """Training data root for taxonomy / boundary metrics (first set env wins)."""
    for key in ("BAPMOS_NNUNET_DATA_ROOT", "BAPMOS_DATA_ROOT", "PTV_DATA_ROOT"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def resolve_boundary_tmpdir() -> str:
    """Scratch root for epoch boundary prediction PNGs (not home / not tiny node /tmp).

    Layout (preferred)::

        $NNUNET_BOUNDARY_TMPDIR/<run_or_seed>/   # set by Slurm per seed
        or $BAP_MOS_EXPORTS_ROOT/../tmp/nnunet_boundary/<run_or_seed>/

    When ``NNUNET_BOUNDARY_TMPDIR`` already points at a seed-specific directory,
    that path is used as-is. Otherwise we append ``RUN_NAME`` / ``SEED`` so
    concurrent seeds never share the same scratch folder.
    """
    from pathlib import Path

    explicit = os.environ.get("NNUNET_BOUNDARY_TMPDIR", "").strip()
    if explicit:
        root = Path(explicit).expanduser()
    else:
        exports = os.environ.get("BAP_MOS_EXPORTS_ROOT", "").strip()
        if exports:
            root = Path(exports).expanduser().resolve().parent / "tmp" / "nnunet_boundary"
        else:
            # Portable default under the checkout (override via NNUNET_BOUNDARY_TMPDIR).
            from bapmos.paths import project_root

            root = project_root() / "tmp" / "nnunet_boundary"

    # If caller already gave a seed-specific dir, don't double-nest.
    run_tag = (
        os.environ.get("RUN_NAME", "").strip()
        or (
            f"pfus1_seed{os.environ['SEED']}"
            if os.environ.get("SEED", "").strip().isdigit()
            else ""
        )
    )
    if run_tag and root.name != run_tag and run_tag not in root.parts:
        root = root / run_tag

    root.mkdir(parents=True, exist_ok=True)
    return str(root)

def resolve_evaluator_organ_labels(data_root: Optional[str] = None) -> List[str]:
    """Evaluator organ labels for checkpoint scoring and boundary metrics."""
    root = (data_root or resolve_bapmos_data_root() or "").strip()
    if not root:
        return []
    try:
        from bapmos.training_taxonomy import get_baseline_taxonomy_profile

        return list(get_baseline_taxonomy_profile(root).evaluator_organ_labels)
    except Exception:
        return []


def checkpoint_objective_metric() -> str:
    raw = os.environ.get("NNUNET_CHECKPOINT_OBJECTIVE", "ptv_kfold_msd").strip()
    if raw == "ema_fg_dice":
        return "organ_balanced_msd"
    return raw


def require_checkpoint_train_env(*, exit_on_error: bool = True) -> bool:
    """
    Preflight before ``nnUNetv2_train`` when using PTV/organ k-fold checkpointing.

    Ensures a resolvable data root and (when set) a valid ``NNUNET_CHECKPOINT_OBJECTIVE_ORGAN``.
    """
    metric = checkpoint_objective_metric()
    if metric != "ptv_kfold_msd":
        return True

    root = resolve_bapmos_data_root()
    if not root:
        msg = (
            "ERROR: NNUNET_CHECKPOINT_OBJECTIVE=ptv_kfold_msd requires "
            "BAPMOS_NNUNET_DATA_ROOT, BAPMOS_DATA_ROOT, or PTV_DATA_ROOT"
        )
        if exit_on_error:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg)
        return False

    organs = resolve_evaluator_organ_labels(root)
    if not organs:
        msg = f"ERROR: no evaluator organs from data root {root!r}"
        if exit_on_error:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg)
        return False

    organ = os.environ.get("NNUNET_CHECKPOINT_OBJECTIVE_ORGAN", "").strip()
    if organ and organ not in organs:
        msg = f"ERROR: NNUNET_CHECKPOINT_OBJECTIVE_ORGAN={organ!r} not in {list(organs)}"
        if exit_on_error:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg)
        return False

    from bapmos.checkpoint_selection import resolve_ptv_evaluator_organ

    target = organ or resolve_ptv_evaluator_organ(organs) or organs[0]
    print(
        f"[nnunet preflight] k-fold checkpoint OK: data_root={root} "
        f"organs={list(organs)} objective_organ={target!r}"
    )
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    require_checkpoint_train_env(exit_on_error=True)


if __name__ == "__main__":
    main()
