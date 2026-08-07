"""Canonical roots for ablation sweeps (keep separate from ``runs/Optimization``)."""

from __future__ import annotations

ABLATION_RUN_ROOT = "runs/Ablation"
ABLATION_LOG_ROOT = "logs/Ablation"


def default_ablation_run_name_prefix(study: str, variant: str) -> str:
    """Human-readable run_name fragment, e.g. ``reward_A1_clip20_ucb``."""
    safe_study = study.replace("/", "_").replace(" ", "_")
    safe_var = variant.replace("/", "_").replace(" ", "_")
    return f"{safe_study}__{safe_var}"
