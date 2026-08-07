"""
Historical ablation/sweep **helpers** (not the main paper training path).

Training still goes through ``bapmos.legacy.optimization.trainer`` (and baseline
modules) with YAML + CLI overrides. This package holds naming presets and path
constants so ablation SLURM and configs stay organized.

See ``docs/EXPERIMENT_LADDER.md``.
"""

from .paths import (
    ABLATION_LOG_ROOT,
    ABLATION_RUN_ROOT,
    default_ablation_run_name_prefix,
)

__all__ = [
    "ABLATION_LOG_ROOT",
    "ABLATION_RUN_ROOT",
    "default_ablation_run_name_prefix",
]
