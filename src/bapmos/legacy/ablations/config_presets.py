"""
Documentation-oriented presets for ablation studies (paper plan).

These are **not** merged automatically into YAML; they list keys to edit when
you copy a baseline config for an ablation sweep. Keeps the “what to
sweep” matrix in code for grep-ability.

See ``docs/EXPERIMENT_LADDER.md``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Reviewer ablation matrix: reward design (YAML / trainer keys depend on your config schema)
REWARD_ABLATION_KEYS: Dict[str, Dict[str, Any]] = {
    "proposed": {
        "description": "MSD, organ-balanced, clipped (current default).",
        "typical_yaml_paths": ("reward", "bandit", "optimization"),
        "notes": "Align with reward_clip_max_msd_mm and organ-balanced val MSD.",
    },
    "A1_unclipped_msd": {
        "description": "MSD organ-balanced without clip (or very large clip).",
        "suggested_change": "Increase reward_clip_max_msd_mm or disable clipping in trainer if supported.",
    },
    "A2_hd95_reward": {
        "description": "Use HD95-based reward instead of MSD (requires trainer branch).",
        "suggested_change": "Not wired by default — implement in OptimizationTrainer reward path only under bapmos.legacy.optimization.",
    },
    "A3_dice_reward": {
        "description": "Dice-based reward (requires trainer branch).",
    },
    "A4_size_weighted": {
        "description": "Non-organ-balanced aggregation (requires trainer branch).",
    },
}

UCB_EXPLORATION_GRID: List[float] = [0.5, 1.0, 2.0, 4.0]

CLIP_THRESHOLD_GRID_MM: List[Any] = [5.0, 10.0, 20.0, None]  # None = document as no clip

# Internal datasets (simulation, Case 1/2): smaller probe/block grids for tiny cohorts.
INTERNAL_PROBE_SIZE_GRID: List[int] = [3, 5, 10]
INTERNAL_BLOCK_SIZE_GRID: List[int] = [10, 25, 50]

# PFUS1 / larger cohorts: defaults aligned with full-dataset ablation plan.
PFUS1_PROBE_SIZE_GRID: List[int] = [5, 10, 20]
PFUS1_BLOCK_SIZE_GRID: List[int] = [25, 50, 100]

# Backward-compatible aliases (PFUS1-scale defaults).
PROBE_SIZE_GRID: List[int] = PFUS1_PROBE_SIZE_GRID
BLOCK_SIZE_GRID: List[int] = PFUS1_BLOCK_SIZE_GRID
