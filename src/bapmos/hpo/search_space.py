"""Optuna search space for BAP-MOS outer-loop HPO.

Canonical profile: ``searched_clip_scale_organ`` (composite_fixed_clip + organ-scale
ring). Searched dimensions (see ``docs/SEARCH_METHODS.md``)::

    clip_max_mm, alpha, r_min, window_size, block_size_batches

Legacy curriculum / fixed-ring search boxes were removed; use this profile only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import optuna


def searched_clip_scale_organ_baseline_overrides() -> Dict[str, Any]:
    """Enqueue baseline: composite reward + organ-scale ring mid-range defaults."""
    return {
        "reward": {
            "mode": "composite_fixed_clip",
            "clip_max_mm": 20.0,
            "hd95_lambda": 3.0,
        },
        "prompt_geometry": {
            "ring_mode": "scale_organ",
            "alpha": 0.1,
            "r_min": 6,
        },
        "bandit": {
            "memory": "sliding",
            "window_size": 50,
            "block_size_batches": 50,
        },
    }



def trial_overrides_searched_clip_scale_organ(trial: "optuna.Trial") -> Dict[str, Any]:
    """Composite bandit reward + searched τ + organ-ring + bandit window/block."""
    return {
        "reward": {
            "mode": "composite_fixed_clip",
            "clip_max_mm": float(trial.suggest_int("clip_max_mm", 5, 30)),
            "hd95_lambda": 3.0,
        },
        "prompt_geometry": {
            "ring_mode": "scale_organ",
            "alpha": trial.suggest_float("alpha", 0.05, 0.15),
            "r_min": trial.suggest_int("r_min", 2, 20),
        },
        "bandit": {
            "memory": "sliding",
            "window_size": trial.suggest_int("window_size", 20, 100, step=5),
            "block_size_batches": trial.suggest_int("block_size_batches", 20, 100, step=5),
        },
    }


def flat_params_searched_clip_scale_organ(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Flat 5-D Optuna params only (bandit memory is fixed to sliding, not searched)."""
    geom = overrides.get("prompt_geometry", {})
    bandit = overrides.get("bandit", {})
    reward = overrides.get("reward", {})
    return {
        "clip_max_mm": reward.get("clip_max_mm", 20.0),
        "alpha": geom.get("alpha"),
        "r_min": geom.get("r_min"),
        "window_size": bandit.get("window_size"),
        "block_size_batches": bandit.get("block_size_batches"),
    }


def overrides_from_flat_searched_clip_scale_organ(params: Dict[str, Any]) -> Dict[str, Any]:
    """Flat Optuna params → composite reward + searched τ (λ fixed at 3)."""
    return {
        "reward": {
            "mode": "composite_fixed_clip",
            "clip_max_mm": float(params.get("clip_max_mm", 20.0)),
            "hd95_lambda": 3.0,
        },
        "prompt_geometry": {
            "ring_mode": "scale_organ",
            "alpha": float(params["alpha"]),
            "r_min": int(params["r_min"]),
        },
        "bandit": {
            "memory": "sliding",
            "window_size": int(params["window_size"]),
            "block_size_batches": int(params["block_size_batches"]),
        },
    }


@dataclass(frozen=True)
class SearchSpaceProfile:
    profile_id: str
    description: str
    trial_overrides: Callable[["optuna.Trial"], Dict[str, Any]]
    baseline_overrides: Callable[[], Dict[str, Any]]
    overrides_to_flat: Callable[[Dict[str, Any]], Dict[str, Any]]
    flat_to_overrides: Callable[[Dict[str, Any]], Dict[str, Any]]


SEARCH_SPACE_PROFILES: Dict[str, SearchSpaceProfile] = {
    "searched_clip_scale_organ": SearchSpaceProfile(
        profile_id="searched_clip_scale_organ",
        description=(
            "BO-searched fixed clip τ + organ-scale ring "
            "(composite_fixed_clip, λ=3; 5-D paper search box)"
        ),
        trial_overrides=trial_overrides_searched_clip_scale_organ,
        baseline_overrides=searched_clip_scale_organ_baseline_overrides,
        overrides_to_flat=flat_params_searched_clip_scale_organ,
        flat_to_overrides=overrides_from_flat_searched_clip_scale_organ,
    ),
}

# Live HPO suites → the single paper search box.
HPO_SUITE_SEARCH_PROFILE: Dict[str, str] = {
    "bapmos_bo": "searched_clip_scale_organ",
    "bapmos_bo_random": "searched_clip_scale_organ",
    "bapmos_bo_heuristic": "searched_clip_scale_organ",
    "bapmos_bo_greedy": "searched_clip_scale_organ",
    "bapmos_bo_medsam_pooled": "searched_clip_scale_organ",
    "bapmos_bo_sam": "searched_clip_scale_organ",
    "bapmos_bo_medsam": "searched_clip_scale_organ",
}


def search_profile_for_suite(hpo_suite: str) -> SearchSpaceProfile:
    key = hpo_suite.lower().replace("-", "_")
    profile_id = HPO_SUITE_SEARCH_PROFILE.get(key)
    if profile_id is None:
        choices = ", ".join(sorted(HPO_SUITE_SEARCH_PROFILE))
        raise ValueError(
            f"Unknown HPO suite {hpo_suite!r} for search space; choose from: {choices}. "
            "Only searched_clip_scale_organ is supported."
        )
    return SEARCH_SPACE_PROFILES[profile_id]


def trial_overrides_for_suite(hpo_suite: str, trial: "optuna.Trial") -> Dict[str, Any]:
    return search_profile_for_suite(hpo_suite).trial_overrides(trial)


def baseline_overrides_for_suite(hpo_suite: str) -> Dict[str, Any]:
    return search_profile_for_suite(hpo_suite).baseline_overrides()


def overrides_to_flat_params(hpo_suite: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    return search_profile_for_suite(hpo_suite).overrides_to_flat(overrides)


def flat_params_to_overrides(hpo_suite: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return search_profile_for_suite(hpo_suite).flat_to_overrides(params)


# Default aliases → searched_clip_scale_organ (paper box).
def trial_overrides(trial: "optuna.Trial") -> Dict[str, Any]:
    return trial_overrides_searched_clip_scale_organ(trial)


def overrides_to_flat_params_legacy(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return flat_params_searched_clip_scale_organ(overrides)


def flat_params_to_overrides_legacy(params: Dict[str, Any]) -> Dict[str, Any]:
    return overrides_from_flat_searched_clip_scale_organ(params)
