"""Fixed candidate catalogs for BAP-MOS outer-loop search ablations (heuristic / greedy)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from bapmos.hpo.search_space import (
    baseline_overrides_for_suite,
    overrides_to_flat_params,
)

SEARCH_PARAM_KEYS: Tuple[str, ...] = (
    "clip_max_mm",
    "alpha",
    "r_min",
    "window_size",
    "block_size_batches",
)

GREEDY_DIM_ORDER: Tuple[str, ...] = SEARCH_PARAM_KEYS
GREEDY_GRID_SIZE = 7

# Outer-loop default (prostate + bladder TPE); 20 startup + budget 100.
OUTER_LOOP_TPE_N_STARTUP_TRIALS = 20
OUTER_LOOP_TPE_N_TRIALS = 100
TPE_N_STARTUP_TRIALS = OUTER_LOOP_TPE_N_STARTUP_TRIALS
TPE_DEFAULT_N_TRIALS = OUTER_LOOP_TPE_N_TRIALS
RANDOM_DEFAULT_N_TRIALS = OUTER_LOOP_TPE_N_TRIALS
GREEDY_DEFAULT_N_TRIALS = OUTER_LOOP_TPE_N_TRIALS
HEURISTIC_DEFAULT_N_TRIALS = OUTER_LOOP_TPE_N_TRIALS

# Deprecated aliases (pre-rename); same values as OUTER_LOOP_TPE_*.
INNER_LOOP_TPE_N_STARTUP_TRIALS = OUTER_LOOP_TPE_N_STARTUP_TRIALS
INNER_LOOP_TPE_N_TRIALS = OUTER_LOOP_TPE_N_TRIALS


def _param_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(params[k] for k in SEARCH_PARAM_KEYS)


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for params in candidates:
        key = _param_key(params)
        if key in seen:
            continue
        seen.add(key)
        out.append(params)
    return out


def baseline_flat_params(*, hpo_suite: str = "bapmos_bo") -> Dict[str, Any]:
    return overrides_to_flat_params(hpo_suite, baseline_overrides_for_suite(hpo_suite))


def _snap_stepped_int(value: float, *, low: int, high: int, step: int) -> int:
    snapped = int(round(value / step) * step)
    return max(low, min(high, snapped))


def greedy_grid_values(dim: str) -> List[Any]:
    """Seven evenly spaced probe values for one greedy coordinate step."""
    if dim == "clip_max_mm":
        return [5, 9, 13, 17, 21, 25, 30]
    if dim == "alpha":
        return [0.05, 0.0667, 0.0833, 0.10, 0.1167, 0.1333, 0.15]
    if dim == "r_min":
        return [2, 5, 8, 11, 14, 17, 20]
    if dim in ("window_size", "block_size_batches"):
        raw = [20, 33, 46, 60, 73, 86, 100]
        return [_snap_stepped_int(v, low=20, high=100, step=5) for v in raw]
    raise ValueError(f"Unknown greedy dimension {dim!r}")


def _heuristic_grid_levels() -> Dict[str, List[Any]]:
    """Coarse grid levels aligned with the outer-loop / greedy probe ranges."""
    return {
        "clip_max_mm": list(greedy_grid_values("clip_max_mm")),
        "alpha": list(greedy_grid_values("alpha")),
        "r_min": list(greedy_grid_values("r_min")),
        "window_size": [20, 40, 60, 80, 100],
        "block_size_batches": [20, 40, 60, 80, 100],
    }


def heuristic_catalog(
    *,
    hpo_suite: str = "bapmos_bo",
    max_trials: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic fixed catalog for heuristic outer-loop (no adaptive sampler).

    Candidates are assembled in priority tiers, deduplicated, then capped at
    ``max_trials`` (default 100 — same budget as TPE/random/greedy):

    1. Enqueue baseline (searched_clip_scale_organ defaults)
    2. Univariate sweeps (one of clip / alpha / r_min / window / block varied)
    3. Sparse 3-D geometry lattice (clip × alpha × r_min; bandit at baseline)
    4. Bandit joint grid (window × block; geometry at baseline)

    All points lie in the same 5-D ``searched_clip_scale_organ`` box used by TPE.
    """
    budget = HEURISTIC_DEFAULT_N_TRIALS if max_trials is None else int(max_trials)
    if budget < 1:
        raise ValueError(f"heuristic trial budget must be >= 1, got {budget}")

    base = baseline_flat_params(hpo_suite=hpo_suite)
    levels = _heuristic_grid_levels()
    candidates: List[Dict[str, Any]] = []

    def add(params: Dict[str, Any]) -> None:
        candidates.append(dict(params))

    # Tier 1 — baseline anchor
    add(base)

    # Tier 2 — univariate sweeps
    for clip in levels["clip_max_mm"]:
        add({**base, "clip_max_mm": clip})
    for alpha in levels["alpha"]:
        add({**base, "alpha": alpha})
    for r_min in levels["r_min"]:
        add({**base, "r_min": r_min})
    for window_size in levels["window_size"]:
        add({**base, "window_size": window_size})
    for block_size_batches in levels["block_size_batches"]:
        add({**base, "block_size_batches": block_size_batches})

    # Tier 3 — sparse geometry lattice (corners + mid-box of clip/alpha/r_min)
    for clip in (5, 13, 21, 30):
        for alpha in (0.05, 0.10, 0.15):
            for r_min in (2, 8, 14, 20):
                add(
                    {
                        **base,
                        "clip_max_mm": clip,
                        "alpha": alpha,
                        "r_min": r_min,
                    }
                )

    # Tier 4 — bandit window/block joint grid (geometry fixed at baseline)
    for window_size in levels["window_size"]:
        for block_size_batches in levels["block_size_batches"]:
            add(
                {
                    **base,
                    "window_size": window_size,
                    "block_size_batches": block_size_batches,
                }
            )

    deduped = _dedupe_candidates(candidates)
    return deduped[:budget]


def greedy_wave_candidates(
    current: Dict[str, Any],
    dim_index: int,
) -> List[Dict[str, Any]]:
    """Build one greedy coordinate wave (seven trials, one dimension varied)."""
    dim = GREEDY_DIM_ORDER[dim_index]
    return [{**current, dim: value} for value in greedy_grid_values(dim)]


def greedy_trial_plan(*, max_trials: int | None = None) -> List[Dict[str, Any]]:
    """
    Deterministic greedy schedule up to ``max_trials`` (default 100, same as TPE/random).

    One baseline trial, then repeated coordinate sweeps (5 dims × 7 probes) from the
    current best until the budget is exhausted. The final wave may be partial.
    """
    budget = GREEDY_DEFAULT_N_TRIALS if max_trials is None else int(max_trials)
    if budget < 1:
        raise ValueError(f"greedy trial budget must be >= 1, got {budget}")

    steps: List[Dict[str, Any]] = []
    remaining = budget

    steps.append({"kind": "baseline", "round": -1, "wave": -1, "n_probes": 1})
    remaining -= 1

    round_idx = 0
    while remaining > 0:
        for wave in range(len(GREEDY_DIM_ORDER)):
            if remaining <= 0:
                break
            n_probes = min(GREEDY_GRID_SIZE, remaining)
            steps.append(
                {
                    "kind": "wave",
                    "round": round_idx,
                    "wave": wave,
                    "n_probes": n_probes,
                }
            )
            remaining -= n_probes
        round_idx += 1

    return steps


def greedy_total_trials(*, max_trials: int | None = None) -> int:
    """Total enqueued trials for the greedy schedule (default 100)."""
    return sum(step["n_probes"] for step in greedy_trial_plan(max_trials=max_trials))


def expected_trial_count(hpo_suite: str) -> int:
    """Default worker count for a prostate outer-loop search method."""
    from bapmos.hpo.paths import HpoPaths

    paths = HpoPaths.for_suite(hpo_suite)
    method = paths.spec.get("search_method", "tpe")
    if method == "tpe":
        return int(paths.spec.get("default_n_trials", TPE_DEFAULT_N_TRIALS))
    if method == "random":
        return int(paths.spec.get("default_n_trials", RANDOM_DEFAULT_N_TRIALS))
    if method == "heuristic":
        configured = paths.spec.get("default_n_trials")
        budget = int(configured) if configured else HEURISTIC_DEFAULT_N_TRIALS
        return len(heuristic_catalog(hpo_suite=hpo_suite, max_trials=budget))
    if method == "greedy":
        configured = paths.spec.get("default_n_trials")
        budget = int(configured) if configured else GREEDY_DEFAULT_N_TRIALS
        return greedy_total_trials(max_trials=budget)
    raise ValueError(f"No trial budget for HPO suite {hpo_suite!r}")
