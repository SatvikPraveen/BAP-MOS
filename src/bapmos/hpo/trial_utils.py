"""Helpers for reading Optuna trial params and filtering valid objectives."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

_FAILED_OBJECTIVE = 1e6

if TYPE_CHECKING:
    import optuna

SEARCH_PARAM_KEYS: tuple[str, ...] = (
    "clip_max_mm",
    "alpha",
    "r_min",
    "window_size",
    "block_size_batches",
)


def _canonical_param_value(key: str, value: Any) -> Any:
    if key == "alpha":
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def search_params_from_training_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the 5-D outer-loop search vector from a composed training config."""
    return {
        "clip_max_mm": cfg["reward"]["clip_max_mm"],
        "alpha": cfg["prompt_geometry"]["alpha"],
        "r_min": cfg["prompt_geometry"]["r_min"],
        "window_size": cfg["bandit"]["window_size"],
        "block_size_batches": cfg["bandit"]["block_size_batches"],
    }


def search_params_key(params: Dict[str, Any]) -> tuple:
    """Comparable key for the 5-D outer-loop search vector (ignores bandit_memory etc.)."""
    items = []
    for key in SEARCH_PARAM_KEYS:
        if key not in params:
            continue
        items.append((key, _canonical_param_value(key, params[key])))
    return tuple(items)


def trial_flat_params(trial: "optuna.trial.FrozenTrial") -> Dict[str, Any]:
    """
    Flat search params for a trial.

    Optuna 4.x stores ``enqueue_trial`` values in ``system_attrs['fixed_params']``
    until the trial runs; ``trial.params`` is empty while WAITING.
    """
    if trial.params:
        return dict(trial.params)
    fixed = (trial.system_attrs or {}).get("fixed_params")
    if isinstance(fixed, dict) and fixed:
        return dict(fixed)
    return {}


def _params_key(params: Dict[str, Any]) -> tuple:
    return search_params_key(params)


def find_prior_successful_trial_with_params(
    study: "optuna.Study",
    params: Dict[str, Any],
    *,
    exclude_trial: int | None = None,
) -> int | None:
    """Return an earlier COMPLETE trial number with the same search params, or None."""
    from optuna.trial import TrialState

    if not params:
        return None
    target = _params_key(params)
    for trial in study.trials:
        if exclude_trial is not None and trial.number >= exclude_trial:
            continue
        flat = trial_flat_params(trial)
        if not flat or _params_key(flat) != target:
            continue
        if trial.state == TrialState.COMPLETE and not is_poisoned_objective(trial.value):
            return int(trial.number)
    return None


def occupied_search_param_keys(
    study: "optuna.Study",
    *,
    exclude_trial: int | None = None,
) -> set[tuple]:
    """Param keys held by any non-poisoned COMPLETE or RUNNING trial (parallel guard)."""
    from optuna.trial import TrialState

    keys: set[tuple] = set()
    for trial in study.trials:
        if exclude_trial is not None and trial.number == exclude_trial:
            continue
        if trial.state == TrialState.COMPLETE:
            if is_poisoned_objective(trial.value):
                continue
        elif trial.state != TrialState.RUNNING:
            continue
        flat = trial_flat_params(trial)
        if flat:
            keys.add(_params_key(flat))
    return keys


def should_prune_duplicate_params(
    study: "optuna.Study",
    params: Dict[str, Any],
    trial_number: int,
) -> tuple[bool, str]:
    """
    True when this trial should skip training because another trial already
    holds the same 5-D params. Lowest trial number wins (WAITING/RUNNING/COMPLETE).
    """
    from optuna.trial import TrialState

    if not params:
        return False, ""
    target = _params_key(params)
    holder: int | None = None
    holder_state = ""
    for trial in study.trials:
        if trial.number == trial_number:
            continue
        flat = trial_flat_params(trial)
        if not flat or _params_key(flat) != target:
            continue
        if trial.state == TrialState.COMPLETE:
            if is_poisoned_objective(trial.value):
                continue
        elif trial.state not in (TrialState.WAITING, TrialState.RUNNING):
            continue
        if holder is None or trial.number < holder:
            holder = int(trial.number)
            holder_state = trial.state.name
    if holder is not None and holder < trial_number:
        return True, f"trial #{holder} ({holder_state})"
    return False, ""


def is_poisoned_objective(value: float | None) -> bool:
    """True when a trial objective is the sentinel failure score."""
    if value is None:
        return True
    return float(value) >= _FAILED_OBJECTIVE


def successful_trials(study: "optuna.Study") -> List["optuna.trial.FrozenTrial"]:
    """COMPLETE trials with a real (non-sentinel) objective value."""
    from optuna.trial import TrialState

    return [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE
        and not is_poisoned_objective(t.value)
    ]


def best_successful_trial(study: "optuna.Study") -> "optuna.trial.FrozenTrial | None":
    complete = successful_trials(study)
    if not complete:
        return None
    return min(complete, key=lambda t: float(t.value))
