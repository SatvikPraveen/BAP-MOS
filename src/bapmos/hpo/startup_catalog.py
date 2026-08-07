"""Deterministic TPE startup catalogs (baseline + space-filling configs)."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from bapmos.hpo.catalog_search import (
    OUTER_LOOP_TPE_N_STARTUP_TRIALS,
    baseline_flat_params,
)
from bapmos.hpo.paths import HpoPaths
from bapmos.hpo.trial_utils import search_params_key, trial_flat_params

STARTUP_CATALOG_SEED = 42
STARTUP_CATALOG_FILENAME_SUFFIX = "_startup_catalog.csv"

# Matches ``trial_overrides_searched_clip_scale_organ`` bounds.
_DIM_SPECS: tuple[tuple[str, str, float, float, int | None], ...] = (
    ("clip_max_mm", "int", 5.0, 30.0, 1),
    ("alpha", "float", 0.05, 0.15, None),
    ("r_min", "int", 2.0, 20.0, 1),
    ("window_size", "int", 20.0, 100.0, 5),
    ("block_size_batches", "int", 20.0, 100.0, 5),
)


def uses_prequeued_startup(hpo_suite: str) -> bool:
    """True when init should enqueue a fixed startup catalog (TPE suites)."""
    return HpoPaths.for_suite(hpo_suite).spec.get("sampler") == "tpe"


def n_startup_trials_for_suite(hpo_suite: str) -> int:
    spec = HpoPaths.for_suite(hpo_suite).spec
    return int(spec.get("n_startup_trials", OUTER_LOOP_TPE_N_STARTUP_TRIALS))


def startup_catalog_path(paths: HpoPaths, dataset: str) -> Path:
    db = paths.study_db_path(dataset)
    return db.with_name(f"{db.stem}{STARTUP_CATALOG_FILENAME_SUFFIX}")


def _normalize_catalog_row(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the 5-D search vector with Optuna-compatible types."""
    from bapmos.hpo.trial_utils import SEARCH_PARAM_KEYS

    row: Dict[str, Any] = {}
    for key in SEARCH_PARAM_KEYS:
        if key not in params:
            raise KeyError(f"Missing startup param {key!r}")
        if key == "alpha":
            row[key] = float(params[key])
        else:
            row[key] = int(float(params[key]))
    return row


def _snap_int(value: float, *, low: int, high: int, step: int) -> int:
    snapped = int(round(value / step) * step)
    return max(low, min(high, snapped))


def _decode_lhs_row(unit_row: Sequence[float]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for u, (key, kind, low, high, step) in zip(unit_row, _DIM_SPECS):
        value = low + float(u) * (high - low)
        if kind == "float":
            params[key] = float(value)
        elif step is None:
            params[key] = int(round(value))
        else:
            params[key] = _snap_int(value, low=int(low), high=int(high), step=step)
    return params


def _latin_hypercube_unit(n: int, d: int, rng: random.Random) -> List[List[float]]:
    if n < 1:
        return []
    samples = [[0.0] * d for _ in range(n)]
    for j in range(d):
        perm = list(range(n))
        rng.shuffle(perm)
        for i, bucket in enumerate(perm):
            samples[i][j] = (bucket + rng.random()) / n
    return samples


def generate_startup_catalog(
    *,
    hpo_suite: str = "bapmos_bo",
    n_startup: int | None = None,
    seed: int = STARTUP_CATALOG_SEED,
) -> List[Dict[str, Any]]:
    """
    Baseline anchor + ``n_startup - 1`` Latin-hypercube configs in the 5-D box.

    All vectors are unique and deterministic for a given ``seed``.
    """
    n = n_startup if n_startup is not None else n_startup_trials_for_suite(hpo_suite)
    if n < 1:
        raise ValueError(f"startup catalog size must be >= 1, got {n}")

    baseline = _normalize_catalog_row(baseline_flat_params(hpo_suite=hpo_suite))
    catalog: List[Dict[str, Any]] = [baseline]
    seen = {search_params_key(baseline)}
    rng = random.Random(seed)

    lhs_n = max(0, n - 1)
    attempts = 0
    max_attempts = lhs_n * 50 + 100
    while len(catalog) < n and attempts < max_attempts:
        attempts += 1
        need = n - len(catalog)
        unit = _latin_hypercube_unit(need, len(_DIM_SPECS), rng)
        for row in unit:
            params = _normalize_catalog_row(_decode_lhs_row(row))
            key = search_params_key(params)
            if key in seen:
                continue
            seen.add(key)
            catalog.append(params)
            if len(catalog) >= n:
                break

    if len(catalog) < n:
        raise RuntimeError(
            f"Could only generate {len(catalog)} unique startup configs (wanted {n})"
        )
    return catalog


def write_startup_catalog(catalog: Sequence[Dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key, *_ in _DIM_SPECS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in catalog:
            writer.writerow({k: row[k] for k in fieldnames})
    return path


def read_startup_catalog(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    catalog: List[Dict[str, Any]] = []
    for row in rows:
        catalog.append(
            _normalize_catalog_row(
                {
                    "clip_max_mm": row["clip_max_mm"],
                    "alpha": row["alpha"],
                    "r_min": row["r_min"],
                    "window_size": row["window_size"],
                    "block_size_batches": row["block_size_batches"],
                }
            )
        )
    return catalog


def load_or_create_startup_catalog(
    paths: HpoPaths,
    dataset: str,
    *,
    hpo_suite: str,
    n_startup: int | None = None,
    seed: int = STARTUP_CATALOG_SEED,
) -> tuple[List[Dict[str, Any]], Path]:
    cat_path = startup_catalog_path(paths, dataset)
    if cat_path.is_file():
        return read_startup_catalog(cat_path), cat_path
    catalog = generate_startup_catalog(
        hpo_suite=hpo_suite,
        n_startup=n_startup,
        seed=seed,
    )
    write_startup_catalog(catalog, cat_path)
    return catalog, cat_path


def catalog_param_keys(catalog: Sequence[Dict[str, Any]]) -> set[tuple]:
    return {search_params_key(params) for params in catalog}


def _catalog_keys_successfully_completed(study, catalog: Sequence[Dict[str, Any]]) -> set[tuple]:
    """Catalog param keys with at least one successful COMPLETE trial."""
    from bapmos.hpo.trial_utils import is_poisoned_objective, successful_trials

    needed = catalog_param_keys(catalog)
    seen: set[tuple] = set()
    for trial in successful_trials(study):
        if is_poisoned_objective(getattr(trial, "value", None)):
            continue
        flat = trial_flat_params(trial)
        if not flat:
            continue
        key = search_params_key(flat)
        if key in needed:
            seen.add(key)
    return seen


def _params_already_in_study(study, params: Dict[str, Any]) -> bool:
    from optuna.trial import TrialState
    from bapmos.hpo.trial_utils import is_poisoned_objective, successful_trials

    target = search_params_key(params)
    for trial in successful_trials(study):
        if is_poisoned_objective(getattr(trial, "value", None)):
            continue
        flat = trial_flat_params(trial)
        if flat and search_params_key(flat) == target:
            return True
    for trial in study.trials:
        if trial.state not in (TrialState.WAITING, TrialState.RUNNING):
            continue
        flat = trial_flat_params(trial)
        if flat and search_params_key(flat) == target:
            return True
    return False


def enqueue_startup_catalog(
    study,
    catalog: Sequence[Dict[str, Any]],
    *,
    hpo_suite: str,
) -> int:
    """Enqueue fixed startup configs once at init. Returns count newly enqueued."""
    added = 0
    for params in catalog:
        if _params_already_in_study(study, params):
            continue
        study.enqueue_trial(dict(params), skip_if_exists=True)
        added += 1
    return added


def startup_completion_count(study, catalog: Sequence[Dict[str, Any]]) -> int:
    """How many catalog configs have at least one successful COMPLETE trial."""
    return len(_catalog_keys_successfully_completed(study, catalog))


def startup_missing_catalog_rows(
    study,
    catalog: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Catalog rows with no successful COMPLETE trial yet."""
    done = _catalog_keys_successfully_completed(study, catalog)
    return [row for row in catalog if search_params_key(row) not in done]


def startup_workers_needed(
    study,
    catalog: Sequence[Dict[str, Any]],
    *,
    slurm_tpe_workers: int = 0,
) -> int:
    """
    How many Slurm workers to submit for startup phase.

    Counts catalog configs still missing a successful COMPLETE trial. Also
    submits workers to drain WAITING/RUNNING startup trials when no Slurm job
    is already assigned (WAITING alone does not execute).
    """
    from optuna.trial import TrialState

    missing_keys = {search_params_key(row) for row in startup_missing_catalog_rows(study, catalog)}
    if not missing_keys:
        return 0

    active_keys: set[tuple] = set()
    pending_trials = 0
    for trial in study.trials:
        if trial.state not in (TrialState.WAITING, TrialState.RUNNING):
            continue
        flat = trial_flat_params(trial)
        if not flat:
            continue
        key = search_params_key(flat)
        if key not in missing_keys:
            continue
        active_keys.add(key)
        pending_trials += 1

    uncovered = len(missing_keys - active_keys)
    need_for_queue = max(0, pending_trials - max(0, slurm_tpe_workers))
    return max(uncovered, need_for_queue)


def startup_phase_complete(
    study,
    catalog: Sequence[Dict[str, Any]],
    *,
    n_startup: int | None = None,
) -> bool:
    target = n_startup if n_startup is not None else len(catalog)
    return startup_completion_count(study, catalog) >= target


def startup_trial_numbers_for_study(
    study,
    catalog: Sequence[Dict[str, Any]],
) -> List[int]:
    """Optuna trial numbers whose params belong to the startup catalog."""
    keys = catalog_param_keys(catalog)
    numbers: List[int] = []
    for trial in study.trials:
        flat = trial_flat_params(trial)
        if flat and search_params_key(flat) in keys:
            numbers.append(int(trial.number))
    return sorted(numbers)
