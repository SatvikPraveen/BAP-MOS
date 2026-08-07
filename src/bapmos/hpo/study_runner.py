"""CLI for BAP-MOS outer-loop Bayesian optimization (Optuna TPE)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bapmos.method.config_utils import dataset_filename
from bapmos.method.data_adapter import repo_root
from bapmos.hpo.objective import make_objective
from bapmos.hpo.paths import DEFAULT_HPO_SUITE, HPO_SUITES, HpoPaths
from bapmos.hpo.catalog_search import (
    GREEDY_DIM_ORDER,
    GREEDY_GRID_SIZE,
    baseline_flat_params,
    greedy_wave_candidates,
    heuristic_catalog,
)
from bapmos.hpo.search_space import (
    baseline_overrides_for_suite,
    flat_params_to_overrides,
    overrides_to_flat_params,
    search_profile_for_suite,
)
from bapmos.hpo.saturation_stop import (
    clear_saturation_flag,
    make_saturation_callback,
    wait_until_finalize_ready,
    worker_should_skip,
)
from bapmos.hpo.startup_catalog import (
    enqueue_startup_catalog,
    load_or_create_startup_catalog,
    n_startup_trials_for_suite,
    startup_completion_count,
    startup_missing_catalog_rows,
    startup_phase_complete,
    uses_prequeued_startup,
)
from bapmos.hpo.parallel_coord import (
    run_one_trial_ask_tell,
    wait_parallel_running_registration,
)
from bapmos.hpo.trial_utils import (
    best_successful_trial,
    search_params_key,
    successful_trials,
    trial_flat_params,
)

logger = logging.getLogger(__name__)

DATASETS = ("simulation", "case1", "case2")
PFUS1_DATASETS = ("pfus1_advanced",)
BLADDER_PFUS1_DATASETS = ("pfus1",)
PROSTATE_POOL_DATASETS = ("pooled",)
ALL_DATASETS = DATASETS + PFUS1_DATASETS + BLADDER_PFUS1_DATASETS + PROSTATE_POOL_DATASETS

PFUS1_HPO_SUITES = frozenset()  # reserved; PFUS1-advanced HPO suites not shipped

PROSTATE_HPO_SUITES = frozenset(
    {
        "bapmos_bo",
        "bapmos_bo_random",
        "bapmos_bo_heuristic",
        "bapmos_bo_greedy",
        "bapmos_bo_medsam_pooled",
    }
)
BLADDER_HPO_SUITES = frozenset({"bapmos_bo_sam", "bapmos_bo_medsam"})


def datasets_for_suite(hpo_suite: str) -> tuple[str, ...]:
    if hpo_suite in BLADDER_HPO_SUITES:
        return BLADDER_PFUS1_DATASETS
    if hpo_suite in PROSTATE_HPO_SUITES:
        return PROSTATE_POOL_DATASETS
    return DATASETS


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required for HPO. Install with: pip install -r requirements-hpo.txt"
        ) from exc
    return optuna


def _count_existing_trials(db_path: Path, study_name: str) -> int:
    """Cheap SQLite trial count so RandomSampler seed can advance across recreate."""
    if not db_path.is_file():
        return 0
    try:
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trials t "
                "JOIN studies s ON s.study_id = t.study_id "
                "WHERE s.study_name = ?",
                (study_name,),
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _sampler_for_suite(hpo_suite: str, *, n_past_trials: int = 0):
    """Optuna sampler for the given HPO suite (TPE / random / heuristic / greedy)."""
    optuna = _require_optuna()
    paths = HpoPaths.for_suite(hpo_suite)
    sampler_kind = paths.spec.get("sampler", "tpe")
    if sampler_kind == "tpe":
        # Pre-enqueued startup catalog covers space-filling; sampler should use
        # TPE immediately for any new ask() after the catalog is drained.
        if uses_prequeued_startup(hpo_suite):
            sampler_startup = 0
        else:
            sampler_startup = int(paths.spec.get("n_startup_trials", 20))
        sequential = os.environ.get("HPO_SEQUENTIAL", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        return optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=sampler_startup,
            constant_liar=not sequential,
            multivariate=True,
            consider_endpoints=True,
        )
    # RandomSampler(seed=42) is recreated on every create_study() under
    # HPO_SEQUENTIAL. Without an offset it always proposes the same first
    # sample, which then forever PRUNEs as a duplicate of trial #0.
    # Catalog suites (greedy/heuristic) also fall through here when the
    # WAITING queue is empty; offsetting still avoids an infinite prune loop.
    seed = (42 + max(0, int(n_past_trials))) % (2**31 - 1)
    return optuna.samplers.RandomSampler(seed=seed)


def create_study(dataset: str, *, hpo_suite: str = DEFAULT_HPO_SUITE, storage: Optional[str] = None):
    """Open or create an Optuna study (SQLite-safe under parallel Slurm workers)."""
    optuna = _require_optuna()
    paths = HpoPaths.for_suite(hpo_suite)
    db_path = paths.study_db_path(dataset)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = storage or f"sqlite:///{db_path.resolve()}"
    study_name = paths.study_name(dataset)
    n_past = _count_existing_trials(db_path, study_name)
    sampler = _sampler_for_suite(hpo_suite, n_past_trials=n_past)

    last_exc: Optional[BaseException] = None
    for attempt in range(15):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=url,
                load_if_exists=True,
                direction="minimize",
                sampler=sampler,
            )
            # create_study(..., load_if_exists=True) may keep a prior sampler
            # instance; force the freshly offset RandomSampler / TPE config.
            study.sampler = sampler
            return study, paths
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            retryable = (
                "already exists" in msg
                or "database is locked" in msg
                or "locked" in msg
            )
            if not retryable:
                raise
            try:
                study = optuna.load_study(
                    study_name=study_name, storage=url, sampler=sampler
                )
                study.sampler = sampler
                return study, paths
            except Exception:
                pass
            time.sleep(0.2 + random.uniform(0.0, 0.4) * (attempt + 1))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to open Optuna study {study_name!r}")


def cmd_init(args: argparse.Namespace) -> None:
    """Create study DB/schema once before parallel worker submission."""
    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    paths = HpoPaths.for_suite(args.hpo_suite)
    for ds in datasets:
        study, _ = create_study(ds, hpo_suite=args.hpo_suite, storage=args.storage)
        method = paths.spec.get("search_method")
        if method == "heuristic":
            raw_budget = os.environ.get("N_TRIALS", "").strip()
            max_trials = int(raw_budget) if raw_budget else None
            added = enqueue_heuristic_catalog(
                study,
                hpo_suite=args.hpo_suite,
                max_trials=max_trials,
            )
            print(
                f"Initialized study {study.study_name!r} -> {paths.study_db_path(ds)} "
                f"(heuristic catalog: {added} enqueued)"
            )
        elif method == "greedy":
            if not study.trials:
                enqueue_baseline(study, hpo_suite=args.hpo_suite)
            if not paths.greedy_state_path().is_file():
                save_greedy_state(
                    paths,
                    {
                        "best_params": baseline_flat_params(hpo_suite=args.hpo_suite),
                        "wave": -1,
                        "best_value": None,
                    },
                )
            print(
                f"Initialized study {study.study_name!r} -> {paths.study_db_path(ds)} "
                "(greedy: baseline enqueued)"
            )
        else:
            if uses_prequeued_startup(args.hpo_suite):
                n_startup = n_startup_trials_for_suite(args.hpo_suite)
                catalog, cat_path = load_or_create_startup_catalog(
                    paths,
                    ds,
                    hpo_suite=args.hpo_suite,
                    n_startup=n_startup,
                )
                added = enqueue_startup_catalog(
                    study,
                    catalog,
                    hpo_suite=args.hpo_suite,
                )
                print(
                    f"Initialized study {study.study_name!r} -> {paths.study_db_path(ds)} "
                    f"(TPE startup catalog: {len(catalog)} configs, {added} newly enqueued) "
                    f"-> {cat_path}"
                )
            else:
                print(f"Initialized study {study.study_name!r} -> {paths.study_db_path(ds)}")


def enqueue_baseline(study, *, hpo_suite: str = DEFAULT_HPO_SUITE) -> bool:
    """Enqueue searched_clip_scale_organ baseline once. Returns False if already queued."""
    flat = overrides_to_flat_params(hpo_suite, baseline_overrides_for_suite(hpo_suite))
    if _params_already_enqueued(study, flat):
        return False
    study.enqueue_trial(flat)
    return True


def _params_already_enqueued(study, params: Dict[str, Any]) -> bool:
    target = search_params_key(params)
    for trial in study.trials:
        flat = trial_flat_params(trial)
        if flat and search_params_key(flat) == target:
            return True
    return False


def _params_has_active_or_successful(study, params: Dict[str, Any]) -> bool:
    """True when params are waiting, running, or successfully completed."""
    from optuna.trial import TrialState

    target = search_params_key(params)
    for trial in study.trials:
        flat = trial_flat_params(trial)
        if not flat or search_params_key(flat) != target:
            continue
        if trial.state in (TrialState.WAITING, TrialState.RUNNING):
            return True
        if trial.state == TrialState.COMPLETE and trial.value is not None:
            return True
    return False


def enqueue_heuristic_catalog(
    study,
    *,
    hpo_suite: str = DEFAULT_HPO_SUITE,
    max_trials: int | None = None,
) -> int:
    added = 0
    for params in heuristic_catalog(hpo_suite=hpo_suite, max_trials=max_trials):
        if _params_already_enqueued(study, params):
            continue
        study.enqueue_trial(params)
        added += 1
    return added


def load_greedy_state(paths: HpoPaths) -> Dict[str, Any]:
    path = paths.greedy_state_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "best_params": baseline_flat_params(hpo_suite=paths.suite),
        "wave": -1,
        "best_value": None,
    }


def save_greedy_state(paths: HpoPaths, state: Dict[str, Any]) -> None:
    path = paths.greedy_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _trial_matches_greedy_wave(
    params: Dict[str, Any],
    current: Dict[str, Any],
    dim: str,
) -> bool:
    grid_values = {
        c[dim] for c in greedy_wave_candidates(current, GREEDY_DIM_ORDER.index(dim))
    }
    if params.get(dim) not in grid_values:
        return False
    for key, value in current.items():
        if key == dim:
            continue
        if params.get(key) != value:
            return False
    return True


def greedy_wave_successful_trials(
    study,
    *,
    current: Dict[str, Any],
    wave: int,
    n_probes: int = GREEDY_GRID_SIZE,
) -> list:
    """Successful trials belonging to one greedy wave (optionally partial)."""
    dim = GREEDY_DIM_ORDER[wave]
    matched = [
        t
        for t in successful_trials(study)
        if _trial_matches_greedy_wave(trial_flat_params(t), current, dim)
    ]
    if n_probes >= GREEDY_GRID_SIZE:
        return matched
    allowed = {
        c[dim]
        for c in greedy_wave_candidates(current, wave)[:n_probes]
    }
    return [t for t in matched if trial_flat_params(t)[dim] in allowed]


def finalize_greedy_baseline(study, paths: HpoPaths) -> Dict[str, Any]:
    """Record baseline trial objective before the first coordinate wave."""
    baseline = baseline_flat_params(hpo_suite=paths.suite)
    complete = [
        t
        for t in successful_trials(study)
        if search_params_key(trial_flat_params(t)) == search_params_key(baseline)
    ]
    if not complete:
        raise SystemExit(
            "No successful baseline trial to finalize "
            f"(expected params {baseline})"
        )
    best = min(complete, key=lambda t: float(t.value))
    new_state = {
        "best_params": dict(baseline),
        "wave": -1,
        "round": -1,
        "best_value": float(best.value),
        "best_trial": best.number,
        "dim": None,
    }
    save_greedy_state(paths, new_state)
    return new_state


def finalize_greedy_wave(
    study,
    paths: HpoPaths,
    wave: int,
    *,
    round_idx: int = 0,
    n_probes: int = GREEDY_GRID_SIZE,
) -> Dict[str, Any]:
    """Pick best completed trial from a greedy wave and advance greedy state."""
    if wave < 0 or wave >= len(GREEDY_DIM_ORDER):
        raise ValueError(f"greedy wave index out of range: {wave}")

    state = load_greedy_state(paths)
    current = dict(state["best_params"])
    dim = GREEDY_DIM_ORDER[wave]

    complete = greedy_wave_successful_trials(
        study, current=current, wave=wave, n_probes=n_probes
    )
    if not complete:
        raise SystemExit(
            f"No completed greedy wave {wave} ({dim}) trials to finalize "
            f"(round={round_idx}, n_probes={n_probes}, center={current})"
        )

    best = min(complete, key=lambda t: float(t.value))
    best_params = trial_flat_params(best)
    updated = dict(current)
    updated[dim] = best_params[dim]
    new_state = {
        "best_params": updated,
        "wave": wave,
        "round": round_idx,
        "best_value": float(best.value),
        "best_trial": best.number,
        "dim": dim,
    }
    save_greedy_state(paths, new_state)
    return new_state


def ensure_greedy_queue(
    study,
    paths: HpoPaths,
    *,
    budget: int | None = None,
) -> int:
    """
    Keep the greedy WAITING queue non-empty under sequential ``run``.

    Catalog greedy only trains enqueued trials. After baseline (and each wave)
    completes, finalize state and enqueue the next coordinate wave. Without
    this, ``study.ask()`` falls through to RandomSampler and infinite-duplicates.
    Returns number of newly enqueued trials.
    """
    from optuna.trial import TrialState

    from bapmos.hpo.catalog_search import GREEDY_DEFAULT_N_TRIALS

    target = (
        int(budget)
        if budget is not None
        else int(paths.spec.get("default_n_trials", GREEDY_DEFAULT_N_TRIALS))
    )
    waiting = [t for t in study.trials if t.state == TrialState.WAITING]
    if waiting:
        return 0
    n_succ = len(successful_trials(study))
    if n_succ >= target:
        return 0

    state = load_greedy_state(paths)
    baseline = baseline_flat_params(hpo_suite=paths.suite)

    # Finalize baseline once its COMPLETE trial exists.
    if state.get("best_value") is None:
        baseline_done = [
            t
            for t in successful_trials(study)
            if search_params_key(trial_flat_params(t)) == search_params_key(baseline)
        ]
        if baseline_done:
            state = finalize_greedy_baseline(study, paths)
            print(
                f"Greedy auto: finalized baseline "
                f"(trial #{state.get('best_trial')}, "
                f"{paths.objective_label}={state.get('best_value'):.4f})"
            )
        elif not _params_has_active_or_successful(study, baseline):
            study.enqueue_trial(dict(baseline))
            print("Greedy auto: enqueued missing baseline trial")
            return 1
        else:
            return 0  # baseline still RUNNING

    remaining = target - len(successful_trials(study))
    if remaining <= 0:
        return 0

    # Advance past any wave whose probes are already all COMPLETE (e.g. resume).
    safety = 0
    while safety < 64:
        safety += 1
        state = load_greedy_state(paths)
        last_wave = int(state.get("wave", -1))
        last_round = int(state.get("round", -1))
        if last_wave < 0:
            next_wave, next_round = 0, 0
        else:
            next_wave = last_wave + 1
            next_round = last_round
            if next_wave >= len(GREEDY_DIM_ORDER):
                next_wave = 0
                next_round = last_round + 1

        n_probes = min(GREEDY_GRID_SIZE, remaining)
        current = dict(state["best_params"])
        candidates = greedy_wave_candidates(current, next_wave)[:n_probes]
        if not candidates:
            return 0

        complete = greedy_wave_successful_trials(
            study, current=current, wave=next_wave, n_probes=n_probes
        )
        if len(complete) >= len(candidates):
            state = finalize_greedy_wave(
                study,
                paths,
                next_wave,
                round_idx=next_round,
                n_probes=n_probes,
            )
            print(
                f"Greedy auto: finalized wave {next_wave} "
                f"({GREEDY_DIM_ORDER[next_wave]}) round={next_round} | "
                f"best trial #{state.get('best_trial')} "
                f"{paths.objective_label}={state.get('best_value'):.4f}"
            )
            remaining = target - len(successful_trials(study))
            if remaining <= 0:
                return 0
            continue

        # Some probes still RUNNING/WAITING — wait; do not enqueue duplicates.
        for params in candidates:
            flat_key = search_params_key(params)
            for t in study.trials:
                if t.state not in (TrialState.WAITING, TrialState.RUNNING):
                    continue
                flat = trial_flat_params(t)
                if flat and search_params_key(flat) == flat_key:
                    return 0

        added = 0
        for params in candidates:
            if _params_has_active_or_successful(study, params):
                continue
            study.enqueue_trial(dict(params))
            added += 1
        if added:
            print(
                f"Greedy auto: enqueued wave {next_wave} "
                f"({GREEDY_DIM_ORDER[next_wave]}) round={next_round} "
                f"— {added} new trial(s) (n_probes={n_probes})"
            )
        return added

    return 0


def _assert_prostate_suite_env(hpo_suite: str) -> None:
    """Refuse to mix Optuna CLI suite with a conflicting Slurm env."""
    if hpo_suite not in PROSTATE_HPO_SUITES:
        return
    env_suite = os.environ.get("PI_BAPMOS_HPO_SUITE")
    if env_suite and env_suite != hpo_suite:
        raise SystemExit(
            f"PI_BAPMOS_HPO_SUITE={env_suite!r} conflicts with --hpo-suite {hpo_suite!r}"
        )


def _reload_study(args: argparse.Namespace):
    study, _ = create_study(
        args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
    )
    return study


def cmd_run(args: argparse.Namespace) -> None:
    _assert_prostate_suite_env(args.hpo_suite)
    study, paths = create_study(args.dataset, hpo_suite=args.hpo_suite, storage=args.storage)
    skip, skip_reason = worker_should_skip(
        study, hpo_suite=args.hpo_suite, dataset=args.dataset
    )
    if skip:
        print(
            f"{args.dataset}: skipping worker — {skip_reason} "
            f"(no new trial scheduled)"
        )
        _print_best(study, args.dataset, paths)
        return

    if args.enqueue_baseline:
        if uses_prequeued_startup(args.hpo_suite):
            print(
                f"{args.dataset}: --enqueue-baseline ignored "
                "(baseline is in the pre-enqueued TPE startup catalog; run init)"
            )
        elif enqueue_baseline(study, hpo_suite=args.hpo_suite):
            print(f"{args.dataset}: enqueued baseline trial")
        else:
            print(
                f"{args.dataset}: baseline params already in study — "
                "skipping enqueue (no duplicate fixed trial)"
            )
    objective = make_objective(
        args.dataset,
        hpo_suite=args.hpo_suite,
        experiment_name=args.experiment,
        no_wandb=args.no_wandb,
    )
    callbacks = [
        make_saturation_callback(
            hpo_suite=args.hpo_suite,
            dataset=args.dataset,
            reload_study=lambda: _reload_study(args),
        )
    ]
    target_trials = max(1, args.n_trials)
    sequential = os.environ.get("HPO_SEQUENTIAL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    max_attempts = int(os.environ.get("HPO_MAX_DUPLICATE_RETRIES", "15"))
    if sequential:
        # TPE may need many duplicate prunes before a novel config; do not exit early.
        max_attempts = int(
            os.environ.get(
                "HPO_MAX_DUPLICATE_RETRIES",
                os.environ.get("HPO_SEQUENTIAL_DUPLICATE_RETRIES", "200"),
            )
        )

    completed_this_worker = 0
    reload = lambda: _reload_study(args)

    startup_missing_keys_fn = None
    if uses_prequeued_startup(args.hpo_suite):
        catalog, _ = load_or_create_startup_catalog(
            paths,
            args.dataset,
            hpo_suite=args.hpo_suite,
            n_startup=n_startup_trials_for_suite(args.hpo_suite),
        )
        n_startup = n_startup_trials_for_suite(args.hpo_suite)
        startup_guard_enabled = os.environ.get(
            "HPO_STARTUP_GUARD", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        if startup_guard_enabled and not startup_phase_complete(
            study, catalog, n_startup=n_startup
        ):
            def startup_missing_keys_fn(study):
                if startup_phase_complete(study, catalog, n_startup=n_startup):
                    return None
                missing = startup_missing_catalog_rows(study, catalog)
                if not missing:
                    return None
                return {search_params_key(row) for row in missing}

            print(
                f"{args.dataset}: startup guard active "
                f"({startup_completion_count(study, catalog)}/{n_startup} catalog configs)"
            )
        else:
            print(f"{args.dataset}: startup catalog complete — TPE sampling enabled")

    if sequential and startup_missing_keys_fn is not None:
        print(
            f"{args.dataset}: WARN sequential TPE with incomplete startup catalog "
            "— startup guard still active"
        )

    while completed_this_worker < target_trials:
        study, paths = create_study(
            args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
        )
        if paths.spec.get("search_method") == "greedy":
            ensure_greedy_queue(study, paths)
            study, paths = create_study(
                args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
            )
        skip, skip_reason = worker_should_skip(
            study, hpo_suite=args.hpo_suite, dataset=args.dataset
        )
        if skip:
            print(
                f"{args.dataset}: skipping worker — {skip_reason} "
                f"(no new trial scheduled)"
            )
            _print_best(study, args.dataset, paths)
            return

        if os.environ.get("HPO_SEQUENTIAL", "0").strip().lower() not in (
            "1",
            "true",
            "yes",
        ):
            wait_parallel_running_registration(reload)
        before = len(successful_trials(study))
        ran = run_one_trial_ask_tell(
            reload,
            objective,
            callbacks,
            max_attempts=max_attempts,
            startup_missing_keys_fn=startup_missing_keys_fn,
        )
        if ran:
            study, paths = create_study(
                args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
            )
            after = len(successful_trials(study))
            if after > before:
                completed_this_worker += 1
            elif sequential:
                print(
                    f"{args.dataset}: trial finished but successful count "
                    f"unchanged ({before}); continuing sequential loop"
                )
            else:
                print(
                    f"{args.dataset}: trial finished but no new successful "
                    f"trial ({before} -> {after})"
                )
            continue
        study, paths = create_study(
            args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
        )
        skip, skip_reason = worker_should_skip(
            study, hpo_suite=args.hpo_suite, dataset=args.dataset
        )
        if skip:
            print(
                f"{args.dataset}: stopping after no distinct trial — {skip_reason}"
            )
            _print_best(study, args.dataset, paths)
            return
        if sequential:
            print(
                f"{args.dataset}: no distinct trial after {max_attempts} ask "
                f"attempts; retrying (sequential mode)"
            )
            continue
        print(
            f"{args.dataset}: gave up after {max_attempts} attempts "
            f"without a new COMPLETE trial"
        )
        break

    _purge_after_run(args, paths)
    study, _ = create_study(args.dataset, hpo_suite=args.hpo_suite, storage=args.storage)
    _print_best(study, args.dataset, paths)


def _purge_after_run(args: argparse.Namespace, paths: HpoPaths) -> None:
    from bapmos.hpo.study_purge import (
        purge_duplicates_enabled,
        purge_noncontributing_enabled,
        purge_study_artifacts,
    )

    if not purge_noncontributing_enabled() and not purge_duplicates_enabled():
        return
    removed_states, removed_duplicates = purge_study_artifacts(
        paths,
        args.dataset,
        purge_duplicates=purge_duplicates_enabled(),
    )
    if removed_states:
        print(
            f"{args.dataset}: purged {len(removed_states)} non-contributing trial(s) "
            f"(FAIL/PRUNED): {removed_states}"
        )
    if removed_duplicates:
        print(
            f"{args.dataset}: purged {len(removed_duplicates)} duplicate COMPLETE "
            f"trial(s) (kept best per param set): {removed_duplicates}"
        )


def cmd_purge_trials(args: argparse.Namespace) -> None:
    from bapmos.hpo.study_purge import (
        DEFAULT_PURGE_STATES,
        purge_noncontributing_trials_for_suite,
    )

    states = tuple(args.states) if args.states else DEFAULT_PURGE_STATES
    purge_duplicates = not getattr(args, "no_duplicates", False)
    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    for ds in datasets:
        removed_states, removed_duplicates = purge_noncontributing_trials_for_suite(
            args.hpo_suite,
            ds,
            states=states,
            purge_duplicates=purge_duplicates,
        )
        if removed_states:
            print(f"{ds}: purged {len(removed_states)} trial(s) {list(states)} -> #{removed_states}")
        if removed_duplicates:
            print(
                f"{ds}: purged {len(removed_duplicates)} duplicate COMPLETE "
                f"trial(s) -> #{removed_duplicates}"
            )
        if not removed_states and not removed_duplicates:
            print(f"{ds}: nothing to purge")


def cmd_worker(args: argparse.Namespace) -> None:
    """Run one or more trials (intended for Slurm workers)."""
    cmd_run(args)


def cmd_summary(args: argparse.Namespace) -> None:
    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    paths = HpoPaths.for_suite(args.hpo_suite)
    for ds in datasets:
        study, _ = create_study(ds, hpo_suite=args.hpo_suite, storage=args.storage)
        _print_best(study, ds, paths)


def _fail_study_trial(study, trial_number: int, *, running_only: bool) -> str:
    """Mark an Optuna trial as FAIL (e.g. worker died without reporting)."""
    from optuna.trial import TrialState

    trial = next((t for t in study.trials if t.number == trial_number), None)
    if trial is None:
        raise SystemExit(f"Trial #{trial_number} not found in study {study.study_name!r}")

    state = trial.state
    if state == TrialState.FAIL:
        return f"Trial #{trial_number} already FAIL"
    if running_only and state != TrialState.RUNNING:
        raise SystemExit(
            f"Trial #{trial_number} is {state.name}, not RUNNING "
            "(drop --running-only to force FAIL)"
        )
    if state == TrialState.COMPLETE:
        raise SystemExit(
            f"Trial #{trial_number} is COMPLETE; refusing to mark FAIL "
            "(drop --running-only only applies to non-COMPLETE states)"
        )

    ok = study._storage.set_trial_state_values(
        trial._trial_id,
        TrialState.FAIL,
        values=None,
    )
    if not ok:
        raise SystemExit(f"Failed to update trial #{trial_number} to FAIL")
    return f"Trial #{trial_number}: {state.name} -> FAIL"


def _outer_loop_experiments_root(paths: HpoPaths) -> Path:
    """``experiments/<site>/bapmos`` for archive / figures / logs."""
    if paths.suite in BLADDER_HPO_SUITES:
        return repo_root() / "experiments" / "bladder" / "bapmos"
    return repo_root() / "experiments" / "prostate" / "bapmos"


def _outer_loop_archive_label(paths: HpoPaths) -> str:
    if paths.suite in BLADDER_HPO_SUITES:
        return "medsam" if "medsam" in paths.suite else "sam"
    # Prostate MedSAM TPE must not share the Meta SAM "tpe" archive label.
    if paths.suite == "bapmos_bo_medsam_pooled":
        return "medsam"
    return paths.spec.get("search_method", paths.suite)


def _outer_loop_archive_root(paths: HpoPaths, stamp: str) -> Path:
    label = _outer_loop_archive_label(paths)
    return (
        _outer_loop_experiments_root(paths)
        / "archive"
        / f"outer_loop_{label}"
        / stamp
    )


def _outer_loop_figures_root(paths: HpoPaths) -> Path:
    label = _outer_loop_archive_label(paths)
    return _outer_loop_experiments_root(paths) / "figures" / "outer_loop" / label


def _outer_loop_logs_root(paths: HpoPaths) -> Path:
    label = _outer_loop_archive_label(paths)
    return _outer_loop_experiments_root(paths) / "logs" / "outer_loop" / label


def _archive_prior_outer_loop_artifacts(
    paths: HpoPaths,
    dataset: str,
    *,
    stamp: str,
) -> None:
    """
    Move prior-cluster / prior-run artifacts aside so a fresh study cannot
    pick up old exports, convergence plots, or trial run dirs.
    """
    import shutil

    method = paths.spec.get("search_method", paths.suite)
    archive_root = _outer_loop_archive_root(paths, stamp)
    archive_root.mkdir(parents=True, exist_ok=True)
    moved_any = False

    trial_root_rel = paths.trial_run_root()
    if trial_root_rel:
        trial_root = repo_root() / trial_root_rel
        if trial_root.is_dir() and any(trial_root.iterdir()):
            dest = archive_root / "runs"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(trial_root), str(dest))
            trial_root.mkdir(parents=True, exist_ok=True)
            print(f"{dataset}: archived prior trial runs -> {dest}")
            moved_any = True

    fig_root = _outer_loop_figures_root(paths)
    if fig_root.is_dir() and any(fig_root.iterdir()):
        dest = archive_root / "figures"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(fig_root), str(dest))
        fig_root.mkdir(parents=True, exist_ok=True)
        print(f"{dataset}: archived prior figures -> {dest}")
        moved_any = True

    selected = paths.selected_params_path(dataset)
    if selected.is_file():
        dest = archive_root / selected.name
        shutil.copy2(selected, dest)
        baseline = baseline_flat_params(hpo_suite=paths.suite)
        placeholder = {
            "selection_meta": {
                "study_name": paths.study_name(dataset),
                "objective": paths.objective_label,
                "metric_note": "Placeholder — run outer-loop export after fresh HPO",
                "search_method": method,
                "hpo_suite": paths.suite,
                "generation": 0,
                "params": baseline,
            },
            **flat_params_to_overrides(paths.suite, baseline),
        }
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(
            "# Placeholder until outer-loop export overwrites this file.\n"
            + yaml.safe_dump(placeholder, sort_keys=False),
            encoding="utf-8",
        )
        print(f"{dataset}: reset selected export (archived prior -> {dest})")
        moved_any = True

    log_root = _outer_loop_logs_root(paths)
    if log_root.is_dir():
        stale_logs = [
            p
            for p in log_root.glob("trial_*")
            if p.is_file()
        ]
        stale_logs.extend(log_root.glob("finalize_*"))
        stale_logs.extend(log_root.glob("tpe_sequential_*"))
        if stale_logs:
            dest_dir = archive_root / "logs"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for path in stale_logs:
                shutil.move(str(path), str(dest_dir / path.name))
            print(f"{dataset}: archived {len(stale_logs)} prior log file(s) -> {dest_dir}")
            moved_any = True

    if not moved_any:
        print(f"{dataset}: no prior outer-loop artifacts to archive")


def cmd_reset_study(args: argparse.Namespace) -> None:
    """Delete poisoned or corrupted study state and re-initialize cleanly."""
    import shutil
    from datetime import datetime, timezone

    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    paths = HpoPaths.for_suite(args.hpo_suite)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for ds in datasets:
        _archive_prior_outer_loop_artifacts(paths, ds, stamp=stamp)

        db_path = paths.study_db_path(ds)
        if db_path.is_file():
            backup = db_path.with_name(f"{db_path.stem}.bak.{stamp}{db_path.suffix}")
            shutil.move(str(db_path), str(backup))
            print(f"{ds}: backed up study DB -> {backup}")

        if paths.spec.get("search_method") == "greedy":
            greedy_state = paths.greedy_state_path()
            if greedy_state.is_file():
                backup_state = greedy_state.with_name(
                    f"greedy_state.bak.{stamp}.json"
                )
                shutil.move(str(greedy_state), str(backup_state))
                print(f"{ds}: backed up greedy state -> {backup_state}")

        sat_flag = paths.study_db_path(ds).with_name(
            f"{paths.study_db_path(ds).stem}.saturated.json"
        )
        if sat_flag.is_file():
            backup_sat = sat_flag.with_name(f"{sat_flag.stem}.bak.{stamp}.json")
            shutil.move(str(sat_flag), str(backup_sat))
            print(f"{ds}: backed up saturation flag -> {backup_sat}")
        else:
            clear_saturation_flag(paths, ds)

        if uses_prequeued_startup(args.hpo_suite):
            from bapmos.hpo.startup_catalog import startup_catalog_path

            cat_path = startup_catalog_path(paths, ds)
            if cat_path.is_file():
                backup_cat = cat_path.with_name(
                    f"{cat_path.stem}.bak.{stamp}{cat_path.suffix}"
                )
                shutil.move(str(cat_path), str(backup_cat))
                print(f"{ds}: backed up startup catalog -> {backup_cat}")

        study, _ = create_study(ds, hpo_suite=args.hpo_suite, storage=args.storage)
        method = paths.spec.get("search_method")
        if method == "heuristic":
            raw_budget = os.environ.get("N_TRIALS", "").strip()
            max_trials = int(raw_budget) if raw_budget else None
            added = enqueue_heuristic_catalog(
                study,
                hpo_suite=args.hpo_suite,
                max_trials=max_trials,
            )
            print(
                f"{ds}: re-initialized study {study.study_name!r} "
                f"(heuristic catalog: {added} enqueued)"
            )
        elif method == "greedy":
            enqueue_baseline(study, hpo_suite=args.hpo_suite)
            save_greedy_state(
                paths,
                {
                    "best_params": baseline_flat_params(hpo_suite=args.hpo_suite),
                    "wave": -1,
                    "best_value": None,
                },
            )
            print(
                f"{ds}: re-initialized study {study.study_name!r} "
                "(greedy: baseline enqueued)"
            )
        elif uses_prequeued_startup(args.hpo_suite):
            n_startup = n_startup_trials_for_suite(args.hpo_suite)
            catalog, cat_path = load_or_create_startup_catalog(
                paths,
                ds,
                hpo_suite=args.hpo_suite,
                n_startup=n_startup,
            )
            added = enqueue_startup_catalog(
                study,
                catalog,
                hpo_suite=args.hpo_suite,
            )
            print(
                f"{ds}: re-initialized study {study.study_name!r} "
                f"(TPE startup catalog: {len(catalog)} configs, {added} newly enqueued) "
                f"-> {cat_path}"
            )
        else:
            print(f"{ds}: re-initialized study {study.study_name!r}")


def cmd_fail_trial(args: argparse.Namespace) -> None:
    """Mark stuck RUNNING trials as FAIL so TPE can advance."""
    if not args.trial:
        raise SystemExit("At least one --trial is required")

    study, paths = create_study(
        args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
    )
    for trial_number in args.trial:
        msg = _fail_study_trial(
            study,
            trial_number,
            running_only=not getattr(args, "force", False),
        )
        print(f"{paths.study_name(args.dataset)}: {msg}")


def cmd_prune_startup_trials(args: argparse.Namespace) -> None:
    """Fail WAITING/RUNNING startup trials whose catalog config is already COMPLETE."""
    from optuna.trial import TrialState

    from bapmos.hpo.startup_catalog import (
        _catalog_keys_successfully_completed,
        load_or_create_startup_catalog,
        n_startup_trials_for_suite,
    )

    study, paths = create_study(
        args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
    )
    catalog, _ = load_or_create_startup_catalog(
        paths,
        args.dataset,
        hpo_suite=args.hpo_suite,
        n_startup=n_startup_trials_for_suite(args.hpo_suite),
    )
    done = _catalog_keys_successfully_completed(study, catalog)
    n_pruned = 0
    for trial in study.trials:
        if trial.state not in (TrialState.WAITING, TrialState.RUNNING):
            continue
        flat = trial_flat_params(trial)
        if not flat:
            continue
        from bapmos.hpo.trial_utils import search_params_key

        if search_params_key(flat) not in done:
            continue
        msg = _fail_study_trial(study, trial.number, running_only=False)
        print(f"{paths.study_name(args.dataset)}: {msg}")
        n_pruned += 1
    print(f"Pruned {n_pruned} redundant startup trial(s) (config already COMPLETE).")


def _outer_loop_search_distributions():
    """Optuna distributions for bapmos_bo 5-D outer-loop search vector."""
    from optuna.distributions import FloatDistribution, IntDistribution

    return {
        "clip_max_mm": IntDistribution(5, 30),
        "alpha": FloatDistribution(0.05, 0.15),
        "r_min": IntDistribution(2, 20),
        "window_size": IntDistribution(20, 100, step=5),
        "block_size_batches": IntDistribution(20, 100, step=5),
    }


def cmd_record_catalog_complete(args: argparse.Namespace) -> None:
    """
    Backfill a successful COMPLETE startup-catalog trial from a finished GPU run.

    Idempotent: skips when the catalog config already has a successful COMPLETE trial.
    """
    from optuna.trial import TrialState, create_trial

    from bapmos.hpo.startup_catalog import (
        _catalog_keys_successfully_completed,
        load_or_create_startup_catalog,
        n_startup_trials_for_suite,
        read_startup_catalog,
        search_params_key,
        startup_completion_count,
        startup_phase_complete,
    )
    from bapmos.hpo.trial_utils import search_params_key as params_key

    study, paths = create_study(
        args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
    )
    catalog, cat_path = load_or_create_startup_catalog(
        paths,
        args.dataset,
        hpo_suite=args.hpo_suite,
        n_startup=n_startup_trials_for_suite(args.hpo_suite),
    )
    if args.catalog_csv:
        catalog = read_startup_catalog(args.catalog_csv)

    row = None
    for candidate in catalog:
        if int(candidate["clip_max_mm"]) == int(args.clip):
            row = dict(candidate)
            break
    if row is None:
        raise SystemExit(
            f"clip={args.clip} not found in startup catalog ({cat_path})"
        )

    key = search_params_key(row)
    done = _catalog_keys_successfully_completed(study, catalog)
    if key in done:
        for trial in study.trials:
            flat = trial_flat_params(trial)
            if (
                flat
                and params_key(flat) == key
                and trial.state == TrialState.COMPLETE
                and trial.value is not None
            ):
                print(
                    f"{paths.study_name(args.dataset)}: clip={args.clip} already "
                    f"COMPLETE (trial #{trial.number}, PTV_MSD={trial.value:.4f})"
                )
                print(
                    f"Startup: {startup_completion_count(study, catalog)}/"
                    f"{len(catalog)}"
                )
                return

    value = float(args.value)
    trial = create_trial(
        params=dict(row),
        distributions=_outer_loop_search_distributions(),
        value=value,
        state=TrialState.COMPLETE,
    )
    study.add_trial(trial)
    study, _ = create_study(
        args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
    )
    complete = startup_completion_count(study, catalog)
    print(
        f"{paths.study_name(args.dataset)}: recorded clip={args.clip} COMPLETE "
        f"(PTV_MSD={value:.4f})"
    )
    print(
        f"Startup: {complete}/{len(catalog)} | "
        f"phase_done={startup_phase_complete(study, catalog)}"
    )


def _print_best(study, dataset: str, paths: HpoPaths) -> None:
    complete = successful_trials(study)
    if not complete:
        poisoned = sum(
            1
            for t in study.trials
            if t.state.name == "COMPLETE"
        )
        if poisoned:
            print(
                f"{dataset}: no valid completed trials "
                f"({poisoned} poisoned/failed COMPLETE trial(s) ignored)"
            )
        else:
            print(f"{dataset}: no completed trials")
        return
    best = best_successful_trial(study)
    assert best is not None
    best_params = trial_flat_params(best)
    print(
        f"{dataset} [{paths.hpo_version}]: best trial #{best.number} | "
        f"{paths.objective_label}={best.value:.4f}"
    )
    for key in sorted(best_params):
        print(f"  {key}: {best_params[key]}")


def cmd_export(args: argparse.Namespace) -> None:
    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    paths = HpoPaths.for_suite(args.hpo_suite)
    manifest: Dict[str, Any] = {
        "production_version": paths.production_version,
        "hpo_version": paths.hpo_version,
        "hpo_suite": paths.suite,
        "objective": paths.objective_label,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for ds in datasets:
        study, _ = create_study(ds, hpo_suite=args.hpo_suite, storage=args.storage)
        complete = successful_trials(study)
        if not complete:
            raise SystemExit(f"No valid completed trials to export for {ds}")

        best = best_successful_trial(study)
        assert best is not None
        best_params = trial_flat_params(best)
        overrides = flat_params_to_overrides(args.hpo_suite, best_params)
        profile = search_profile_for_suite(args.hpo_suite)
        if paths.suite in BLADDER_HPO_SUITES:
            objective_note = (
                "Bladder k-fold mean MSD (5-fold, PFUS1, pixel units). "
                f"Search: {profile.description}"
            )
        else:
            objective_note = (
                f"PTV k-fold mean MSD (5-fold, full-res). Search: {profile.description}"
            )
        db_path = paths.study_db_path(ds).resolve()
        try:
            study_db_meta = str(db_path.relative_to(repo_root().resolve()))
        except ValueError:
            study_db_meta = str(db_path)
        payload: Dict[str, Any] = {
            "selection_meta": {
                "study_name": paths.study_name(ds),
                "study_db": study_db_meta,
                "trial_number": best.number,
                "objective": paths.objective_label,
                "metric_note": objective_note,
                "value": float(best.value),
                "exported_at": manifest["exported_at"],
                "generation": args.generation,
                "search_method": paths.spec.get("search_method", paths.suite),
                "hpo_suite": paths.suite,
                "params": dict(best_params),
            },
        }
        payload.update(overrides)

        out_path = paths.selected_params_path(ds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {out_path}")

        manifest["datasets"][ds] = {
            "selected_path": str(out_path.relative_to(repo_root())),
            "trial_number": best.number,
            "best_val_msd_mm": float(best.value),
            "objective": paths.objective_label,
            "search_method": paths.spec.get("search_method", paths.suite),
            "params": dict(best_params),
        }

    manifest_path = paths.selected_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


def cmd_wait_saturated(args: argparse.Namespace) -> None:
    """Poll until saturation or max budget, then export selected params."""
    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    if len(datasets) != 1:
        raise SystemExit("wait-saturated supports one dataset at a time")

    ds = datasets[0]
    report, reason = wait_until_finalize_ready(
        lambda: _reload_study(args),
        hpo_suite=args.hpo_suite,
        dataset=ds,
        poll_seconds=float(args.poll_seconds),
        timeout_seconds=(
            float(args.timeout_seconds) if args.timeout_seconds is not None else None
        ),
    )
    print(
        f"{ds}: finalize ready ({reason}) | completed={report.n_completed} "
        f"saturated={report.saturated} T_last={report.t_last} N_eff={report.n_eff}"
    )
    if report.best_value is not None:
        print(f"{ds}: best={report.best_value:.4f} trial=#{report.best_trial}")

    if args.export:
        cmd_export(args)


def cmd_convergence_report(args: argparse.Namespace) -> None:
    from bapmos.hpo.convergence_report import (
        run_convergence_report,
        run_convergence_report_all,
    )

    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    kwargs: Dict[str, Any] = {}
    if args.confirmation_trials is not None:
        kwargs["confirmation_trials"] = args.confirmation_trials
    if args.eps_abs is not None:
        kwargs["eps_abs"] = args.eps_abs
    if args.eps_rel is not None:
        kwargs["eps_rel"] = args.eps_rel
    if args.n_startup is not None:
        kwargs["n_startup"] = args.n_startup
    if getattr(args, "max_concurrent_gpus", None) is not None:
        kwargs["max_concurrent_gpus"] = args.max_concurrent_gpus
    out_dir = Path(args.output_dir) if args.output_dir else None

    if len(datasets) == 1:
        outputs = run_convergence_report(
            hpo_suite=args.hpo_suite,
            dataset=datasets[0],
            output_dir=out_dir,
            **kwargs,
        )
        from bapmos.hpo.convergence_report import (
            _startup_param_keys_for_report,
            analyze_saturation,
            convergence_defaults_for_suite,
        )
        from bapmos.hpo.study_runner import create_study

        paths = HpoPaths.for_suite(args.hpo_suite)
        defaults = convergence_defaults_for_suite(args.hpo_suite)
        study, _ = create_study(datasets[0], hpo_suite=args.hpo_suite)
        k = kwargs.get("confirmation_trials", defaults.confirmation_trials)
        report = analyze_saturation(
            study.trials,
            dataset=datasets[0],
            hpo_suite=args.hpo_suite,
            objective_label=paths.objective_label,
            n_startup=kwargs.get("n_startup", defaults.n_startup),
            confirmation_trials=k,
            eps_abs=kwargs.get("eps_abs", defaults.eps_abs),
            eps_rel=kwargs.get("eps_rel", defaults.eps_rel),
            startup_param_keys=_startup_param_keys_for_report(
                paths, datasets[0], args.hpo_suite
            ),
        )
        if report.best_value is not None:
            print(
                f"{report.dataset}: best={report.best_value:.4f} "
                f"T_last={report.t_last} saturated={report.saturated} "
                f"GPU-h(N_eff)={report.gpu_h_to_n_eff:.1f} waste={report.gpu_h_waste:.1f}"
            )
        for label, path in outputs.items():
            print(f"{datasets[0]} {label}: {path}")
        return

    reports, outputs = run_convergence_report_all(
        hpo_suite=args.hpo_suite,
        datasets=datasets,
        output_dir=out_dir,
        **kwargs,
    )
    for report in reports:
        if report.best_value is not None:
            print(
                f"{report.dataset}: best={report.best_value:.4f} "
                f"T_last={report.t_last} saturated={report.saturated} "
                f"GPU-h(N_eff)={report.gpu_h_to_n_eff:.1f} waste={report.gpu_h_waste:.1f}"
            )
        else:
            print(f"{report.dataset}: no completed trials")
    for label, path in outputs.items():
        print(f"{label}: {path}")


def cmd_coverage_report(args: argparse.Namespace) -> None:
    from bapmos.hpo.coverage_report import run_coverage_report

    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    paths = HpoPaths.for_suite(args.hpo_suite)
    n_startup = int(paths.spec.get("n_startup_trials", 20))
    for ds in datasets:
        outputs = run_coverage_report(
            hpo_suite=args.hpo_suite,
            dataset=ds,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            n_startup=n_startup,
        )
        for label, path in outputs.items():
            print(f"{ds} {label}: {path}")


def cmd_plot_report(args: argparse.Namespace) -> None:
    """Convergence, parameter importance, slice, and related Optuna/matplotlib figures.

    Uses every successful COMPLETE trial in the study (startup catalog + post-startup
    through saturation). Saturation (K, ε) gates early stopping only — not plot data.
    """
    from bapmos.hpo.convergence_report import default_report_output_dir
    from bapmos.method.scripts.plot_hpo_study import plot_suite

    datasets = _resolve_datasets(args.dataset, args.hpo_suite)
    out_dir = Path(args.output_dir) if args.output_dir else default_report_output_dir(args.hpo_suite)
    written = plot_suite(hpo_suite=args.hpo_suite, datasets=datasets, out_dir=out_dir)
    if not written:
        print(
            f"{args.hpo_suite}: no figures written "
            f"(need >=2 successful trials for importance/slice plots)"
        )
    for path in written:
        print(f"{args.hpo_suite} plot: {path}")


def cmd_greedy_wave(args: argparse.Namespace) -> None:
    """Enqueue one greedy coordinate wave (up to seven trials) around current best."""
    study, paths = create_study(args.dataset, hpo_suite=args.hpo_suite, storage=args.storage)
    state = load_greedy_state(paths)
    current = dict(state["best_params"])
    wave = args.wave
    if wave < 0 or wave >= len(GREEDY_DIM_ORDER):
        raise SystemExit(f"--wave must be in [0, {len(GREEDY_DIM_ORDER) - 1}]")

    n_probes = int(getattr(args, "n_probes", GREEDY_GRID_SIZE))
    if n_probes < 1 or n_probes > GREEDY_GRID_SIZE:
        raise SystemExit(f"--n-probes must be in [1, {GREEDY_GRID_SIZE}]")

    candidates = greedy_wave_candidates(current, wave)[:n_probes]
    added = 0
    for params in candidates:
        if _params_has_active_or_successful(study, params):
            continue
        study.enqueue_trial(params)
        added += 1
    dim = GREEDY_DIM_ORDER[wave]
    print(
        f"Greedy wave {wave} ({dim}): enqueued {added} new trial(s) "
        f"(n_probes={n_probes}) around {current}"
    )


def cmd_greedy_finalize(args: argparse.Namespace) -> None:
    """After a greedy wave completes, update greedy_state.json from best completed trial."""
    study, paths = create_study(args.dataset, hpo_suite=args.hpo_suite, storage=args.storage)
    state = finalize_greedy_wave(
        study,
        paths,
        args.wave,
        round_idx=int(getattr(args, "round", 0)),
        n_probes=int(getattr(args, "n_probes", GREEDY_GRID_SIZE)),
    )
    print(
        f"Greedy wave {args.wave} finalized | best trial #{state.get('best_trial')} | "
        f"{paths.objective_label}={state.get('best_value'):.4f}"
    )
    for key in sorted(state["best_params"]):
        print(f"  {key}: {state['best_params'][key]}")


def cmd_greedy_baseline_finalize(args: argparse.Namespace) -> None:
    """Record baseline trial objective in greedy_state.json."""
    study, paths = create_study(args.dataset, hpo_suite=args.hpo_suite, storage=args.storage)
    state = finalize_greedy_baseline(study, paths)
    print(
        f"Greedy baseline finalized | trial #{state.get('best_trial')} | "
        f"{paths.objective_label}={state.get('best_value'):.4f}"
    )


def cmd_wait_parallel_running(args: argparse.Namespace) -> None:
    """Block until sibling parallel workers register RUNNING trials in Optuna."""
    wait_parallel_running_registration(
        lambda: _reload_study(args),
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )


def cmd_wait_startup(args: argparse.Namespace) -> None:
    """Block until all pre-enqueued TPE startup configs have successful COMPLETE trials."""
    paths = HpoPaths.for_suite(args.hpo_suite)
    if not uses_prequeued_startup(args.hpo_suite):
        raise SystemExit(
            f"HPO suite {args.hpo_suite!r} does not use a pre-enqueued startup catalog"
        )
    n_startup = n_startup_trials_for_suite(args.hpo_suite)
    poll_s = float(args.poll_seconds)
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None

    while True:
        study, _ = create_study(
            args.dataset, hpo_suite=args.hpo_suite, storage=args.storage
        )
        catalog, cat_path = load_or_create_startup_catalog(
            paths,
            args.dataset,
            hpo_suite=args.hpo_suite,
            n_startup=n_startup,
        )
        done = startup_completion_count(study, catalog)
        if startup_phase_complete(study, catalog, n_startup=n_startup):
            print(
                f"{args.dataset}: startup phase complete "
                f"({done}/{n_startup} catalog configs trained) | catalog={cat_path}"
            )
            _print_best(study, args.dataset, paths)
            return
        print(
            f"{args.dataset}: startup progress {done}/{n_startup} "
            f"(waiting for catalog configs to finish) | catalog={cat_path}"
        )
        if deadline is not None and time.monotonic() >= deadline:
            raise SystemExit(
                f"Timed out after {args.timeout_seconds}s waiting for startup completion"
            )
        time.sleep(poll_s)


def _resolve_datasets(dataset_arg: str, hpo_suite: str) -> List[str]:
    pool = datasets_for_suite(hpo_suite)
    if dataset_arg == "all":
        return list(pool)
    key = dataset_filename(dataset_arg)
    if key not in pool:
        choices = ", ".join(pool)
        raise SystemExit(
            f"Dataset {dataset_arg!r} is not valid for HPO suite {hpo_suite!r}; "
            f"choose from: {choices}, or all"
        )
    return [key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BAP-MOS outer-loop Bayesian optimization (Optuna TPE per dataset)",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--hpo-suite",
        default=DEFAULT_HPO_SUITE,
        choices=sorted(HPO_SUITES.keys()),
        help="BO config suite (default: bapmos_bo)",
    )
    common.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite under optuna_studies/<hpo_version>/)",
    )
    common.add_argument(
        "--experiment",
        default="bo_trial_seed42",
        help="BO experiment name from version.yaml",
    )
    common.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB for BO trials",
    )
    common.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Optuna progress bar",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        parents=[common],
        help="Run N Optuna trials sequentially on this machine",
    )
    run_p.add_argument("--dataset", required=True, choices=[*ALL_DATASETS, "all"])
    run_p.add_argument("--n-trials", type=int, default=1)
    run_p.add_argument(
        "--enqueue-baseline",
        action="store_true",
        help="Enqueue searched_clip_scale_organ baseline as the first trial",
    )
    run_p.set_defaults(func=cmd_run)

    init_p = sub.add_parser(
        "init",
        parents=[common],
        help="Create Optuna study DB once (run before parallel sbatch workers)",
    )
    init_p.add_argument("--dataset", required=True, choices=[*ALL_DATASETS, "all"])
    init_p.set_defaults(func=cmd_init)

    wait_startup_p = sub.add_parser(
        "wait-startup",
        parents=[common],
        help="Block until all pre-enqueued TPE startup configs finish training",
    )
    wait_startup_p.add_argument("--dataset", required=True, choices=[*ALL_DATASETS, "all"])
    wait_startup_p.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Status poll interval (default: 60)",
    )
    wait_startup_p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock timeout (default: wait indefinitely)",
    )
    wait_startup_p.set_defaults(func=cmd_wait_startup)

    wait_parallel_p = sub.add_parser(
        "wait-parallel-running",
        parents=[common],
        help="Wait for sibling GPU workers to register RUNNING trials (parallel coord)",
    )
    wait_parallel_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    wait_parallel_p.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Poll interval (default: HPO_PARALLEL_OCCUPIED_POLL_SEC or 120)",
    )
    wait_parallel_p.add_argument(
        "--max-wait-seconds",
        type=float,
        default=None,
        help="Max wait (default: HPO_PARALLEL_OCCUPIED_WAIT_SEC or 300)",
    )
    wait_parallel_p.set_defaults(func=cmd_wait_parallel_running)

    worker_p = sub.add_parser(
        "worker",
        parents=[common],
        help="Run trials (Slurm worker; same as run, default n_trials=1)",
    )
    worker_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    worker_p.add_argument("--n-trials", type=int, default=1)
    worker_p.add_argument("--enqueue-baseline", action="store_true")
    worker_p.set_defaults(func=cmd_worker)

    sum_p = sub.add_parser("summary", parents=[common], help="Print best trial per dataset")
    sum_p.add_argument("--dataset", default="all", choices=[*ALL_DATASETS, "all"])
    sum_p.set_defaults(func=cmd_summary)

    exp_p = sub.add_parser(
        "export",
        parents=[common],
        help="Export best trial params to experiments/.../inner_loop/selected/<dataset>.yaml",
    )
    exp_p.add_argument("--dataset", default="all", choices=[*ALL_DATASETS, "all"])
    exp_p.add_argument(
        "--generation",
        type=int,
        default=1,
        help="Selection generation counter stored in selection_meta",
    )
    exp_p.set_defaults(func=cmd_export)

    fail_p = sub.add_parser(
        "fail-trial",
        parents=[common],
        help="Mark trial(s) as FAIL (e.g. Slurm worker died without reporting)",
    )
    fail_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    fail_p.add_argument(
        "--trial",
        type=int,
        action="append",
        metavar="N",
        help="Trial number to fail (repeat for multiple trials)",
    )
    fail_p.add_argument(
        "--force",
        action="store_true",
        help="Fail even if trial is not RUNNING (never overwrites COMPLETE)",
    )
    fail_p.set_defaults(func=cmd_fail_trial)

    prune_p = sub.add_parser(
        "prune-startup",
        parents=[common],
        help="Fail redundant WAITING/RUNNING startup trials (config already COMPLETE)",
    )
    prune_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    prune_p.set_defaults(func=cmd_prune_startup_trials)

    record_p = sub.add_parser(
        "record-catalog-complete",
        parents=[common],
        help="Backfill COMPLETE for a startup catalog config (finished GPU run)",
    )
    record_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    record_p.add_argument(
        "--clip",
        type=int,
        required=True,
        help="clip_max_mm of the startup catalog row to record",
    )
    record_p.add_argument(
        "--value",
        type=float,
        required=True,
        help="PTV k-fold MSD objective (mm) from the finished run log",
    )
    record_p.add_argument(
        "--catalog-csv",
        default=None,
        help="Optional startup catalog CSV (default: study catalog path)",
    )
    record_p.set_defaults(func=cmd_record_catalog_complete)

    reset_p = sub.add_parser(
        "reset-study",
        parents=[common],
        help="Backup and recreate Optuna study DB (clears poisoned/failed trials)",
    )
    reset_p.add_argument("--dataset", required=True, choices=[*ALL_DATASETS, "all"])
    reset_p.set_defaults(func=cmd_reset_study)

    purge_p = sub.add_parser(
        "purge-trials",
        parents=[common],
        help="Delete FAIL/PRUNED trials from Optuna DB (default: FAIL + PRUNED)",
    )
    purge_p.add_argument("--dataset", required=True, choices=[*ALL_DATASETS, "all"])
    purge_p.add_argument(
        "--states",
        nargs="+",
        choices=["FAIL", "PRUNED"],
        default=None,
        help="Trial states to remove (default: FAIL PRUNED)",
    )
    purge_p.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip removing duplicate COMPLETE trials (same params; keep best)",
    )
    purge_p.set_defaults(func=cmd_purge_trials)

    cov_p = sub.add_parser(
        "coverage-report",
        parents=[common],
        help="Export startup coverage grid PNG + trial CSV",
    )
    cov_p.add_argument("--dataset", default="pooled", choices=ALL_DATASETS)
    cov_p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: experiments/prostate/bapmos/figures/outer_loop/<method>)",
    )
    cov_p.set_defaults(func=cmd_coverage_report)

    plot_p = sub.add_parser(
        "plot-report",
        parents=[common],
        help=(
            "Export convergence, parameter importance, slice, and related "
            "Optuna/matplotlib figures"
        ),
    )
    plot_p.add_argument("--dataset", default="all", choices=[*ALL_DATASETS, "all"])
    plot_p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: figures path for suite/method)",
    )
    plot_p.set_defaults(func=cmd_plot_report)

    conv_p = sub.add_parser(
        "convergence-report",
        parents=[common],
        help="Export saturation + GPU-hour cost tables and figures",
    )
    conv_p.add_argument("--dataset", default="all", choices=[*ALL_DATASETS, "all"])
    conv_p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: docs/figures/hpo_convergence/<suite> or suite figures path)",
    )
    conv_p.add_argument(
        "--confirmation-trials",
        type=int,
        default=None,
        help="K: trials without epsilon-improvement required for saturation (default: suite spec)",
    )
    conv_p.add_argument(
        "--n-startup",
        type=int,
        default=None,
        help="Startup trial count N0 (default: suite spec)",
    )
    conv_p.add_argument(
        "--eps-abs",
        type=float,
        default=None,
        help="Absolute epsilon in objective units (default: suite eps_abs / eps_abs_physical_mm)",
    )
    conv_p.add_argument(
        "--eps-rel",
        type=float,
        default=None,
        help="Relative epsilon as fraction of best-so-far (default: 0.005)",
    )
    conv_p.add_argument(
        "--max-concurrent-gpus",
        type=int,
        default=None,
        help="Assumed parallel GPU workers for estimated campaign wall-clock in time-cost summary",
    )
    conv_p.set_defaults(func=cmd_convergence_report)

    wait_p = sub.add_parser(
        "wait-saturated",
        parents=[common],
        help=(
            "Poll until saturation (K, ε) or max trial budget, optionally export "
            "best params for outer-loop"
        ),
    )
    wait_p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    wait_p.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Seconds between study reloads (default: 60)",
    )
    wait_p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Abort if finalize is not ready within this many seconds",
    )
    wait_p.add_argument(
        "--export",
        action="store_true",
        help="Export best trial to selected/<dataset>.yaml when finalize is ready",
    )
    wait_p.add_argument(
        "--generation",
        type=int,
        default=1,
        help="Selection generation counter passed to export (default: 1)",
    )
    wait_p.set_defaults(func=cmd_wait_saturated)

    gw_p = sub.add_parser(
        "greedy-wave",
        parents=[common],
        help="Enqueue greedy coordinate wave for bapmos_bo_greedy",
    )
    gw_p.add_argument("--dataset", required=True, choices=PROSTATE_POOL_DATASETS)
    gw_p.add_argument("--wave", type=int, required=True, help="Coordinate index 0..4")
    gw_p.add_argument(
        "--n-probes",
        type=int,
        default=GREEDY_GRID_SIZE,
        help=f"Probes to enqueue for this wave (1..{GREEDY_GRID_SIZE}, default: full grid)",
    )
    gw_p.set_defaults(func=cmd_greedy_wave)

    gf_p = sub.add_parser(
        "greedy-finalize",
        parents=[common],
        help="Finalize greedy wave: update greedy_state.json from best completed trial",
    )
    gf_p.add_argument("--dataset", required=True, choices=PROSTATE_POOL_DATASETS)
    gf_p.add_argument("--wave", type=int, required=True, help="Coordinate index 0..4")
    gf_p.add_argument(
        "--round",
        type=int,
        default=0,
        help="Coordinate sweep round (0-based, for logging/state)",
    )
    gf_p.add_argument(
        "--n-probes",
        type=int,
        default=GREEDY_GRID_SIZE,
        help=f"Probes expected for this wave (default: {GREEDY_GRID_SIZE})",
    )
    gf_p.set_defaults(func=cmd_greedy_finalize)

    gb_p = sub.add_parser(
        "greedy-baseline-finalize",
        parents=[common],
        help="Finalize greedy baseline trial in greedy_state.json",
    )
    gb_p.add_argument("--dataset", required=True, choices=PROSTATE_POOL_DATASETS)
    gb_p.set_defaults(func=cmd_greedy_baseline_finalize)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.dataset == "all":
        for ds in datasets_for_suite(args.hpo_suite):
            args.dataset = ds
            cmd_run(args)
        return
    if args.command == "init" and args.dataset == "all":
        for ds in datasets_for_suite(args.hpo_suite):
            args.dataset = ds
            cmd_init(args)
        return
    if args.command == "reset-study" and args.dataset == "all":
        for ds in datasets_for_suite(args.hpo_suite):
            args.dataset = ds
            cmd_reset_study(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
