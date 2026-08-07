"""Remove FAIL / PRUNED and duplicate COMPLETE trials from Optuna SQLite storage.

- FAIL / PRUNED: Optuna counts these toward ``TPESampler.n_startup_trials``.
- Duplicate COMPLETE: parallel workers can race and train the same 5-D params twice;
  keep the best objective and delete the rest.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bapmos.hpo.paths import HpoPaths

DEFAULT_PURGE_STATES: Tuple[str, ...] = ("FAIL", "PRUNED")
PURGE_ENV = "HPO_PURGE_NONCONTRIBUTING"
PURGE_DUPLICATES_ENV = "HPO_PURGE_DUPLICATE_COMPLETE"

_TRIAL_CHILD_TABLES: Tuple[str, ...] = (
    "trial_user_attributes",
    "trial_system_attributes",
    "trial_params",
    "trial_values",
    "trial_intermediate_values",
    "trial_heartbeats",
)


def purge_noncontributing_enabled() -> bool:
    raw = os.environ.get(PURGE_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def purge_duplicates_enabled() -> bool:
    raw = os.environ.get(PURGE_DUPLICATES_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def duplicate_complete_trial_numbers(study) -> List[int]:
    """
    Trial numbers to delete: for each 5-D param set with multiple COMPLETE runs,
    keep the lowest objective (tie-break: lowest trial #); remove the others.
    """
    from optuna.trial import TrialState

    from bapmos.hpo.trial_utils import (
        is_poisoned_objective,
        search_params_key,
        trial_flat_params,
    )

    groups: Dict[tuple, list] = {}
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            continue
        if is_poisoned_objective(trial.value):
            continue
        flat = trial_flat_params(trial)
        if not flat:
            continue
        key = search_params_key(flat)
        groups.setdefault(key, []).append(trial)

    remove: List[int] = []
    for trials in groups.values():
        if len(trials) < 2:
            continue
        keeper = min(trials, key=lambda t: (float(t.value), t.number))
        for trial in trials:
            if trial.number != keeper.number:
                remove.append(int(trial.number))
    return sorted(remove)


def _delete_trial_ids(conn: sqlite3.Connection, trial_ids: Sequence[int]) -> None:
    if not trial_ids:
        return
    id_ph = ",".join("?" for _ in trial_ids)
    for table in _TRIAL_CHILD_TABLES:
        conn.execute(
            f"DELETE FROM {table} WHERE trial_id IN ({id_ph})",
            list(trial_ids),
        )
    conn.execute(
        f"DELETE FROM trials WHERE trial_id IN ({id_ph})",
        list(trial_ids),
    )


def purge_trial_numbers_from_db(
    db_path: Path,
    study_name: str,
    trial_numbers: Sequence[int],
    *,
    timeout_s: float = 30.0,
) -> List[int]:
    """Delete trials by Optuna trial number. Returns numbers actually removed."""
    numbers = sorted({int(n) for n in trial_numbers})
    if not numbers:
        return []
    db_path = Path(db_path)
    if not db_path.is_file():
        return []

    num_ph = ",".join("?" for _ in numbers)
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                conn.execute("PRAGMA foreign_keys=ON")
                row = conn.execute(
                    "SELECT study_id FROM studies WHERE study_name = ?",
                    (study_name,),
                ).fetchone()
                if row is None:
                    return []
                study_id = int(row[0])
                trial_rows = conn.execute(
                    f"""
                    SELECT trial_id, number FROM trials
                    WHERE study_id = ? AND number IN ({num_ph})
                    """,
                    (study_id, *numbers),
                ).fetchall()
                if not trial_rows:
                    return []
                removed = sorted(int(r[1]) for r in trial_rows)
                trial_ids = [int(r[0]) for r in trial_rows]
                with conn:
                    _delete_trial_ids(conn, trial_ids)
                return removed
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_exc = exc
            time.sleep(0.25)
        except Exception:
            raise

    if last_exc is not None:
        raise last_exc
    return []


def purge_trials_from_db(
    db_path: Path,
    study_name: str,
    *,
    states: Sequence[str] = DEFAULT_PURGE_STATES,
    timeout_s: float = 30.0,
) -> List[int]:
    """Delete trials in *states*. Returns trial numbers removed."""
    if not states:
        return []
    db_path = Path(db_path)
    if not db_path.is_file():
        return []

    placeholders = ",".join("?" for _ in states)
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None

    while time.monotonic() < deadline:
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                conn.execute("PRAGMA foreign_keys=ON")
                row = conn.execute(
                    "SELECT study_id FROM studies WHERE study_name = ?",
                    (study_name,),
                ).fetchone()
                if row is None:
                    return []
                study_id = int(row[0])
                trial_rows = conn.execute(
                    f"""
                    SELECT trial_id, number FROM trials
                    WHERE study_id = ? AND state IN ({placeholders})
                    """,
                    (study_id, *states),
                ).fetchall()
                if not trial_rows:
                    return []
                removed = sorted(int(r[1]) for r in trial_rows)
                trial_ids = [int(r[0]) for r in trial_rows]
                with conn:
                    _delete_trial_ids(conn, trial_ids)
                return removed
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_exc = exc
            time.sleep(0.25)
        except Exception:
            raise

    if last_exc is not None:
        raise last_exc
    return []


def purge_noncontributing_trials(
    paths: HpoPaths,
    dataset: str,
    *,
    states: Sequence[str] = DEFAULT_PURGE_STATES,
) -> List[int]:
    """Purge FAIL/PRUNED trials for one dataset study."""
    return purge_trials_from_db(
        paths.study_db_path(dataset),
        paths.study_name(dataset),
        states=states,
    )


def purge_duplicate_complete_trials(paths: HpoPaths, dataset: str) -> List[int]:
    """Remove duplicate COMPLETE trials (same 5-D params); keep best objective."""
    import optuna

    db_path = paths.study_db_path(dataset)
    study_name = paths.study_name(dataset)
    if not db_path.is_file():
        return []
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.resolve()}",
    )
    to_remove = duplicate_complete_trial_numbers(study)
    if not to_remove:
        return []
    return purge_trial_numbers_from_db(db_path, study_name, to_remove)


def purge_study_artifacts(
    paths: HpoPaths,
    dataset: str,
    *,
    states: Sequence[str] = DEFAULT_PURGE_STATES,
    purge_duplicates: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Full cleanup: FAIL/PRUNED, then duplicate COMPLETE (best kept).

    Clears a stale saturation flag when the study is no longer saturated.

    Returns ``(removed_states, removed_duplicates)`` trial number lists.
    """
    removed_states = purge_noncontributing_trials(paths, dataset, states=states)
    removed_duplicates: List[int] = []
    if purge_duplicates:
        removed_duplicates = purge_duplicate_complete_trials(paths, dataset)
    _refresh_saturation_flag(paths, dataset)
    return removed_states, removed_duplicates


def _refresh_saturation_flag(paths: HpoPaths, dataset: str) -> None:
    """Remove saturation flag if the study no longer meets K/no-improvement criteria."""
    from bapmos.hpo.saturation_stop import (
        load_saturation_flag,
        saturation_flag_path,
        saturation_report_for_study,
    )

    flag_path = saturation_flag_path(paths, dataset)
    if not flag_path.is_file():
        return
    import optuna

    db_path = paths.study_db_path(dataset)
    if not db_path.is_file():
        flag_path.unlink(missing_ok=True)
        return
    study = optuna.load_study(
        study_name=paths.study_name(dataset),
        storage=f"sqlite:///{db_path.resolve()}",
    )
    report = saturation_report_for_study(
        study, hpo_suite=paths.suite, dataset=dataset
    )
    if not report.saturated:
        flag_path.unlink(missing_ok=True)
        return
    # Refresh payload when still saturated but metrics changed (e.g. after dedup).
    meta = load_saturation_flag(flag_path) or {}
    if meta.get("q") != report.q or meta.get("n_completed") != report.n_completed:
        from bapmos.hpo.saturation_stop import write_saturation_flag, FINALIZE_REASON_SATURATED
        from dataclasses import asdict
        import json
        from datetime import datetime, timezone

        payload = {
            "reason": FINALIZE_REASON_SATURATED,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **asdict(report),
        }
        flag_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def purge_noncontributing_trials_for_suite(
    hpo_suite: str,
    dataset: str,
    *,
    states: Sequence[str] = DEFAULT_PURGE_STATES,
    purge_duplicates: bool = True,
) -> Tuple[List[int], List[int]]:
    paths = HpoPaths.for_suite(hpo_suite)
    return purge_study_artifacts(
        paths,
        dataset,
        states=states,
        purge_duplicates=purge_duplicates,
    )
