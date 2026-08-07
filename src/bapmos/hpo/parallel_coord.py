"""Coordinate parallel TPE workers: wait for sibling RUNNING trials, reserve distinct params."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from typing import Callable, List

from bapmos.hpo.trial_utils import (
    occupied_search_param_keys,
    search_params_key,
    should_prune_duplicate_params,
    trial_flat_params,
)


def count_sibling_tpe_workers(*, exclude_job_id: str | None = None) -> int:
    """Count other outer-loop TPE Slurm jobs (RUNNING+PENDING) for this user.

    Matches job names set by workers, or ``HPO_SIBLING_JOB_NAME_RE`` if set.
    Default pattern: ``pi-bapmos-*-tpe-t<digits>`` (covers legacy ``s1`` names).
    """
    pattern = os.environ.get(
        "HPO_SIBLING_JOB_NAME_RE",
        r"^pi-bapmos-.+-tpe-t\d+$",
    )
    try:
        name_re = re.compile(pattern)
    except re.error:
        name_re = re.compile(r"^pi-bapmos-.+-tpe-t\d+$")
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, KeyError):
        return 0
    count = 0
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        jid, name = parts
        if exclude_job_id and jid == str(exclude_job_id):
            continue
        if name_re.match(name):
            count += 1
    return count


def wait_parallel_running_registration(
    reload_study: Callable,
    *,
    poll_seconds: float | None = None,
    max_wait_seconds: float | None = None,
) -> None:
    """
  Before suggesting params, wait until parallel sibling workers have registered
  RUNNING trials in Optuna (2–5 min poll by default).

  Skipped when HPO_WAIT_PARALLEL_RUNNING=0 or no sibling TPE workers are queued.
  """
    if os.environ.get("HPO_WAIT_PARALLEL_RUNNING", "1") != "1":
        return

    poll = float(poll_seconds or os.environ.get("HPO_PARALLEL_OCCUPIED_POLL_SEC", "120"))
    max_wait = float(
        max_wait_seconds or os.environ.get("HPO_PARALLEL_OCCUPIED_WAIT_SEC", "300")
    )
    exclude = os.environ.get("SLURM_JOB_ID")
    siblings = count_sibling_tpe_workers(exclude_job_id=exclude)
    if siblings <= 0:
        return

    from optuna.trial import TrialState

    deadline = time.monotonic() + max_wait
    print(
        f"parallel coord: waiting up to {max_wait:.0f}s for "
        f"{siblings} sibling worker(s) to register RUNNING trials "
        f"(poll={poll:.0f}s)"
    )
    while time.monotonic() < deadline:
        study = reload_study()
        running = [t for t in study.trials if t.state == TrialState.RUNNING]
        if len(running) >= siblings:
            nums = sorted(t.number for t in running)
            print(
                f"parallel coord: {len(running)} RUNNING trial(s) visible "
                f"(#{nums[0]}..#{nums[-1]}); proceeding"
            )
            return
        remaining = deadline - time.monotonic()
        print(
            f"parallel coord: {len(running)}/{siblings} RUNNING registered; "
            f"sleep {min(poll, remaining):.0f}s"
        )
        time.sleep(min(poll, max(0.0, remaining)))

    print(
        "parallel coord: WARN timed out waiting for sibling RUNNING trials; "
        "proceeding with occupied-param check"
    )


def _acquire_startup_trial(study, missing_keys: set[tuple]):
    """Pop WAITING queue trials until one matches a missing startup catalog key."""
    import optuna
    from optuna.trial import TrialState

    for _ in range(max(1, len(missing_keys)) + 8):
        trial_id = study._pop_waiting_trial_id()
        if trial_id is None:
            return None
        frozen = study._storage.get_trial(trial_id)
        flat = trial_flat_params(frozen)
        if not flat and frozen.params:
            flat = dict(frozen.params)
        key = search_params_key(flat) if flat else tuple()
        if key in missing_keys:
            print(
                f"startup queue: claimed trial #{frozen.number} "
                f"(clip={flat.get('clip_max_mm') if flat else '?'})"
            )
            return optuna.trial.Trial(study, trial_id)
        study.tell(frozen.number, state=TrialState.PRUNED)
        print(
            f"CPU prune: stale WAITING trial #{frozen.number} "
            f"(clip={flat.get('clip_max_mm') if flat else '?'})"
        )
    return None


def run_one_trial_ask_tell(
    reload_study: Callable,
    objective: Callable,
    callbacks: List[Callable],
    *,
    max_attempts: int | None = None,
    startup_missing_keys_fn: Callable | None = None,
) -> bool:
    """
  Suggest one trial via study.ask(), pruning duplicates on CPU before training.

  Returns True when a trial completes with a new COMPLETE result.
  """
    optuna = __import__("optuna")
    from optuna.trial import TrialState

    attempts = int(max_attempts or os.environ.get("HPO_MAX_DUPLICATE_RETRIES", "15"))
    for attempt in range(1, attempts + 1):
        study = reload_study()
        missing_keys = None
        if startup_missing_keys_fn is not None:
            missing_keys = startup_missing_keys_fn(study)

        trial = None
        if missing_keys:
            trial = _acquire_startup_trial(study, missing_keys)
        if trial is None:
            trial = study.ask()
        flat = trial_flat_params(trial)
        if not flat and trial.params:
            flat = dict(trial.params)
        key = search_params_key(flat) if flat else tuple()

        if startup_missing_keys_fn is not None:
            if missing_keys is None:
                missing_keys = startup_missing_keys_fn(study)
            if missing_keys is not None and key not in missing_keys:
                study.tell(trial.number, state=TrialState.PRUNED)
                print(
                    f"CPU prune: non-startup trial #{trial.number} during startup "
                    f"phase (clip={flat.get('clip_max_mm') if flat else '?'}) "
                    f"(attempt {attempt}/{attempts})"
                )
                continue

        occupied = occupied_search_param_keys(study, exclude_trial=trial.number)
        if key and key in occupied:
            study.tell(trial.number, state=TrialState.PRUNED)
            print(
                f"CPU prune: params already occupied by COMPLETE/RUNNING sibling "
                f"(attempt {attempt}/{attempts})"
            )
            continue

        prune, reason = should_prune_duplicate_params(study, flat, trial.number)
        if prune:
            study.tell(trial.number, state=TrialState.PRUNED)
            print(
                f"CPU prune: duplicate of {reason} "
                f"(attempt {attempt}/{attempts})"
            )
            continue

        def _mark_fail_on_signal(signum, _frame) -> None:
            try:
                study.tell(trial.number, state=TrialState.FAIL)
                print(
                    f"Trial #{trial.number}: marked FAIL on signal {signum} "
                    "(Slurm preemption / timeout)"
                )
            except Exception as exc:
                print(
                    f"Trial #{trial.number}: could not mark FAIL on "
                    f"signal {signum}: {exc}"
                )
            raise SystemExit(128 + signum)

        prev_term = signal.signal(signal.SIGTERM, _mark_fail_on_signal)
        try:
            value = objective(trial)
            study.tell(trial.number, value)
            fresh = reload_study()
            for callback in callbacks:
                callback(fresh, trial)
            return True
        except optuna.TrialPruned:
            study.tell(trial.number, state=TrialState.PRUNED)
            print(f"pruned during training (attempt {attempt}/{attempts})")
        except Exception as exc:
            # Mark FAIL so the 5-D vector is not stuck as RUNNING (blocks siblings).
            try:
                study.tell(trial.number, state=TrialState.FAIL)
            except Exception as tell_exc:
                print(
                    f"Trial #{trial.number}: training failed ({exc!r}) and "
                    f"could not mark FAIL: {tell_exc}"
                )
            else:
                print(
                    f"Trial #{trial.number}: training failed; marked FAIL "
                    f"({exc!r}) (attempt {attempt}/{attempts})"
                )
            raise
        finally:
            signal.signal(signal.SIGTERM, prev_term)

    return False
