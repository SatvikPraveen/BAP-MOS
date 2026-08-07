"""Stop parallel HPO workers once empirical saturation (K, ε) is reached."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from bapmos.hpo.catalog_search import expected_trial_count
from bapmos.hpo.convergence_report import (
    SaturationReport,
    _startup_param_keys_for_report,
    analyze_saturation,
    convergence_defaults_for_suite,
)
from bapmos.hpo.paths import HpoPaths

logger = logging.getLogger(__name__)

SATURATION_ENV = "HPO_STOP_ON_SATURATION"
FINALIZE_REASON_SATURATED = "saturated"
FINALIZE_REASON_MAX_BUDGET = "max_budget"
FINALIZE_REASON_FLAG = "flag"


def stop_on_saturation_enabled(*, hpo_suite: str) -> bool:
    """Whether workers should skip new trials after saturation."""
    raw = os.environ.get(SATURATION_ENV, "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    method = HpoPaths.for_suite(hpo_suite).spec.get("search_method", "")
    if method in ("heuristic", "greedy"):
        return False
    return True


def startup_saturation_gate_open(
    study,
    *,
    hpo_suite: str,
    dataset: str,
) -> bool:
    """
    True when K/ε saturation and convergence finalize may apply.

    For TPE suites with a pre-enqueued startup catalog, saturation is deferred
    until all ``n_startup`` catalog configs have a successful COMPLETE trial.
  """
    from bapmos.hpo.startup_catalog import (
        load_or_create_startup_catalog,
        n_startup_trials_for_suite,
        startup_phase_complete,
        uses_prequeued_startup,
    )

    if not uses_prequeued_startup(hpo_suite):
        return True
    paths = HpoPaths.for_suite(hpo_suite)
    n_startup = n_startup_trials_for_suite(hpo_suite)
    catalog, _ = load_or_create_startup_catalog(
        paths,
        dataset,
        hpo_suite=hpo_suite,
        n_startup=n_startup,
    )
    return startup_phase_complete(study, catalog, n_startup=n_startup)


def saturation_flag_path(paths: HpoPaths, dataset: str) -> Path:
    db = paths.study_db_path(dataset)
    return db.with_name(f"{db.stem}.saturated.json")


def load_saturation_flag(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def saturation_report_for_study(
    study,
    *,
    hpo_suite: str,
    dataset: str,
    confirmation_trials: Optional[int] = None,
    eps_abs: Optional[float] = None,
    eps_rel: Optional[float] = None,
    n_startup: Optional[int] = None,
) -> SaturationReport:
    paths = HpoPaths.for_suite(hpo_suite)
    defaults = convergence_defaults_for_suite(hpo_suite)
    return analyze_saturation(
        study.trials,
        dataset=dataset,
        hpo_suite=hpo_suite,
        objective_label=paths.objective_label,
        n_startup=n_startup if n_startup is not None else defaults.n_startup,
        confirmation_trials=(
            confirmation_trials
            if confirmation_trials is not None
            else defaults.confirmation_trials
        ),
        eps_abs=eps_abs if eps_abs is not None else defaults.eps_abs,
        eps_rel=eps_rel if eps_rel is not None else defaults.eps_rel,
        startup_param_keys=_startup_param_keys_for_report(paths, dataset, hpo_suite),
    )


def write_saturation_flag(
    path: Path,
    report: SaturationReport,
    *,
    reason: str,
) -> bool:
    """Write saturation marker once. Returns True if this call created the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **asdict(report),
    }
    try:
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return True
    except FileExistsError:
        return False


def mark_saturation_if_ready(
    study,
    *,
    hpo_suite: str,
    dataset: str,
    reason: str = FINALIZE_REASON_SATURATED,
) -> Tuple[bool, Optional[SaturationReport]]:
    """If the study is saturated, write the shared flag (first writer wins)."""
    paths = HpoPaths.for_suite(hpo_suite)
    flag_path = saturation_flag_path(paths, dataset)
    if flag_path.is_file():
        return False, None

    if not startup_saturation_gate_open(study, hpo_suite=hpo_suite, dataset=dataset):
        return False, saturation_report_for_study(
            study, hpo_suite=hpo_suite, dataset=dataset
        )

    report = saturation_report_for_study(study, hpo_suite=hpo_suite, dataset=dataset)
    if not report.saturated:
        return False, report

    created = write_saturation_flag(flag_path, report, reason=reason)
    if created:
        logger.info(
            "%s: saturation reached (T_last=%s N_eff=%s best=%.4f) -> %s",
            dataset,
            report.t_last,
            report.n_eff,
            report.best_value if report.best_value is not None else float("nan"),
            flag_path,
        )
    return created, report


def worker_should_skip(
    study,
    *,
    hpo_suite: str,
    dataset: str,
) -> Tuple[bool, str]:
    """Return (skip, reason) before a worker claims another trial."""
    from bapmos.hpo.trial_utils import successful_trials

    paths = HpoPaths.for_suite(hpo_suite)
    budget = _trial_budget(hpo_suite)
    if len(successful_trials(study)) >= budget:
        return True, f"max trial budget reached ({budget})"

    flag_path = saturation_flag_path(paths, dataset)
    if flag_path.is_file() and startup_saturation_gate_open(
        study, hpo_suite=hpo_suite, dataset=dataset
    ):
        meta = load_saturation_flag(flag_path) or {}
        return True, f"saturation flag present (reason={meta.get('reason', '?')})"

    if not stop_on_saturation_enabled(hpo_suite=hpo_suite):
        return False, ""

    if not startup_saturation_gate_open(study, hpo_suite=hpo_suite, dataset=dataset):
        return False, ""

    report = saturation_report_for_study(study, hpo_suite=hpo_suite, dataset=dataset)
    if report.saturated:
        mark_saturation_if_ready(
            study,
            hpo_suite=hpo_suite,
            dataset=dataset,
            reason=FINALIZE_REASON_SATURATED,
        )
        return True, (
            f"study saturated (T_last={report.t_last} N_eff={report.n_eff} "
            f"best={report.best_value:.4f})"
            if report.best_value is not None
            else "study saturated"
        )
    return False, ""


def _trial_budget(hpo_suite: str) -> int:
    defaults = convergence_defaults_for_suite(hpo_suite)
    if defaults.default_n_trials > 0:
        return defaults.default_n_trials
    return expected_trial_count(hpo_suite)


def finalize_ready(
    study,
    *,
    hpo_suite: str,
    dataset: str,
) -> Tuple[bool, SaturationReport, str]:
    """Whether outer-loop is ready to export (saturated, flag, or max budget)."""
    paths = HpoPaths.for_suite(hpo_suite)
    flag_path = saturation_flag_path(paths, dataset)
    if flag_path.is_file() and startup_saturation_gate_open(
        study, hpo_suite=hpo_suite, dataset=dataset
    ):
        meta = load_saturation_flag(flag_path) or {}
        report = saturation_report_for_study(study, hpo_suite=hpo_suite, dataset=dataset)
        return True, report, str(meta.get("reason", FINALIZE_REASON_FLAG))

    report = saturation_report_for_study(study, hpo_suite=hpo_suite, dataset=dataset)
    if (
        startup_saturation_gate_open(study, hpo_suite=hpo_suite, dataset=dataset)
        and report.saturated
    ):
        return True, report, FINALIZE_REASON_SATURATED

    if report.n_completed >= _trial_budget(hpo_suite):
        return True, report, FINALIZE_REASON_MAX_BUDGET

    return False, report, ""


def make_saturation_callback(
    *,
    hpo_suite: str,
    dataset: str,
    reload_study: Callable[[], Any],
) -> Callable:
    """Optuna callback: mark saturation after each completed trial."""

    def _callback(study, trial) -> None:  # noqa: ARG001
        if not stop_on_saturation_enabled(hpo_suite=hpo_suite):
            return
        fresh = reload_study()
        if not startup_saturation_gate_open(
            fresh, hpo_suite=hpo_suite, dataset=dataset
        ):
            return
        report = saturation_report_for_study(fresh, hpo_suite=hpo_suite, dataset=dataset)
        if report.saturated:
            mark_saturation_if_ready(
                fresh,
                hpo_suite=hpo_suite,
                dataset=dataset,
                reason=FINALIZE_REASON_SATURATED,
            )
            fresh.stop()

    return _callback


def wait_until_finalize_ready(
    reload_study: Callable[[], Any],
    *,
    hpo_suite: str,
    dataset: str,
    poll_seconds: float = 60.0,
    timeout_seconds: Optional[float] = None,
) -> Tuple[SaturationReport, str]:
    """Poll until export is allowed (saturation, flag, or max budget)."""
    t0 = time.monotonic()
    last_log = 0.0
    while True:
        study = reload_study()
        ready, report, reason = finalize_ready(
            study, hpo_suite=hpo_suite, dataset=dataset
        )
        if ready:
            paths = HpoPaths.for_suite(hpo_suite)
            if reason == FINALIZE_REASON_SATURATED and not saturation_flag_path(
                paths, dataset
            ).is_file():
                mark_saturation_if_ready(
                    study,
                    hpo_suite=hpo_suite,
                    dataset=dataset,
                    reason=reason,
                )
            return report, reason

        elapsed = time.monotonic() - t0
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s waiting for HPO finalize "
                f"({hpo_suite}/{dataset}; {report.n_completed} completed trials)"
            )

        if elapsed - last_log >= max(poll_seconds, 1.0):
            logger.info(
                "%s/%s: waiting for finalize (%d completed, saturated=%s)",
                hpo_suite,
                dataset,
                report.n_completed,
                report.saturated,
            )
            last_log = elapsed
        time.sleep(poll_seconds)


def clear_saturation_flag(paths: HpoPaths, dataset: str) -> None:
    flag = saturation_flag_path(paths, dataset)
    if flag.is_file():
        flag.unlink()
