"""Empirical HPO saturation and GPU-hour cost reports for outer-loop Optuna studies."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bapmos.method.data_adapter import repo_root
from bapmos.hpo.figure_export import save_matplotlib_figure
from bapmos.hpo.paths import HpoPaths
from bapmos.hpo.trial_utils import search_params_key, trial_flat_params

DEFAULT_CONFIRMATION_TRIALS = 15
DEFAULT_EPS_REL = 0.005
# Physical plateau tolerance; converted to objective units via suite pixel_spacing.
DEFAULT_EPS_ABS_PHYSICAL_MM = 0.001


@dataclass(frozen=True)
class ConvergenceDefaults:
    n_startup: int
    confirmation_trials: int
    eps_abs: float
    eps_rel: float
    default_n_trials: int
    eps_abs_physical_mm: float
    pixel_spacing_mm: float
    objective_units: str


@dataclass(frozen=True)
class SaturationReport:
    dataset: str
    hpo_suite: str
    objective_label: str
    n_completed: int
    best_value: Optional[float]
    best_trial: Optional[int]
    t_last: Optional[int]
    q: int
    n_eff: Optional[int]
    saturated: bool
    n_improvements: int
    n_startup: int
    confirmation_trials: int
    eps_abs: float
    eps_rel: float
    gpu_h_total: float
    gpu_h_to_t_last: float
    gpu_h_to_n_eff: float
    gpu_h_waste: float
    gpu_h_startup: float
    median_trial_min: float
    mean_trial_min: float


def convergence_defaults_for_suite(hpo_suite: str) -> ConvergenceDefaults:
    paths = HpoPaths.for_suite(hpo_suite)
    spec = paths.spec
    pixel_spacing_mm = float(spec.get("pixel_spacing_mm", 0.159072))
    eps_abs_physical_mm = float(
        spec.get("eps_abs_physical_mm", DEFAULT_EPS_ABS_PHYSICAL_MM)
    )
    # Optuna objectives are read from metrics.csv ``best_val_msd_mm`` (mm at dataset spacing).
    if "eps_abs" in spec:
        eps_abs = float(spec["eps_abs"])
    else:
        eps_abs = eps_abs_physical_mm
    objective_units = str(
        spec.get(
            "objective_units",
            "val_msd_mm",
        )
    )
    return ConvergenceDefaults(
        n_startup=int(spec.get("n_startup_trials", 20)),
        confirmation_trials=int(
            spec.get("confirmation_trials", DEFAULT_CONFIRMATION_TRIALS)
        ),
        eps_abs=eps_abs,
        eps_rel=float(spec.get("eps_rel", DEFAULT_EPS_REL)),
        default_n_trials=int(spec.get("default_n_trials", 100)),
        eps_abs_physical_mm=eps_abs_physical_mm,
        pixel_spacing_mm=pixel_spacing_mm,
        objective_units=objective_units,
    )


def trial_duration_s(trial) -> float:
    """Per-trial training wall time in seconds."""
    raw = trial.user_attrs.get("wall_time_s")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
    if trial.duration is not None:
        return max(0.0, trial.duration.total_seconds())
    return 0.0


def _epsilon_threshold(best: float, *, eps_abs: float, eps_rel: float) -> float:
    if best >= float("inf"):
        return eps_abs
    return max(eps_abs, eps_rel * best)


def _startup_param_keys_for_report(
    paths: HpoPaths,
    dataset: str,
    hpo_suite: str,
) -> set[tuple] | None:
    from bapmos.hpo.startup_catalog import (
        catalog_param_keys,
        read_startup_catalog,
        startup_catalog_path,
        uses_prequeued_startup,
    )

    if not uses_prequeued_startup(hpo_suite):
        return None
    cat_path = startup_catalog_path(paths, dataset)
    if not cat_path.is_file():
        return None
    return catalog_param_keys(read_startup_catalog(cat_path))


def _flat_search_params(trial) -> Dict[str, Any]:
    """Best-effort flat search params (safe for test doubles without system_attrs)."""
    params = getattr(trial, "params", None)
    if params:
        return dict(params)
    system_attrs = getattr(trial, "system_attrs", None)
    if system_attrs is not None:
        fixed = system_attrs.get("fixed_params")
        if isinstance(fixed, dict) and fixed:
            return dict(fixed)
    try:
        return trial_flat_params(trial)
    except AttributeError:
        return {}


def _trial_is_startup_catalog(
    trial,
    *,
    startup_param_keys: set[tuple] | None,
    n_startup: int,
) -> bool:
    """True when a trial belongs to the pre-enqueued startup catalog (or legacy # < n_startup)."""
    flat = _flat_search_params(trial)
    if flat and startup_param_keys is not None:
        return search_params_key(flat) in startup_param_keys
    return int(trial.number) < n_startup


def _startup_catalog_phase_complete(
    completed: Sequence,
    *,
    startup_param_keys: set[tuple] | None,
    n_startup: int,
) -> bool:
    """All ``n_startup`` catalog configs have at least one successful COMPLETE trial."""
    if startup_param_keys is None:
        return True
    seen: set[tuple] = set()
    numbered_startup = 0
    for trial in completed:
        if not _trial_is_startup_catalog(
            trial, startup_param_keys=startup_param_keys, n_startup=n_startup
        ):
            continue
        flat = _flat_search_params(trial)
        if flat:
            seen.add(search_params_key(flat))
        else:
            numbered_startup += 1
    return len(seen) >= n_startup or numbered_startup >= n_startup


def _saturation_counts_post_startup(
    completed: Sequence,
    *,
    startup_param_keys: set[tuple] | None,
    n_startup: int,
    confirmation_trials: int,
    eps_abs: float,
    eps_rel: float,
) -> tuple[int, int, int, bool, Optional[int], float]:
    """
    Saturation stats for TPE: K applies only after startup catalog is complete and
    only to post-startup (TPE-sampled) trials.

    Returns (q, n_improvements, t_last, saturated, n_eff, best).
  """
    if not _startup_catalog_phase_complete(
        completed,
        startup_param_keys=startup_param_keys,
        n_startup=n_startup,
    ):
        best = min((float(t.value) for t in completed), default=float("inf"))
        return 0, 0, None, False, None, best

    post_startup = [
        t
        for t in completed
        if not _trial_is_startup_catalog(
            t, startup_param_keys=startup_param_keys, n_startup=n_startup
        )
    ]

    best = float("inf")
    t_last: Optional[int] = None
    n_improvements = 0
    for trial in completed:
        value = float(trial.value)
        eps = _epsilon_threshold(best, eps_abs=eps_abs, eps_rel=eps_rel)
        if value < best - eps:
            best = value
            n_improvements += 1
            if trial in post_startup:
                t_last = trial.number

    if t_last is not None:
        after_last = [t for t in post_startup if t.number > t_last]
        q = len(after_last)
        if len(after_last) >= confirmation_trials:
            n_eff = after_last[confirmation_trials - 1].number
        else:
            n_eff = t_last + confirmation_trials
    else:
        q = len(post_startup)
        n_eff = None

    saturated = q >= confirmation_trials
    return q, n_improvements, t_last, saturated, n_eff, best


def analyze_saturation(
    trials: Sequence,
    *,
    dataset: str,
    hpo_suite: str,
    objective_label: str,
    n_startup: int,
    confirmation_trials: int,
    eps_abs: float,
    eps_rel: float,
    startup_param_keys: set[tuple] | None = None,
) -> SaturationReport:
    completed = sorted(
        [t for t in trials if t.state.name == "COMPLETE" and t.value is not None],
        key=lambda t: t.number,
    )
    if not completed:
        return SaturationReport(
            dataset=dataset,
            hpo_suite=hpo_suite,
            objective_label=objective_label,
            n_completed=0,
            best_value=None,
            best_trial=None,
            t_last=None,
            q=0,
            n_eff=None,
            saturated=False,
            n_improvements=0,
            n_startup=n_startup,
            confirmation_trials=confirmation_trials,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            gpu_h_total=0.0,
            gpu_h_to_t_last=0.0,
            gpu_h_to_n_eff=0.0,
            gpu_h_waste=0.0,
            gpu_h_startup=0.0,
            median_trial_min=0.0,
            mean_trial_min=0.0,
        )

    durations = [trial_duration_s(t) for t in completed]
    last_number = completed[-1].number

    if startup_param_keys is not None:
        q, n_improvements, t_last, saturated, n_eff, best = _saturation_counts_post_startup(
            completed,
            startup_param_keys=startup_param_keys,
            n_startup=n_startup,
            confirmation_trials=confirmation_trials,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )
    else:
        best = float("inf")
        t_last = None
        n_improvements = 0
        for trial in completed:
            value = float(trial.value)
            eps = _epsilon_threshold(best, eps_abs=eps_abs, eps_rel=eps_rel)
            if value < best - eps:
                best = value
                t_last = trial.number
                n_improvements += 1

        if t_last is not None:
            q = sum(1 for t in completed if t.number > t_last)
            after_last = [t for t in completed if t.number > t_last]
            if len(after_last) >= confirmation_trials:
                n_eff = after_last[confirmation_trials - 1].number
            else:
                n_eff = t_last + confirmation_trials
        else:
            q = 0
            n_eff = None
        saturated = q >= confirmation_trials

    def _gpu_h_through(trial_number: int) -> float:
        return sum(
            trial_duration_s(t) for t in completed if t.number <= trial_number
        ) / 3600.0

    gpu_h_total = sum(durations) / 3600.0
    gpu_h_to_t_last = _gpu_h_through(t_last) if t_last is not None else 0.0
    if n_eff is not None:
        cap = min(n_eff, last_number)
        gpu_h_to_n_eff = _gpu_h_through(cap)
    else:
        gpu_h_to_n_eff = 0.0
    gpu_h_waste = max(0.0, gpu_h_total - gpu_h_to_n_eff)

    def _is_startup_trial(trial) -> bool:
        return _trial_is_startup_catalog(
            trial, startup_param_keys=startup_param_keys, n_startup=n_startup
        )

    gpu_h_startup = sum(
        trial_duration_s(t) for t in completed if _is_startup_trial(t)
    ) / 3600.0

    trial_min = [d / 60.0 for d in durations if d > 0]
    if trial_min:
        median_min = float(sorted(trial_min)[len(trial_min) // 2])
        mean_min = sum(trial_min) / len(trial_min)
    else:
        median_min = mean_min = 0.0

    best_trial = min(completed, key=lambda t: float(t.value)).number

    return SaturationReport(
        dataset=dataset,
        hpo_suite=hpo_suite,
        objective_label=objective_label,
        n_completed=len(completed),
        best_value=best if best < float("inf") else None,
        best_trial=best_trial,
        t_last=t_last,
        q=q,
        n_eff=n_eff,
        saturated=saturated,
        n_improvements=n_improvements,
        n_startup=n_startup,
        confirmation_trials=confirmation_trials,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        gpu_h_total=gpu_h_total,
        gpu_h_to_t_last=gpu_h_to_t_last,
        gpu_h_to_n_eff=gpu_h_to_n_eff,
        gpu_h_waste=gpu_h_waste,
        gpu_h_startup=gpu_h_startup,
        median_trial_min=median_min,
        mean_trial_min=mean_min,
    )


def default_report_output_dir(hpo_suite: str) -> Path:
    paths = HpoPaths.for_suite(hpo_suite)
    if paths.suite in ("bapmos_bo_sam", "bapmos_bo_medsam"):
        backbone = "medsam" if "medsam" in paths.suite else "sam"
        return (
            repo_root()
            / "experiments"
            / "bladder"
            / "bapmos"
            / "figures"
            / "outer_loop"
            / backbone
        )
    method = paths.spec.get("search_method", paths.suite)
    if paths.suite.startswith("bapmos_bo"):
        return (
            repo_root()
            / "experiments"
            / "prostate"
            / "bapmos"
            / "figures"
            / "outer_loop"
            / method
        )
    return repo_root() / "docs" / "figures" / "hpo_convergence" / paths.hpo_version


def write_saturation_csv(reports: Sequence[SaturationReport], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(reports[0]).keys()) if reports else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))
    return path


def write_trial_cost_csv(study, path: Path) -> Path:
    """Per-trial objective, duration, and cumulative GPU-hours."""
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = sorted(
        [
            t
            for t in study.trials
            if t.state.name == "COMPLETE" and t.value is not None
        ],
        key=lambda t: t.number,
    )
    rows: List[Dict[str, Any]] = []
    cumulative_s = 0.0
    best = float("inf")
    for trial in completed:
        cumulative_s += trial_duration_s(trial)
        value = float(trial.value)
        if value < best:
            best = value
        rows.append(
            {
                "trial": trial.number,
                "value": value,
                "best_so_far": best,
                "duration_min": round(trial_duration_s(trial) / 60.0, 2),
                "cumulative_gpu_h": round(cumulative_s / 3600.0, 3),
                "wall_time_s": round(trial_duration_s(trial), 1),
                "slurm_job_id": trial.user_attrs.get("slurm_job_id", ""),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")
    return path


def write_time_cost_summary(
    report: SaturationReport,
    path: Path,
    *,
    max_concurrent_gpus: Optional[int] = None,
) -> Path:
    """Markdown table of campaign GPU-hours and per-trial timing for write-ups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# outer-loop time cost — {report.hpo_suite} / {report.dataset}",
        "",
        "GPU-hours are summed per-trial training wall times (1 GPU per Slurm worker).",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Completed trials | {report.n_completed} |",
    ]
    if report.best_value is not None:
        lines.append(f"| Best {report.objective_label} | {report.best_value:.4f} |")
        lines.append(f"| Best trial | {report.best_trial} |")
    if report.t_last is not None:
        lines.append(f"| Last improvement trial (T_last) | {report.t_last} |")
    if report.n_eff is not None:
        lines.append(f"| Effective budget N_eff | {report.n_eff} |")
    lines.extend(
        [
            f"| Saturated (K={report.confirmation_trials}) | {report.saturated} |",
            f"| Total GPU-hours | {report.gpu_h_total:.2f} |",
            f"| GPU-h to T_last | {report.gpu_h_to_t_last:.2f} |",
            f"| GPU-h to N_eff | {report.gpu_h_to_n_eff:.2f} |",
            f"| GPU-h waste (after N_eff) | {report.gpu_h_waste:.2f} |",
            f"| GPU-h startup (trials < {report.n_startup}) | {report.gpu_h_startup:.2f} |",
            f"| Median trial duration (min) | {report.median_trial_min:.1f} |",
            f"| Mean trial duration (min) | {report.mean_trial_min:.1f} |",
        ]
    )
    if max_concurrent_gpus and max_concurrent_gpus > 0:
        wall_h = report.gpu_h_total / float(max_concurrent_gpus)
        lines.extend(
            [
                f"| Max concurrent GPUs (assumed) | {max_concurrent_gpus} |",
                f"| Est. campaign wall-clock (h) | {wall_h:.2f} |",
            ]
        )
    lines.extend(
        [
            "",
            "Per-trial timings: `{dataset}_trials.csv` (params + duration) and "
            "`{dataset}_trial_costs.csv` (cumulative GPU-h).".format(
                dataset=report.dataset
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_saturation_json(
    reports: Sequence[SaturationReport],
    path: Path,
    *,
    hpo_suite: str,
) -> Path:
    defaults = convergence_defaults_for_suite(hpo_suite)
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "hpo_suite": hpo_suite,
        "protocol": {
            "n_startup": defaults.n_startup,
            "confirmation_trials": defaults.confirmation_trials,
            "eps_abs": defaults.eps_abs,
            "eps_abs_physical_mm": defaults.eps_abs_physical_mm,
            "eps_abs_unit": defaults.objective_units,
            "pixel_spacing_mm": defaults.pixel_spacing_mm,
            "eps_rel": defaults.eps_rel,
            "default_n_trials": defaults.default_n_trials,
            "epsilon_note": (
                "Plateau threshold ε = max(eps_abs, eps_rel × best) in objective units "
                "(val_msd_mm at the dataset pixel_spacing). eps_abs_physical_mm is the "
                "prespecified absolute tolerance in millimetres; prostate uses clinical "
                "spacing (0.159 mm/px) while PFUS1 bladder-pop uses placeholder 1.0 mm/px "
                "(1 px ≡ 1 mm in the objective), so best-so-far magnitudes differ even "
                "when eps_abs_physical_mm is matched. eps_rel is a dimensionless fraction "
                "applied to best-so-far in objective units."
            ),
            "saturation_rule": (
                "TPE (pre-enqueued startup catalog): saturation applies only after all "
                "n_startup catalog configs have COMPLETE trials, and Q counts post-startup "
                "(TPE-sampled) trials only. Let T_last be the last post-startup trial "
                "improving global best-so-far by more than max(eps_abs, eps_rel * best). "
                "If no post-startup trial has improved yet, Q = all post-startup completes. "
                "Saturated when Q >= confirmation_trials. Other suites: Q counts all trials "
                "after global T_last. N_eff = T_last + confirmation_trials when defined. "
                "GPU-hours sum per-trial training wall time (1 GPU per trial)."
            ),
        },
        "datasets": {report.dataset: asdict(report) for report in reports},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def render_compute_convergence_figure(
    study,
    *,
    output_path: Path,
    title: str,
    eps_abs: float,
    eps_rel: float,
    confirmation_trials: int,
    n_startup: int = 0,
    startup_param_keys: set[tuple] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    completed = sorted(
        [
            t
            for t in study.trials
            if t.state.name == "COMPLETE" and t.value is not None
        ],
        key=lambda t: t.number,
    )
    if not completed:
        raise ValueError("No completed trials to plot")

    gpu_h: List[float] = []
    best_y: List[float] = []
    cumulative_s = 0.0
    best = float("inf")
    t_last: Optional[int] = None
    for trial in completed:
        cumulative_s += trial_duration_s(trial)
        value = float(trial.value)
        eps = _epsilon_threshold(best, eps_abs=eps_abs, eps_rel=eps_rel)
        if value < best - eps:
            best = value
            t_last = trial.number
        gpu_h.append(cumulative_s / 3600.0)
        best_y.append(best)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gpu_h, best_y, color="#1f77b4", linewidth=2, label="Best so far")
    ax.scatter(gpu_h, best_y, s=12, alpha=0.35, color="#1f77b4")
    if t_last is not None:
        report = analyze_saturation(
            study.trials,
            dataset="",
            hpo_suite="",
            objective_label="",
            n_startup=n_startup,
            confirmation_trials=confirmation_trials,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            startup_param_keys=startup_param_keys,
        )
        if report.n_eff is not None:
            cap = min(report.n_eff, completed[-1].number)
            cutoff_h = sum(
                trial_duration_s(t)
                for t in completed
                if t.number <= cap
            ) / 3600.0
            ax.axvline(cutoff_h, color="#d62728", linestyle="--", linewidth=1.2, label=f"N_eff ({cutoff_h:.1f} GPU-h)")
    ax.set_xlabel("Cumulative GPU-hours")
    ax.set_ylabel("Best validation MSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    saved = save_matplotlib_figure(fig, output_path, dpi=160)
    plt.close(fig)
    return saved.png


def run_convergence_report(
    *,
    hpo_suite: str,
    dataset: str,
    output_dir: Optional[Path] = None,
    confirmation_trials: Optional[int] = None,
    eps_abs: Optional[float] = None,
    eps_rel: Optional[float] = None,
    n_startup: Optional[int] = None,
    max_concurrent_gpus: Optional[int] = None,
) -> Dict[str, Path]:
    from bapmos.hpo.study_runner import create_study

    paths = HpoPaths.for_suite(hpo_suite)
    defaults = convergence_defaults_for_suite(hpo_suite)
    study, _ = create_study(dataset, hpo_suite=hpo_suite)

    k = confirmation_trials if confirmation_trials is not None else defaults.confirmation_trials
    eps_a = eps_abs if eps_abs is not None else defaults.eps_abs
    eps_r = eps_rel if eps_rel is not None else defaults.eps_rel
    n0 = n_startup if n_startup is not None else defaults.n_startup
    startup_keys = _startup_param_keys_for_report(paths, dataset, hpo_suite)

    report = analyze_saturation(
        study.trials,
        dataset=dataset,
        hpo_suite=hpo_suite,
        objective_label=paths.objective_label,
        n_startup=n0,
        confirmation_trials=k,
        eps_abs=eps_a,
        eps_rel=eps_r,
        startup_param_keys=startup_keys,
    )

    out_root = output_dir or default_report_output_dir(hpo_suite)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / f"{dataset}_convergence.csv"
    write_saturation_csv([report], csv_path)

    trial_csv = out_root / f"{dataset}_trial_costs.csv"
    write_trial_cost_csv(study, trial_csv)

    json_path = out_root / f"{dataset}_convergence.json"
    write_saturation_json([report], json_path, hpo_suite=hpo_suite)

    method = paths.spec.get("search_method", hpo_suite)
    fig_path = out_root / f"{dataset}_convergence_by_gpu_h.png"
    render_compute_convergence_figure(
        study,
        output_path=fig_path,
        title=f"BAP-MOS outer-loop | {method} | {dataset}",
        eps_abs=eps_a,
        eps_rel=eps_r,
        confirmation_trials=k,
        n_startup=n0,
        startup_param_keys=startup_keys,
    )

    summary_path = out_root / f"{dataset}_time_cost_summary.md"
    write_time_cost_summary(
        report,
        summary_path,
        max_concurrent_gpus=max_concurrent_gpus,
    )

    return {
        "convergence_csv": csv_path,
        "trial_costs_csv": trial_csv,
        "convergence_json": json_path,
        "convergence_by_gpu_h_png": fig_path,
        "convergence_by_gpu_h_pdf": fig_path.with_suffix(".pdf"),
        "time_cost_summary_md": summary_path,
    }


def run_convergence_report_all(
    *,
    hpo_suite: str,
    datasets: Sequence[str],
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Tuple[List[SaturationReport], Dict[str, Path]]:
    from bapmos.hpo.study_runner import create_study

    paths = HpoPaths.for_suite(hpo_suite)
    defaults = convergence_defaults_for_suite(hpo_suite)
    reports: List[SaturationReport] = []
    outputs: Dict[str, Path] = {}

    k = kwargs.get("confirmation_trials", defaults.confirmation_trials)
    eps_a = kwargs.get("eps_abs", defaults.eps_abs)
    eps_r = kwargs.get("eps_rel", defaults.eps_rel)
    n0 = kwargs.get("n_startup", defaults.n_startup)

    out_root = output_dir or default_report_output_dir(hpo_suite)
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        study, _ = create_study(dataset, hpo_suite=hpo_suite)
        startup_keys = _startup_param_keys_for_report(paths, dataset, hpo_suite)
        report = analyze_saturation(
            study.trials,
            dataset=dataset,
            hpo_suite=hpo_suite,
            objective_label=paths.objective_label,
            n_startup=n0,
            confirmation_trials=k,
            eps_abs=eps_a,
            eps_rel=eps_r,
            startup_param_keys=startup_keys,
        )
        reports.append(report)
        outputs[f"{dataset}_trial_costs_csv"] = write_trial_cost_csv(
            study, out_root / f"{dataset}_trial_costs.csv"
        )
        method = paths.spec.get("search_method", hpo_suite)
        gpu_h_png = out_root / f"{dataset}_convergence_by_gpu_h.png"
        outputs[f"{dataset}_convergence_by_gpu_h_png"] = render_compute_convergence_figure(
            study,
            output_path=gpu_h_png,
            title=f"BAP-MOS outer-loop | {method} | {dataset}",
            eps_abs=eps_a,
            eps_rel=eps_r,
            confirmation_trials=k,
            n_startup=n0,
            startup_param_keys=startup_keys,
        )
        outputs[f"{dataset}_convergence_by_gpu_h_pdf"] = gpu_h_png.with_suffix(".pdf")
        outputs[f"{dataset}_time_cost_summary_md"] = write_time_cost_summary(
            report,
            out_root / f"{dataset}_time_cost_summary.md",
            max_concurrent_gpus=kwargs.get("max_concurrent_gpus"),
        )

    summary_csv = out_root / "convergence_summary.csv"
    write_saturation_csv(reports, summary_csv)
    outputs["convergence_summary_csv"] = summary_csv

    summary_json = out_root / "convergence_summary.json"
    write_saturation_json(reports, summary_json, hpo_suite=hpo_suite)
    outputs["convergence_summary_json"] = summary_json

    return reports, outputs
