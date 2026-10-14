#!/usr/bin/env python3
"""
Plot outer-loop study diagnostics: objective convergence and hyperparameter trends.

Figures go to the suite's report folder (see ``convergence_report.default_report_output_dir``).
The usual entry point is ``python -m bapmos.hpo.study_runner plot-report``.

Usage (repository root)::

    python -m bapmos.hpo.plot_hpo_study --hpo-suite bapmos_bo --dataset pooled
    python -m bapmos.hpo.plot_hpo_study --hpo-suite bapmos_bo_medsam --dataset pfus1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from bapmos.method.data_adapter import repo_root
from bapmos.hpo.convergence_report import (
    convergence_defaults_for_suite,
    render_compute_convergence_figure,
)
from bapmos.hpo.figure_export import save_matplotlib_figure
from bapmos.hpo.paths import HPO_SUITES, HpoPaths, normalize_hpo_suite
from bapmos.hpo.search_space import search_profile_for_suite

FIGURE_DPI = 200

PROSTATE_DATASETS: Tuple[str, ...] = ("pooled",)
PFUS1_DATASETS: Tuple[str, ...] = ("pfus1",)
DATASETS: Tuple[str, ...] = PROSTATE_DATASETS
PFUS1_HPO_SUITES = frozenset({"bapmos_bo_sam", "bapmos_bo_medsam"})
PROSTATE_HPO_SUITES = frozenset(
    {
        "bapmos_bo",
        "bapmos_bo_random",
        "bapmos_bo_heuristic",
        "bapmos_bo_greedy",
        "bapmos_bo_medsam_pooled",
    }
)

DATASET_TITLE: Dict[str, str] = {
    "pooled": "Pooled prostate",
    "pfus1": "PFUS1",
}

DATASET_COLORS: Dict[str, str] = {
    "pooled": "#9467bd",
    "pfus1": "#1f77b4",
}

# Plot order of the searched parameters (tau, alpha, r_min, W, K).
UNIFIED_PARAM_ORDER: Tuple[str, ...] = (
    "clip_max_mm",
    "alpha",
    "r_min",
    "window_size",
    "block_size_batches",
)

PARAM_LABEL: Dict[str, str] = {
    "tau_max_mm": r"$\tau_{\max}$ (mm)",
    "decay_fraction": "decay fraction",
    "clip_max_mm": r"clip $\tau_{\max}$ (mm)",
    "alpha": r"$\alpha$ (organ ring)",
    "r_min": r"$r_{\min}$ (px)",
    "window_size": "bandit window",
    "block_size_batches": "block size (batches)",
}


@dataclass(frozen=True)
class CompletedTrial:
    """Successful trial reduced to its number, objective value and float-cast params."""

    number: int
    value: float
    params: Dict[str, float]


def completed_trial_indices(
    n: int,
    *,
    one_based: bool = False,
) -> List[int]:
    """Sequential x-axis positions for *n* completed trials (no Optuna ID gaps)."""
    start = 1 if one_based else 0
    return list(range(start, start + n))


def completed_trial_rank(
    trials: Sequence[CompletedTrial],
    trial_number: int,
    *,
    one_based: bool = False,
) -> int:
    """Map an Optuna trial number to its sequential plot index."""
    for rank, trial in enumerate(trials):
        if trial.number == trial_number:
            return rank + (1 if one_based else 0)
    raise ValueError(f"Trial #{trial_number} not in completed trial list")


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required. Install with: pip install -r requirements-hpo.txt"
        ) from exc
    return optuna


def datasets_for_suite(hpo_suite: str) -> Tuple[str, ...]:
    """Datasets plotted for a suite (bladder → ``pfus1``, prostate → ``pooled``)."""
    if hpo_suite in PFUS1_HPO_SUITES:
        return PFUS1_DATASETS
    if hpo_suite in PROSTATE_HPO_SUITES:
        return PROSTATE_DATASETS
    return DATASETS


def _objective_ylabel(hpo_suite: str) -> str:
    label = HpoPaths.for_suite(hpo_suite).objective_label
    if "px" in label:
        return "Validation MSD (px)"
    if "ptv" in label or "fullres" in label:
        return "Validation MSD (mm)"
    return "Objective"


def _searched_param_keys(
    hpo_suite: str,
    trials: Sequence[CompletedTrial],
) -> Tuple[str, ...]:
    """Optuna trial param names only (excludes fixed-but-not-searched keys)."""
    if not trials:
        return _param_keys_for_suite(hpo_suite)
    keys = set(trials[0].params)
    for t in trials[1:]:
        keys &= set(t.params)
    preferred = _param_keys_for_suite(hpo_suite)
    ordered = [k for k in preferred if k in keys]
    rest = sorted(k for k in keys if k not in ordered)
    return tuple(ordered + rest)


def _param_keys_for_suite(hpo_suite: str) -> Tuple[str, ...]:
    """Return flat Optuna param keys for plotting (excludes fixed metadata)."""
    profile = search_profile_for_suite(hpo_suite)
    baseline = profile.baseline_overrides()
    flat = profile.overrides_to_flat(baseline)
    skip = {"tau_min_mm", "bandit_memory"}
    keys = [k for k in flat if k not in skip]
    # Preserve curriculum / ring / bandit order when using unified-like spaces.
    ordered = [k for k in UNIFIED_PARAM_ORDER if k in keys]
    rest = [k for k in keys if k not in ordered]
    return tuple(ordered + rest)


def load_completed_trials(hpo_suite: str, dataset: str) -> List[CompletedTrial]:
    """All successful COMPLETE trials (startup catalog + post-startup through saturation).

    Saturation (K, ε) does not filter this list — every successful trial in the
    study is included for plotting.
    """
    optuna = _require_optuna()
    from bapmos.hpo.trial_utils import successful_trials, trial_flat_params

    paths = HpoPaths.for_suite(hpo_suite)
    db_path = paths.study_db_path(dataset)
    if not db_path.is_file():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")

    study = optuna.load_study(
        study_name=paths.study_name(dataset),
        storage=f"sqlite:///{db_path.resolve()}",
    )
    completed = successful_trials(study)
    completed.sort(key=lambda t: t.number)
    rows: List[CompletedTrial] = []
    for t in completed:
        params = trial_flat_params(t)
        if not params:
            continue
        rows.append(
            CompletedTrial(
                number=t.number,
                value=float(t.value),
                params={k: float(params[k]) for k in params},
            )
        )
    return rows


def best_so_far_series(
    trials: Sequence[CompletedTrial],
    *,
    one_based: bool = False,
) -> Tuple[List[int], List[float], List[int]]:
    """Sequential plot indices, best objective values, and best Optuna trial ids."""
    xs = completed_trial_indices(len(trials), one_based=one_based)
    ys: List[float] = []
    best_ids: List[int] = []
    best_val = float("inf")
    best_id = -1
    for t in trials:
        if t.value < best_val:
            best_val = t.value
            best_id = t.number
        ys.append(best_val)
        best_ids.append(best_id)
    return xs, ys, best_ids


def plot_convergence(
    *,
    hpo_suite: str,
    datasets: Sequence[str],
    out_path: Path,
) -> None:
    """One panel per dataset: every trial's objective plus best-so-far vs completed-trial index."""
    paths = HpoPaths.for_suite(hpo_suite)
    production = paths.production_version.replace("_", ".")
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    plotted = False
    for ax, dataset in zip(axes, datasets):
        try:
            trials = load_completed_trials(hpo_suite, dataset)
        except FileNotFoundError:
            ax.set_title(f"{DATASET_TITLE.get(dataset, dataset)} (no study)")
            ax.axis("off")
            continue
        if not trials:
            ax.set_title(f"{DATASET_TITLE.get(dataset, dataset)} (no completed trials)")
            ax.axis("off")
            continue
        plotted = True
        numbers = completed_trial_indices(len(trials))
        values = [t.value for t in trials]
        best_x, best_y, _ = best_so_far_series(trials)
        color = DATASET_COLORS.get(dataset, "#333333")

        ax.scatter(numbers, values, s=18, alpha=0.45, color=color, label="Trial value")
        ax.plot(best_x, best_y, color=color, linewidth=2, label="Best so far")
        ax.set_title(DATASET_TITLE.get(dataset, dataset))
        ax.set_xlabel("Completed trial #")
        ax.set_ylabel(_objective_ylabel(hpo_suite))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if not plotted:
        plt.close(fig)
        print("  skip convergence: no completed trials")
        return

    fig.suptitle(
        f"TPE convergence ({hpo_suite} — {production} Stage-1 objective)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _first_completed_trials(hpo_suite: str, datasets: Sequence[str]) -> List[CompletedTrial]:
    for dataset in datasets:
        try:
            trials = load_completed_trials(hpo_suite, dataset)
        except FileNotFoundError:
            continue
        if trials:
            return trials
    return []


def plot_parameter_trends(
    *,
    hpo_suite: str,
    datasets: Sequence[str],
    out_path: Path,
) -> None:
    """Scatter each searched hyperparameter vs trial #, colored by objective."""
    paths = HpoPaths.for_suite(hpo_suite)
    production = paths.production_version.replace("_", ".")
    seed_trials = _first_completed_trials(hpo_suite, datasets)
    if not seed_trials:
        print("  skip parameter_trends: no completed trials")
        return
    param_keys = _searched_param_keys(hpo_suite, seed_trials)
    nrows = len(param_keys)
    ncols = len(datasets)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.4 * nrows),
        sharex=True,
        squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        try:
            trials = load_completed_trials(hpo_suite, dataset)
        except FileNotFoundError:
            continue
        if not trials:
            continue
        numbers = np.array(completed_trial_indices(len(trials)))
        values = np.array([t.value for t in trials])
        best_trial = min(trials, key=lambda t: t.value)
        best_x = completed_trial_rank(trials, best_trial.number)
        norm = Normalize(vmin=values.min(), vmax=values.max())

        for row, key in enumerate(param_keys):
            ax = axes[row, col]
            param_vals = np.array([t.params[key] for t in trials])
            sc = ax.scatter(
                numbers,
                param_vals,
                c=values,
                cmap="viridis_r",
                s=22,
                alpha=0.85,
                norm=norm,
            )
            ax.scatter(
                [best_x],
                [best_trial.params[key]],
                marker="*",
                s=140,
                c="red",
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
                label="Best trial" if row == 0 and col == 0 else None,
            )
            if row == 0:
                ax.set_title(DATASET_TITLE.get(dataset, dataset))
            if col == 0:
                ax.set_ylabel(PARAM_LABEL.get(key, key))
            if row == nrows - 1:
                ax.set_xlabel("Completed trial #")
            ax.grid(True, alpha=0.25)

        cbar = fig.colorbar(sc, ax=axes[:, col], fraction=0.025, pad=0.02)
        cbar.set_label(_objective_ylabel(hpo_suite), fontsize=8)

    fig.suptitle(
        f"Hyperparameter exploration ({hpo_suite} — {production}); "
        "color = objective, ★ = best trial",
        fontsize=12,
        y=1.01,
    )
    fig.subplots_adjust(top=0.94, hspace=0.35, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_best_so_far_parameters(
    *,
    hpo_suite: str,
    datasets: Sequence[str],
    out_path: Path,
) -> None:
    """Track hyperparameter values from the best-so-far trial as search progresses."""
    paths = HpoPaths.for_suite(hpo_suite)
    production = paths.production_version.replace("_", ".")
    seed_trials = _first_completed_trials(hpo_suite, datasets)
    if not seed_trials:
        print("  skip best_so_far_parameters: no completed trials")
        return
    param_keys = _searched_param_keys(hpo_suite, seed_trials)
    nrows = len(param_keys)
    ncols = len(datasets)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.4 * nrows),
        sharex=True,
        squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        try:
            trials = load_completed_trials(hpo_suite, dataset)
        except FileNotFoundError:
            continue
        if not trials:
            continue
        numbers = completed_trial_indices(len(trials))
        _, _, best_ids = best_so_far_series(trials)
        by_number = {t.number: t for t in trials}
        color = DATASET_COLORS.get(dataset, "#333333")

        for row, key in enumerate(param_keys):
            ax = axes[row, col]
            y = [by_number[bid].params[key] for bid in best_ids]
            ax.plot(numbers, y, color=color, linewidth=1.8, marker=".", markersize=4)
            ax.scatter(
                [numbers[-1] if numbers else 0],
                [y[-1] if y else 0],
                marker="*",
                s=120,
                c="red",
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )
            if row == 0:
                ax.set_title(DATASET_TITLE.get(dataset, dataset))
            if col == 0:
                ax.set_ylabel(PARAM_LABEL.get(key, key))
            if row == nrows - 1:
                ax.set_xlabel("Completed trial #")
            ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Best-so-far hyperparameters vs trial ({hpo_suite} — {production})",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_figure(fig, out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_optimization_history_from_completed(
    trials: Sequence[CompletedTrial],
    *,
    ylabel: str,
    title: str,
) -> plt.Figure:
    """Best-so-far optimization history using dense completed-trial indices.

    Used when Optuna's ``plot_optimization_history`` fails on sparse parallel
    Slurm trial numbers. Includes every trial in *trials* (startup + post-startup).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = completed_trial_indices(len(trials), one_based=True)
    values = [t.value for t in trials]
    best_x, best_y, _ = best_so_far_series(trials, one_based=True)
    ax.scatter(xs, values, s=22, alpha=0.45, label="Trial value")
    ax.plot(best_x, best_y, linewidth=2, label="Best so far")
    ax.set_xlabel("Completed trial #")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _figure_prefix(hpo_suite: str) -> str:
    return HpoPaths.for_suite(hpo_suite).hpo_version


def _load_study(hpo_suite: str, dataset: str):
    optuna = _require_optuna()
    paths = HpoPaths.for_suite(hpo_suite)
    db_path = paths.study_db_path(dataset)
    if not db_path.is_file():
        raise FileNotFoundError(f"Optuna DB not found: {db_path}")
    return optuna.load_study(
        study_name=paths.study_name(dataset),
        storage=f"sqlite:///{db_path.resolve()}",
    )


def plot_optuna_matplotlib_diagnostics(
    *,
    hpo_suite: str,
    dataset: str,
    out_dir: Path,
) -> List[Path]:
    """Per-dataset Optuna/matplotlib diagnostics (slice, contour, importance, …)."""
    from optuna.visualization.matplotlib import (
        plot_contour,
        plot_edf,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )

    from bapmos.hpo.trial_utils import successful_trials

    study = _load_study(hpo_suite, dataset)
    complete = successful_trials(study)
    if len(complete) < 2:
        print(f"  skip {dataset}: need >=2 successful trials (have {len(complete)})")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_completed_trials(hpo_suite, dataset)
    param_keys = list(_searched_param_keys(hpo_suite, trials))
    written: List[Path] = []

    def _save(obj, name: str) -> None:
        path = out_dir / f"{name}.png"
        if isinstance(obj, np.ndarray):
            fig = obj.flat[0].figure
        elif hasattr(obj, "figure"):
            fig = obj.figure
        else:
            fig = obj
        saved = save_matplotlib_figure(fig, path, dpi=FIGURE_DPI)
        plt.close(fig)
        written.extend([saved.png, saved.pdf])

    _save(plot_edf(study), "edf")
    ylabel = _objective_ylabel(hpo_suite)
    title = f"{hpo_suite} — {dataset}"
    try:
        _save(plot_optimization_history(study), "optimization_history")
    except (IndexError, ValueError) as exc:
        print(
            f"  {dataset} optimization_history: Optuna plot failed ({exc}); "
            f"using all {len(trials)} completed trials"
        )
        fig = plot_optimization_history_from_completed(
            trials, ylabel=ylabel, title=title
        )
        _save(fig, "optimization_history")
    _save(plot_param_importances(study), "param_importances")
    _save(plot_slice(study, params=param_keys), "slice")
    _save(plot_parallel_coordinate(study, params=param_keys), "parallel_coordinate")
    if len(param_keys) >= 2:
        _save(
            plot_contour(study, params=param_keys[:2]),
            f"contour_{param_keys[0]}_{param_keys[1]}",
        )

    return written


def discover_hpo_suites_with_studies() -> List[str]:
    """Suite keys with a same-named ``optuna_studies/<suite>/`` directory containing a ``.db``."""
    root = repo_root() / "optuna_studies"
    if not root.is_dir():
        return []
    suites: List[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*.db")):
            key = child.name
            if key in HPO_SUITES:
                suites.append(key)
    return suites


def plot_suite(
    *,
    hpo_suite: str,
    datasets: Sequence[str],
    out_dir: Path,
) -> List[Path]:
    """Write suite-level and per-dataset figures under *out_dir*."""
    prefix = _figure_prefix(hpo_suite)
    written: List[Path] = []

    convergence_path = out_dir / f"{prefix}_convergence.png"
    trends_path = out_dir / f"{prefix}_parameter_trends.png"
    best_params_path = out_dir / f"{prefix}_best_so_far_parameters.png"

    plot_convergence(hpo_suite=hpo_suite, datasets=datasets, out_path=convergence_path)
    if convergence_path.is_file():
        written.extend([convergence_path, convergence_path.with_suffix(".pdf")])
    plot_parameter_trends(hpo_suite=hpo_suite, datasets=datasets, out_path=trends_path)
    if trends_path.is_file():
        written.extend([trends_path, trends_path.with_suffix(".pdf")])
    plot_best_so_far_parameters(hpo_suite=hpo_suite, datasets=datasets, out_path=best_params_path)
    if best_params_path.is_file():
        written.extend([best_params_path, best_params_path.with_suffix(".pdf")])

    for dataset in datasets:
        try:
            paths = HpoPaths.for_suite(hpo_suite)
            defaults = convergence_defaults_for_suite(hpo_suite)
            optuna = _require_optuna()
            db_path = paths.study_db_path(dataset)
            if db_path.is_file():
                study = optuna.load_study(
                    study_name=paths.study_name(dataset),
                    storage=f"sqlite:///{db_path.resolve()}",
                )
                compute_path = out_dir / dataset / "convergence_by_gpu_h.png"
                render_compute_convergence_figure(
                    study,
                    output_path=compute_path,
                    title=f"{hpo_suite} — {DATASET_TITLE.get(dataset, dataset)}",
                    eps_abs=defaults.eps_abs,
                    eps_rel=defaults.eps_rel,
                    confirmation_trials=defaults.confirmation_trials,
                )
                written.extend([compute_path, compute_path.with_suffix(".pdf")])
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skip compute convergence {dataset}: {exc}")

        try:
            paths = plot_optuna_matplotlib_diagnostics(
                hpo_suite=hpo_suite,
                dataset=dataset,
                out_dir=out_dir / dataset,
            )
            written.extend(paths)
        except FileNotFoundError as exc:
            print(f"  skip {dataset}: {exc}")

    return written


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: write suite and per-dataset study figures (see :func:`plot_suite`)."""
    parser = argparse.ArgumentParser(description="Plot Optuna TPE convergence and parameter trends.")
    parser.add_argument(
        "--hpo-suite",
        default="bapmos_bo",
        choices=sorted(HPO_SUITES.keys()),
        help="Outer-loop suite (default: bapmos_bo, prostate TPE)",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=[*PROSTATE_DATASETS, *PFUS1_DATASETS, "all"],
        help="Dataset(s) to plot (default: all for the suite)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output root (default: the suite's report folder under experiments/)"
        ),
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Plot every HPO suite that has an optuna_studies/<suite>/*.db",
    )
    args = parser.parse_args(argv)

    from bapmos.hpo.convergence_report import default_report_output_dir

    hpo_suite = normalize_hpo_suite(args.hpo_suite)
    pool = list(datasets_for_suite(hpo_suite))
    base_out = args.out_dir or default_report_output_dir(hpo_suite)

    if args.all_suites:
        suites = discover_hpo_suites_with_studies()
        if not suites:
            raise SystemExit("No Optuna study DBs found under optuna_studies/")
        for suite_key in suites:
            hpo_suite = normalize_hpo_suite(suite_key)
            pool = list(datasets_for_suite(hpo_suite))
            datasets = pool if args.dataset == "all" else [args.dataset]
            out_dir = base_out if args.out_dir else base_out / _figure_prefix(hpo_suite)
            print(f"Plotting {hpo_suite} -> {out_dir}")
            for path in plot_suite(hpo_suite=hpo_suite, datasets=datasets, out_dir=out_dir):
                print(f"  Wrote {path}")
        return

    if args.dataset == "all":
        datasets = pool
    elif args.dataset in pool:
        datasets = [args.dataset]
    else:
        choices = ", ".join(pool)
        raise SystemExit(f"Dataset {args.dataset!r} invalid for {hpo_suite}; choose: {choices}, all")

    out_dir = base_out
    print(f"Plotting {hpo_suite} -> {out_dir}")
    for path in plot_suite(hpo_suite=hpo_suite, datasets=datasets, out_dir=out_dir):
        print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
