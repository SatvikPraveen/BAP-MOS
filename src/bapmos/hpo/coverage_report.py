"""Startup-phase coverage plots for BAP-MOS outer-loop Optuna studies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bapmos.hpo.catalog_search import SEARCH_PARAM_KEYS, TPE_N_STARTUP_TRIALS
from bapmos.hpo.convergence_report import default_report_output_dir, trial_duration_s
from bapmos.hpo.figure_export import save_matplotlib_figure
from bapmos.hpo.paths import HpoPaths
from bapmos.hpo.startup_catalog import (
    startup_trial_numbers_for_study,
    uses_prequeued_startup,
)
from bapmos.hpo.trial_utils import trial_flat_params

# Bin edges for Excel-style coarse grids (clip × alpha and r_min × window).
CLIP_BINS = [5, 10, 15, 20, 25, 30]
ALPHA_BINS = [0.05, 0.075, 0.10, 0.125, 0.15]
RMIN_BINS = [2, 6, 10, 14, 20]
WINDOW_BINS = [20, 40, 60, 80, 100]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "clip_max_mm": (5.0, 30.0),
    "alpha": (0.05, 0.15),
    "r_min": (2.0, 20.0),
    "window_size": (20.0, 100.0),
    "block_size_batches": (20.0, 100.0),
}


def _nearest_bin(value: float, bins: Sequence[float]) -> int:
    return min(range(len(bins)), key=lambda i: abs(float(bins[i]) - float(value)))


def _trial_rows(study) -> List[Dict[str, Any]]:
    completed = sorted(
        [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None],
        key=lambda t: t.number,
    )
    rows: List[Dict[str, Any]] = []
    cumulative_s = 0.0
    best = float("inf")
    for trial in completed:
        duration_s = trial_duration_s(trial)
        cumulative_s += duration_s
        value = float(trial.value)
        if value < best:
            best = value
        row = {
            "trial": trial.number,
            "value": value,
            "best_so_far": best,
            "state": trial.state.name,
            "duration_min": round(duration_s / 60.0, 2),
            "wall_time_s": round(duration_s, 1),
            "cumulative_gpu_h": round(cumulative_s / 3600.0, 3),
            "slurm_job_id": trial.user_attrs.get("slurm_job_id", ""),
        }
        row.update(trial_flat_params(trial))
        rows.append(row)
    return rows


def startup_trial_numbers(
    study,
    *,
    n_startup: int = TPE_N_STARTUP_TRIALS,
    hpo_suite: str | None = None,
    dataset: str | None = None,
) -> List[int]:
    """Trials that belong to the pre-enqueued startup catalog (or legacy # < n_startup)."""
    if hpo_suite and dataset and uses_prequeued_startup(hpo_suite):
        paths = HpoPaths.for_suite(hpo_suite)
        cat_path = paths.study_db_path(dataset).with_name(
            f"{paths.study_db_path(dataset).stem}_startup_catalog.csv"
        )
        if cat_path.is_file():
            from bapmos.hpo.startup_catalog import read_startup_catalog

            catalog = read_startup_catalog(cat_path)
            return startup_trial_numbers_for_study(study, catalog)
    return [t.number for t in study.trials if t.number < n_startup]


def write_trials_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_coverage_figure(
    study,
    *,
    output_path: Path,
    n_startup: int = TPE_N_STARTUP_TRIALS,
    title: str = "outer-loop startup coverage",
    hpo_suite: str | None = None,
    dataset: str | None = None,
) -> Path:
    """Dual-panel coarse grid: red cells = hit by at least one startup trial."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    startup_ids = set(
        startup_trial_numbers(
            study,
            n_startup=n_startup,
            hpo_suite=hpo_suite,
            dataset=dataset,
        )
    )
    startup_rows = [
        t for t in study.trials if t.number in startup_ids and t.state.name == "COMPLETE"
    ]

    def hits_panel(x_bins, y_bins, x_key, y_key) -> set[Tuple[int, int]]:
        hit: set[Tuple[int, int]] = set()
        for trial in startup_rows:
            params = trial_flat_params(trial)
            if x_key not in params or y_key not in params:
                continue
            xi = _nearest_bin(float(params[x_key]), x_bins)
            yi = _nearest_bin(float(params[y_key]), y_bins)
            hit.add((yi, xi))
        return hit

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (axes[0], CLIP_BINS, ALPHA_BINS, "clip_max_mm", "alpha", "clip × α"),
        (axes[1], RMIN_BINS, WINDOW_BINS, "r_min", "window_size", "r_min × window"),
    ]

    for ax, x_bins, y_bins, x_key, y_key, subtitle in panels:
        hit = hits_panel(x_bins, y_bins, x_key, y_key)
        n_rows, n_cols = len(y_bins), len(x_bins)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
        ax.set_xticks([i + 0.5 for i in range(n_cols)])
        ax.set_xticklabels([str(v) for v in x_bins], rotation=45, ha="right", fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels([str(v) for v in y_bins], fontsize=8)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(False)

        for row in range(n_rows):
            for col in range(n_cols):
                trial = next(
                    (
                        t
                        for t in startup_rows
                        if _nearest_bin(float(trial_flat_params(t).get(x_key, 0)), x_bins) == col
                        and _nearest_bin(float(trial_flat_params(t).get(y_key, 0)), y_bins) == row
                    ),
                    None,
                )
                if trial is None:
                    label = ""
                    face = "#f5f5f5"
                    edge = "#cccccc"
                else:
                    p = trial_flat_params(trial)
                    label = (
                        f"{int(p.get('clip_max_mm', 0))},"
                        f"{p.get('alpha', 0):.2f},"
                        f"{int(p.get('r_min', 0))},"
                        f"{int(p.get('window_size', 0))},"
                        f"{int(p.get('block_size_batches', 0))}"
                    )
                    face = "#ffcccc" if (row, col) in hit else "#ffffff"
                    edge = "#cc0000" if (row, col) in hit else "#cccccc"

                rect = Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.2,
                )
                ax.add_patch(rect)
                if label:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="#990000" if (row, col) in hit else "#333333",
                    )

    fig.suptitle(
        f"{title}\n"
        f"red = pre-enqueued startup catalog ({n_startup} configs); "
        f"{len(startup_rows)} completed in startup window",
        fontsize=11,
    )
    fig.tight_layout()
    saved = save_matplotlib_figure(fig, output_path, dpi=160)
    plt.close(fig)
    return saved.png


def run_coverage_report(
    *,
    hpo_suite: str,
    dataset: str,
    output_dir: Optional[Path] = None,
    n_startup: int = TPE_N_STARTUP_TRIALS,
) -> Dict[str, Path]:
    from bapmos.hpo.study_runner import create_study

    paths = HpoPaths.for_suite(hpo_suite)
    study, _ = create_study(dataset, hpo_suite=hpo_suite)
    method = paths.spec.get("search_method", hpo_suite)
    out_root = output_dir or default_report_output_dir(hpo_suite)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _trial_rows(study)
    csv_path = out_root / f"{dataset}_trials.csv"
    write_trials_csv(rows, csv_path)

    fig_path = out_root / f"{dataset}_startup_coverage.png"
    render_coverage_figure(
        study,
        output_path=fig_path,
        n_startup=n_startup,
        title=f"BAP-MOS outer-loop | {method} | {dataset}",
        hpo_suite=hpo_suite,
        dataset=dataset,
    )
    return {
        "trials_csv": csv_path,
        "coverage_png": fig_path,
        "coverage_pdf": fig_path.with_suffix(".pdf"),
    }
