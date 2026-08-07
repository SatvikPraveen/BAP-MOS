"""Optuna objective: train one BO trial and return validation MSD."""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from bapmos.method.bap_mos_trainer import run_training_job
from bapmos.method.bapmos_seeds import apply_training_seed_env
from bapmos.method.config_utils import compose_config, deep_merge, resolve_experiment
from bapmos.hpo.paths import DEFAULT_HPO_SUITE, HpoPaths
from bapmos.hpo.search_space import trial_overrides_for_suite
from bapmos.hpo.trial_utils import (
    search_params_from_training_config,
    search_params_key,
    should_prune_duplicate_params,
    trial_flat_params,
)

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

FAILED_OBJECTIVE = 1e6


def _keep_hpo_checkpoints() -> bool:
    """Debug escape hatch: set ``BAPMOS_KEEP_HPO_CHECKPOINTS=1`` to retain trial ``.pth``."""
    return os.environ.get("BAPMOS_KEEP_HPO_CHECKPOINTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def delete_trial_checkpoints(run_dir: Path) -> int:
    """
    Delete SAM checkpoints for a completed outer-loop (search) trial.

    Called only after the objective score is read from ``metrics.csv`` and
    stored on the Optuna trial — the checkpoint has no further use for search.
    Inner-loop / production training never calls this.
    """
    run_dir = Path(run_dir)
    removed = 0
    for pattern in ("*.pth", "*.pth.tmp"):
        for path in run_dir.glob(pattern):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                logger.warning("Could not delete checkpoint %s", path)
    return removed


def cleanup_bo_trial_artifacts(run_dir: Path, *, keep_metrics: bool = True) -> None:
    """
    Post-score cleanup for **outer_loop** trials only: drop checkpoints
    (required) and other heavy artifacts (wandb, logs). Keeps ``metrics.csv``
    for audit. Production (inner_loop) never invokes this.
    """
    if _keep_hpo_checkpoints():
        logger.info(
            "Keeping HPO checkpoints under %s (BAPMOS_KEEP_HPO_CHECKPOINTS set)",
            run_dir,
        )
        return

    n_ckpt = delete_trial_checkpoints(run_dir)
    if n_ckpt:
        logger.info("Deleted %d checkpoint file(s) under %s", n_ckpt, run_dir)

    for sub in ("wandb", "logs"):
        target = Path(run_dir) / sub
        if target.is_dir():
            import shutil

            try:
                shutil.rmtree(target)
            except OSError:
                logger.warning("Could not remove %s", target)

    if not keep_metrics:
        for name in ("metrics.csv", "config.json"):
            try:
                (Path(run_dir) / name).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete %s", Path(run_dir) / name)


def read_best_val_msd_mm(run_dir: Path) -> Optional[float]:
    """Read best validation objective MSD from a completed trial's metrics.csv."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        return None

    best = float("inf")
    saw_value = False
    with open(metrics_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_best = (row.get("best_val_msd_mm") or "").strip()
            if raw_best:
                try:
                    best = min(best, float(raw_best))
                    saw_value = True
                except ValueError:
                    pass
            raw_val = (row.get("val_msd_mm") or "").strip()
            if raw_val:
                try:
                    best = min(best, float(raw_val))
                    saw_value = True
                except ValueError:
                    pass
    return best if saw_value and best < float("inf") else None


def build_trial_config(
    dataset: str,
    trial: "optuna.Trial",
    *,
    hpo_suite: str = DEFAULT_HPO_SUITE,
    experiment_name: str = "bo_trial_seed42",
    seed: int = 42,
) -> Dict[str, Any]:
    """Compose BO base config and merge Optuna-suggested hyperparameters."""
    paths = HpoPaths.for_suite(hpo_suite)
    cfg = compose_config(paths.hpo_version, dataset)
    cfg = resolve_experiment(cfg, experiment_name)
    hpo_splits = os.environ.get("BPI_HPO_SPLITS_SUBDIR", "").strip()
    if hpo_splits:
        cfg.setdefault("common", {})["splits_subdir"] = hpo_splits
    cfg = deep_merge(cfg, trial_overrides_for_suite(hpo_suite, trial))
    trial_root = paths.trial_run_root()
    if trial_root:
        cfg["run_root"] = trial_root
    cfg["experiment_seed"] = seed
    cfg = apply_training_seed_env(cfg)
    cfg.setdefault("wandb", {})
    tags = list(cfg["wandb"].get("tags", []))
    # resolve_experiment tags production suites; strip that for HPO trial runs.
    tags = [t for t in tags if t != "production"]
    tags.extend(
        [
            "hpo",
            "outer_loop",
            paths.hpo_version,
            paths.suite,
            f"trial={trial.number}",
            f"dataset={dataset}",
            paths.study_metric_tag,
        ]
    )
    search_method = paths.spec.get("search_method")
    if search_method:
        tags.append(f"search:{search_method}")
        tags.append(f"hpo_suite:{paths.suite}")
    cfg["wandb"]["tags"] = list(dict.fromkeys(tags))
    # Group by HPO suite id so bladder SAM vs MedSAM (both TPE) do not collide.
    cfg["wandb"]["group"] = f"{paths.suite}_{dataset}"
    cfg["wandb"]["name"] = paths.trial_run_name(trial.number)
    # Outer-loop trials exist only to score HPs — always drop .pth after scoring.
    cfg.setdefault("evaluation", {})["cleanup_checkpoints_after_train"] = True
    return cfg


def resolve_trial_resume_checkpoint(
    cfg: Dict[str, Any],
    trial_number: int,
    *,
    hpo_suite: str = DEFAULT_HPO_SUITE,
) -> Optional[Path]:
    """
    Checkpoint for Slurm resume / preemption recovery.

    Uses the canonical trial run dir first, then any sibling run dir with the
    same 5-D search params (e.g. after purge + re-enqueue assigns a new number).
    """
    import json

    from bapmos.paths import resolve_under_project

    paths = HpoPaths.for_suite(hpo_suite)
    trial_root = paths.trial_run_root()
    if not trial_root:
        return None
    root = resolve_under_project(trial_root)
    root.mkdir(parents=True, exist_ok=True)
    target_key = search_params_key(search_params_from_training_config(cfg))

    def _ckpt_matches_params(ckpt: Path) -> bool:
        cfg_path = ckpt.parent / "config.json"
        if not cfg_path.is_file():
            return False
        try:
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            key = search_params_key(search_params_from_training_config(saved))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False
        return key == target_key

    # Canonical trial dir first — but only if its saved search params match.
    # After purge/re-enqueue, trial numbers can collide with stale orphan dirs.
    primary = root / paths.trial_run_name(trial_number) / "last_checkpoint.pth"
    if primary.is_file():
        if _ckpt_matches_params(primary):
            return primary
        logger.warning(
            "Trial %d: ignoring stale checkpoint %s (search params mismatch)",
            trial_number,
            primary,
        )

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        ckpt = child / "last_checkpoint.pth"
        if not ckpt.is_file():
            continue
        if _ckpt_matches_params(ckpt):
            logger.info(
                "Trial %d: resuming from preempted run dir %s",
                trial_number,
                ckpt,
            )
            return ckpt
    return None


def run_trial_training(
    cfg: Dict[str, Any],
    trial_number: int,
    *,
    hpo_suite: str = DEFAULT_HPO_SUITE,
    no_wandb: bool = False,
) -> Path:
    """Train one BO trial; returns run directory."""
    if no_wandb:
        cfg = dict(cfg)
        cfg.setdefault("wandb", {})["enabled"] = False
    cfg.setdefault("evaluation", {})["run_test_after_train"] = False

    run_name = HpoPaths.for_suite(hpo_suite).trial_run_name(trial_number)
    resume_ckpt = resolve_trial_resume_checkpoint(
        cfg, trial_number, hpo_suite=hpo_suite
    )
    return run_training_job(
        cfg,
        skip_test=True,
        run_name=run_name,
        resume_ckpt=resume_ckpt,
    )


def make_objective(
    dataset: str,
    *,
    hpo_suite: str = DEFAULT_HPO_SUITE,
    experiment_name: str = "bo_trial_seed42",
    no_wandb: bool = False,
):
    """Return an Optuna objective callable for *dataset*."""
    paths = HpoPaths.for_suite(hpo_suite)

    def objective(trial: "optuna.Trial") -> float:
        import optuna

        # Reload so parallel workers see RUNNING/COMPLETE siblings, not a stale cache.
        fresh_study = optuna.load_study(
            study_name=paths.study_name(dataset),
            storage=trial.study._storage,
        )
        cfg = build_trial_config(
            dataset,
            trial,
            hpo_suite=hpo_suite,
            experiment_name=experiment_name,
        )
        flat = trial_flat_params(trial)
        prune, reason = should_prune_duplicate_params(
            fresh_study, flat, trial.number
        )
        if prune:
            logger.warning(
                "Trial %d: duplicate params of %s; skipping training",
                trial.number,
                reason,
            )
            raise optuna.TrialPruned(f"duplicate params of {reason}")
        t0 = time.perf_counter()
        try:
            run_dir = run_trial_training(
                cfg,
                trial.number,
                hpo_suite=hpo_suite,
                no_wandb=no_wandb,
            )
        except Exception:
            logger.exception("Trial %d failed during training", trial.number)
            raise
        wall_time_s = time.perf_counter() - t0

        best_msd = read_best_val_msd_mm(run_dir)
        if best_msd is None:
            msg = f"Trial {trial.number}: no val MSD in {run_dir}"
            logger.error(msg)
            raise RuntimeError(msg)

        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("best_val_msd_mm", best_msd)
        trial.set_user_attr("metric", paths.objective_label)
        trial.set_user_attr("hpo_suite", paths.suite)
        trial.set_user_attr("search_method", paths.spec.get("search_method", paths.suite))
        trial.set_user_attr("wall_time_s", round(wall_time_s, 2))
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            trial.set_user_attr("slurm_job_id", slurm_job_id)

        # Search only needs the Optuna score; delete trial checkpoints unconditionally
        # (unless BAPMOS_KEEP_HPO_CHECKPOINTS=1). Do not delete on mid-train failure
        # so Slurm preemption can still resume last_checkpoint.pth.
        cleanup_bo_trial_artifacts(run_dir)

        logger.info(
            "Trial %d finished | %s=%.4f | run_dir=%s",
            trial.number,
            paths.objective_label,
            best_msd,
            run_dir,
        )
        return float(best_msd)

    return objective
