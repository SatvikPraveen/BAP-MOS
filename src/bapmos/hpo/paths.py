"""Filesystem paths for Optuna studies and BO trial runs (outer_loop search → inner_loop production)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from bapmos.method.config_utils import dataset_filename
from bapmos.method.data_adapter import repo_root

HPO_SUITES: Dict[str, Dict[str, str]] = {
    "bapmos_bo": {
        "hpo_version": "bapmos_outer_loop",
        "study_metric_tag": "ptv_kfold_sclip",
        "production_version": "bapmos",
        "objective_label": "ptv_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "prostate_bapmos_outer_loop_tpe",
        "selected_root": "experiments/prostate/bapmos/inner_loop",
        "selected_subdir": "selected/tpe",
        "trial_run_root": "runs/prostate/bapmos/outer_loop/tpe",
        "search_method": "tpe",
        "sampler": "tpe",
        "default_n_trials": "100",
        "n_startup_trials": "20",
        "confirmation_trials": "15",
        "eps_abs_physical_mm": "0.001",
        "eps_abs": "0.001",
        "pixel_spacing_mm": "0.159072",
        "objective_units": "val_msd_mm",
    },
    "bapmos_bo_random": {
        "hpo_version": "bapmos_outer_loop_random",
        "study_metric_tag": "ptv_kfold_sclip",
        "production_version": "bapmos",
        "objective_label": "ptv_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "prostate_bapmos_outer_loop_random",
        "selected_root": "experiments/prostate/bapmos/inner_loop",
        "selected_subdir": "selected/random",
        "trial_run_root": "runs/prostate/bapmos/outer_loop/random",
        "search_method": "random",
        "sampler": "random",
        "default_n_trials": "100",
        "n_startup_trials": "20",
        "confirmation_trials": "15",
        "eps_abs_physical_mm": "0.001",
        "eps_abs": "0.001",
        "pixel_spacing_mm": "0.159072",
        "objective_units": "val_msd_mm",
    },
    "bapmos_bo_heuristic": {
        "hpo_version": "bapmos_outer_loop_heuristic",
        "study_metric_tag": "ptv_kfold_sclip",
        "production_version": "bapmos",
        "objective_label": "ptv_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "prostate_bapmos_outer_loop_heuristic",
        "selected_root": "experiments/prostate/bapmos/inner_loop",
        "selected_subdir": "selected/heuristic",
        "trial_run_root": "runs/prostate/bapmos/outer_loop/heuristic",
        "search_method": "heuristic",
        "sampler": "catalog",
        "default_n_trials": "100",
    },
    "bapmos_bo_greedy": {
        "hpo_version": "bapmos_outer_loop_greedy",
        "study_metric_tag": "ptv_kfold_sclip",
        "production_version": "bapmos",
        "objective_label": "ptv_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "prostate_bapmos_outer_loop_greedy",
        "selected_root": "experiments/prostate/bapmos/inner_loop",
        "selected_subdir": "selected/greedy",
        "trial_run_root": "runs/prostate/bapmos/outer_loop/greedy",
        "search_method": "greedy",
        "sampler": "catalog",
        "default_n_trials": "100",
    },
    "bapmos_bo_medsam_pooled": {
        "hpo_version": "bapmos_medsam_pooled_outer_loop",
        "study_metric_tag": "ptv_kfold_sclip",
        "production_version": "bapmos_medsam_pooled",
        "objective_label": "ptv_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "prostate_bapmos_outer_loop_medsam",
        "selected_root": "experiments/prostate/bapmos/inner_loop/medsam",
        "selected_subdir": "selected",
        "trial_run_root": "runs/prostate/bapmos/outer_loop/medsam",
        "search_method": "tpe",
        "sampler": "tpe",
        "default_n_trials": "100",
        "n_startup_trials": "20",
        "confirmation_trials": "15",
        "eps_abs_physical_mm": "0.001",
        "eps_abs": "0.001",
        "pixel_spacing_mm": "0.159072",
        "objective_units": "val_msd_mm",
    },
    "bapmos_bo_sam": {
        "hpo_version": "bapmos_sam_outer_loop",
        "study_metric_tag": "bladder_kfold_sclip",
        "production_version": "bapmos_sam",
        "objective_label": "bladder_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "bladder_bapmos_sam_outer_loop",
        "selected_root": "experiments/bladder/bapmos/inner_loop/sam",
        "selected_subdir": "selected",
        "trial_run_root": "runs/bladder/bapmos/outer_loop/sam",
        "search_method": "tpe",
        "sampler": "tpe",
        "default_n_trials": "100",
        "n_startup_trials": "20",
        "confirmation_trials": "15",
        "eps_abs_physical_mm": "0.001",
        "eps_abs": "0.001",
        "pixel_spacing_mm": "1.0",
        "objective_units": "val_msd_mm",
    },
    "bapmos_bo_medsam": {
        "hpo_version": "bapmos_medsam_outer_loop",
        "study_metric_tag": "bladder_kfold_sclip",
        "production_version": "bapmos_medsam",
        "objective_label": "bladder_kfold_val_msd_mm",
        "studies_root": "optuna_studies",
        "studies_subdir": "bladder_bapmos_medsam_outer_loop",
        "selected_root": "experiments/bladder/bapmos/inner_loop/medsam",
        "selected_subdir": "selected",
        "trial_run_root": "runs/bladder/bapmos/outer_loop/medsam",
        "search_method": "tpe",
        "sampler": "tpe",
        "default_n_trials": "100",
        "n_startup_trials": "20",
        "confirmation_trials": "15",
        "eps_abs_physical_mm": "0.001",
        "eps_abs": "0.001",
        "pixel_spacing_mm": "1.0",
        "objective_units": "val_msd_mm",
    },
}

DEFAULT_HPO_SUITE = "bapmos_bo"


def normalize_hpo_suite(suite: str) -> str:
    key = suite.lower().replace("-", "_")
    if key not in HPO_SUITES:
        choices = ", ".join(sorted(HPO_SUITES))
        raise ValueError(f"Unknown HPO suite {suite!r}; choose from: {choices}")
    return key


@dataclass(frozen=True)
class HpoPaths:
    """Resolved paths for one outer-loop BO suite (``bapmos_bo*``)."""

    suite: str

    @classmethod
    def for_suite(cls, suite: str = DEFAULT_HPO_SUITE) -> "HpoPaths":
        return cls(normalize_hpo_suite(suite))

    @property
    def spec(self) -> Dict[str, str]:
        return HPO_SUITES[self.suite]

    @property
    def hpo_version(self) -> str:
        return self.spec["hpo_version"]

    @property
    def study_metric_tag(self) -> str:
        return self.spec["study_metric_tag"]

    @property
    def production_version(self) -> str:
        return self.spec["production_version"]

    @property
    def objective_label(self) -> str:
        return self.spec["objective_label"]

    def optuna_studies_root(self) -> Path:
        root_rel = self.spec.get("studies_root")
        subdir = self.spec.get("studies_subdir", self.hpo_version)
        if root_rel:
            return repo_root() / root_rel / subdir
        return repo_root() / "optuna_studies" / subdir

    def trial_run_root(self) -> Optional[str]:
        return self.spec.get("trial_run_root")

    def greedy_state_path(self) -> Path:
        return self.optuna_studies_root() / "greedy_state.json"

    def study_db_path(self, dataset: str) -> Path:
        return self.optuna_studies_root() / f"{dataset_filename(dataset)}_{self.study_metric_tag}.db"

    def study_name(self, dataset: str) -> str:
        return f"{self.hpo_version}_{dataset_filename(dataset)}_{self.study_metric_tag}"

    def trial_run_name(self, trial_number: int) -> str:
        method = self.spec.get("search_method", self.suite)
        return f"trial_{method}_{self.study_metric_tag}_{trial_number:04d}"

    def selected_params_path(self, dataset: str) -> Path:
        selected_root = self.spec.get("selected_root")
        selected_subdir = self.spec.get("selected_subdir", "selected")
        if selected_root:
            return (
                repo_root()
                / selected_root
                / selected_subdir
                / f"{dataset_filename(dataset)}.yaml"
            )
        from bapmos.method.config_utils import package_config_dir

        suite = self.production_version.replace("-", "_")
        return (
            package_config_dir()
            / suite
            / "selected"
            / f"{dataset_filename(dataset)}.yaml"
        )

    def selected_manifest_path(self) -> Path:
        selected_root = self.spec.get("selected_root")
        selected_subdir = self.spec.get("selected_subdir", "selected")
        if selected_root:
            return repo_root() / selected_root / selected_subdir / "MANIFEST.json"
        from bapmos.method.config_utils import package_config_dir

        suite = self.production_version.replace("-", "_")
        return package_config_dir() / suite / "selected" / "MANIFEST.json"


# Defaults for the primary prostate TPE outer-loop suite.
_default = HpoPaths.for_suite(DEFAULT_HPO_SUITE)
HPO_VERSION = _default.hpo_version
PRODUCTION_VERSION = _default.production_version
STUDY_METRIC_TAG = _default.study_metric_tag


def optuna_studies_root() -> Path:
    return _default.optuna_studies_root()


def study_db_path(dataset: str) -> Path:
    return _default.study_db_path(dataset)


def study_name(dataset: str) -> str:
    return _default.study_name(dataset)


def trial_run_name(trial_number: int) -> str:
    return _default.trial_run_name(trial_number)


def selected_params_path(dataset: str, *, production_version: str = PRODUCTION_VERSION) -> Path:
    return HpoPaths.for_suite(
        _suite_for_production(production_version)
    ).selected_params_path(dataset)


def selected_manifest_path(*, production_version: str = PRODUCTION_VERSION) -> Path:
    return HpoPaths.for_suite(
        _suite_for_production(production_version)
    ).selected_manifest_path()


def _suite_for_production(production_version: str) -> str:
    """Map production suite id → default HPO suite (prefer TPE / main search method)."""
    key = production_version.lower().replace("-", "_")
    tpe_match: Optional[str] = None
    any_match: Optional[str] = None
    for suite_name, spec in HPO_SUITES.items():
        if spec["production_version"].replace("-", "_") != key:
            continue
        if any_match is None:
            any_match = suite_name
        if spec.get("search_method", "tpe") == "tpe" and tpe_match is None:
            tpe_match = suite_name
    if tpe_match or any_match:
        return tpe_match or any_match  # type: ignore[return-value]
    known = sorted({spec["production_version"] for spec in HPO_SUITES.values()})
    raise ValueError(
        f"No HPO suite maps to production_version={production_version!r}. "
        f"Known production targets: {', '.join(known)}."
    )
