"""
Compose BAP-MOS training configs: version × dataset × experiment overrides.

Preferred (production / paper path)::

    python -m bapmos.method.bap_mos_trainer \\
      --version bapmos --dataset pooled --experiment pooled_seed42

``--version`` + ``--dataset`` merges ``configs/common/protocol.yaml``, site
``common.yaml``, suite ``version.yaml``, and ``selected/`` when
``parameter_source: bayesian_selected``.

Advanced (as-is YAML only; does **not** merge site common or ``selected/``)::

    python -m bapmos.method.bap_mos_trainer --config path/to/fully_specified.yaml

See ``docs/INNER_OUTER_LOOP.md``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from bapmos.method.data_adapter import load_yaml_config, repo_root


def package_config_dir() -> Path:
    """Shared YAML under ``configs/`` (protocol, site commons, optional suite trees)."""
    return repo_root() / "configs"


def prostate_bapmos_experiments_dir() -> Path:
    return repo_root() / "experiments" / "prostate" / "bapmos"


def bladder_bapmos_experiments_dir() -> Path:
    return repo_root() / "experiments" / "bladder" / "bapmos"


# Legacy path aliases → experiments/.
def prostate_investigation_config_dir() -> Path:
    return prostate_bapmos_experiments_dir()


def bladder_pop_investigation_config_dir() -> Path:
    return bladder_bapmos_experiments_dir()


# Suite roots outside ``configs/<suite>/`` (live under ``experiments/``).
# outer_loop = HPO / search trial training; inner_loop = production with selected HPs.
EXTERNAL_SUITE_DIRS: Dict[str, Path] = {
    "bapmos_outer_loop": prostate_bapmos_experiments_dir() / "outer_loop" / "tpe",
    "bapmos_outer_loop_random": prostate_bapmos_experiments_dir() / "outer_loop" / "random",
    "bapmos_outer_loop_greedy": prostate_bapmos_experiments_dir() / "outer_loop" / "greedy",
    "bapmos_outer_loop_heuristic": prostate_bapmos_experiments_dir() / "outer_loop" / "heuristic",
    "bapmos_medsam_pooled_outer_loop": prostate_bapmos_experiments_dir() / "outer_loop" / "medsam",
    "bapmos": prostate_bapmos_experiments_dir() / "inner_loop",
    "bapmos_medsam_pooled": prostate_bapmos_experiments_dir() / "inner_loop" / "medsam",
    "bapmos_sam_outer_loop": bladder_bapmos_experiments_dir() / "outer_loop" / "sam",
    "bapmos_sam": bladder_bapmos_experiments_dir() / "inner_loop" / "sam",
    "bapmos_medsam_outer_loop": bladder_bapmos_experiments_dir() / "outer_loop" / "medsam",
    "bapmos_medsam": bladder_bapmos_experiments_dir() / "inner_loop" / "medsam",
}


# CLI version key → suite dir under experiments/ (see EXTERNAL_SUITE_DIRS) or configs/
SUITE_BY_VERSION: Dict[str, str] = {
    "bapmos_outer_loop": "bapmos_outer_loop",
    "bapmos_outer_loop_random": "bapmos_outer_loop_random",
    "bapmos_outer_loop_greedy": "bapmos_outer_loop_greedy",
    "bapmos_outer_loop_heuristic": "bapmos_outer_loop_heuristic",
    "bapmos_medsam_pooled_outer_loop": "bapmos_medsam_pooled_outer_loop",
    "bapmos": "bapmos",
    "bapmos_medsam_pooled": "bapmos_medsam_pooled",
    "bapmos_sam_outer_loop": "bapmos_sam_outer_loop",
    "bapmos_sam": "bapmos_sam",
    "bapmos_medsam_outer_loop": "bapmos_medsam_outer_loop",
    "bapmos_medsam": "bapmos_medsam",
}

# Removed pre-rename keys — do not silently remap (public API must stay unambiguous).
REMOVED_VERSION_KEYS: Dict[str, str] = {
    "bapmos_inner_loop": (
        "Removed: --version bapmos_inner_loop. "
        "Use --version bapmos_outer_loop for prostate search (TPE), "
        "or --version bapmos for production."
    ),
    "bapmos_sam_inner_loop": (
        "Removed: --version bapmos_sam_inner_loop. "
        "Use --version bapmos_sam_outer_loop for bladder SAM search, "
        "or --version bapmos_sam for production."
    ),
    "bapmos_medsam_inner_loop": (
        "Removed: --version bapmos_medsam_inner_loop. "
        "Use --version bapmos_medsam_outer_loop for bladder MedSAM search, "
        "or --version bapmos_medsam for production."
    ),
    "bapmos_kervadec": (
        "Removed: --version bapmos_kervadec. "
        "Use --version bapmos (same prostate production suite)."
    ),
}

# Suites with optional per-dataset method overrides under suite/datasets/
SUITE_DATASET_OVERRIDE_DIRS: frozenset[str] = frozenset()

# Production (inner_loop) suites: per-dataset selected hyperparameters under selected/
SUITES_WITH_SELECTED_PARAMS = frozenset(
    {
        "bapmos",
        "bapmos_medsam_pooled",
        "bapmos_sam",
        "bapmos_medsam",
    }
)

# Default ``selected/`` folder under each suite root when ``BAPMOS_SELECTED_SUBDIR`` unset.
# Prostate method ablations live under ``selected/<method>/``; main paper = TPE.
SUITE_DEFAULT_SELECTED_SUBDIR: Dict[str, str] = {
    "bapmos": "selected/tpe",
    "bapmos_medsam_pooled": "selected",
    "bapmos_sam": "selected",
    "bapmos_medsam": "selected",
}

# Prostate SAM search-method ablations that share the ``bapmos`` suite but must not
# collide under the same ``runs/.../inner_loop/pooled_seed*`` folder.
PROSTATE_SAM_SEARCH_METHODS = frozenset({"tpe", "random", "greedy", "heuristic"})

# Env: set to 1/true to compose/train against generation-0 placeholder selected YAMLs.
ALLOW_PLACEHOLDER_SELECTED_ENV = "BAPMOS_ALLOW_PLACEHOLDER_SELECTED"


def normalize_version_key(version: str) -> str:
    key = version.lower().replace("-", "_")
    if key.startswith("bapmos"):
        return key
    if not key.startswith("v"):
        key = f"v{key}"
    return key


def reject_removed_version_key(version: str) -> None:
    """Raise if *version* is a removed pre-rename CLI key."""
    msg = REMOVED_VERSION_KEYS.get(normalize_version_key(version))
    if msg:
        raise ValueError(msg)


def suite_dir_for_version(version: str) -> Optional[str]:
    reject_removed_version_key(version)
    return SUITE_BY_VERSION.get(normalize_version_key(version))


def is_suite_version(version: str) -> bool:
    try:
        return suite_dir_for_version(version) is not None
    except ValueError:
        return False


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_yaml_path(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def dataset_filename(dataset: str) -> str:
    key = dataset.lower().replace("-", "").replace("_", "")
    aliases = {
        "simulation": "simulation",
        "sim": "simulation",
        "case1": "case1",
        "case_1": "case1",
        "case2": "case2",
        "case_2": "case2",
        "pfus1advanced": "pfus1_advanced",
        "pfus1_advanced": "pfus1_advanced",
        "pfus1": "pfus1",
        "pooled": "pooled",
        "prostatepooled": "pooled",
        "prostate_pooled": "pooled",
    }
    fname = aliases.get(key)
    if fname is None:
        raise ValueError(
            f"Unknown dataset {dataset!r}; use simulation, case1, case2, pfus1, pfus1_advanced, or pooled"
        )
    return fname


def _dataset_fragment_from_site_common(
    common_path: Path,
    *,
    dataset_name: str,
    data_root: str,
) -> Dict[str, Any]:
    """Wrap flat ``configs/<site>/common.yaml`` keys into a compose-able fragment."""
    raw = _load_yaml_path(common_path)
    common = {k: v for k, v in raw.items() if k not in ("wandb_project",)}
    common.setdefault("data_root", data_root)
    try:
        from bapmos.train.training_taxonomy import get_baseline_taxonomy_profile

        organs = list(get_baseline_taxonomy_profile(common["data_root"]).organ_keys)
    except Exception:
        organs = list(raw.get("organs", []))
    frag: Dict[str, Any] = {
        "common": common,
        "dataset": {"name": dataset_name},
        "bandit": {"organs": organs},
    }
    if "wandb_project" in raw:
        frag["wandb"] = {"project": raw["wandb_project"]}
    return frag


def load_dataset_fragment(dataset: str) -> Dict[str, Any]:
    """Load dataset YAML from experiments/configs trees or site ``common.yaml`` fallbacks."""
    fname = dataset_filename(dataset)
    candidates = [
        prostate_bapmos_experiments_dir() / "datasets" / f"{fname}.yaml",
        bladder_bapmos_experiments_dir() / "datasets" / f"{fname}.yaml",
        package_config_dir() / "prostate" / "datasets" / f"{fname}.yaml",
        package_config_dir() / "bladder" / "datasets" / f"{fname}.yaml",
        package_config_dir() / "datasets" / f"{fname}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return _load_yaml_path(path)

    # Site commons (migrated layout) when dedicated dataset YAMLs are absent.
    if fname == "pooled":
        common = package_config_dir() / "prostate" / "common.yaml"
        if common.is_file():
            return _dataset_fragment_from_site_common(
                common,
                dataset_name="pooled",
                data_root="data/prostate/pooled",
            )
    if fname == "pfus1":
        common = package_config_dir() / "bladder" / "common.yaml"
        if common.is_file():
            return _dataset_fragment_from_site_common(
                common,
                dataset_name="pfus1",
                data_root="data/bladder/pfus1",
            )
    if fname == "pfus1_advanced":
        common = package_config_dir() / "bladder" / "common.yaml"
        if common.is_file():
            return _dataset_fragment_from_site_common(
                common,
                dataset_name="pfus1_advanced",
                data_root="data/bladder/pfus1_advanced",
            )

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Dataset fragment {dataset!r} (file {fname}.yaml) not found. Searched:\n  - {searched}"
    )


def load_suite_version_fragment(suite_dir: str) -> Dict[str, Any]:
    """Load ``version.yaml`` from an external suite root or ``configs/<suite_dir>/``."""
    if suite_dir in EXTERNAL_SUITE_DIRS:
        path = EXTERNAL_SUITE_DIRS[suite_dir] / "version.yaml"
    else:
        path = package_config_dir() / suite_dir / "version.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_yaml_path(path)


def load_suite_dataset_fragment(suite_dir: str, dataset: str) -> Optional[Dict[str, Any]]:
    """Optional per-dataset overrides under suite ``datasets/`` (package or external)."""
    fname = f"{dataset_filename(dataset)}.yaml"
    candidates: List[Path] = []
    if suite_dir in EXTERNAL_SUITE_DIRS:
        candidates.append(EXTERNAL_SUITE_DIRS[suite_dir] / "datasets" / fname)
    candidates.append(package_config_dir() / suite_dir / "datasets" / fname)
    for path in candidates:
        if path.is_file():
            return _load_yaml_path(path)
    return None


def normalize_selected_subdir(subdir: str) -> str:
    """
    Normalize ``BAPMOS_SELECTED_SUBDIR`` / CLI aliases.

    Accepts bare search-method names (``tpe``, ``random``, …) and expands them to
    ``selected/<method>`` so operators are not forced to remember the prefix.
    """
    s = (subdir or "").strip().strip("/")
    if not s:
        return s
    if s in PROSTATE_SAM_SEARCH_METHODS:
        return f"selected/{s}"
    return s


def resolve_selected_subdir(suite_dir: str) -> str:
    """Resolved selected/ subdir for *suite_dir* (env override or suite default)."""
    import os

    env_subdir = (os.environ.get("BAPMOS_SELECTED_SUBDIR") or "").strip()
    default_subdir = SUITE_DEFAULT_SELECTED_SUBDIR.get(suite_dir, "selected")
    return normalize_selected_subdir(env_subdir) or default_subdir


def selected_hyperparameters_path(suite_dir: str, dataset: str) -> Path:
    """Exact path compose expects for ``selected/<dataset>.yaml`` (no fallback)."""
    fname = f"{dataset_filename(dataset)}.yaml"
    selected_subdir = resolve_selected_subdir(suite_dir)
    if suite_dir in EXTERNAL_SUITE_DIRS:
        root = EXTERNAL_SUITE_DIRS[suite_dir]
    else:
        root = package_config_dir() / suite_dir
    return root / selected_subdir / fname


def load_selected_hyperparameters(suite_dir: str, dataset: str) -> Optional[Dict[str, Any]]:
    """Load BO-selected overrides from the exact selected path (no silent fallback)."""
    path = selected_hyperparameters_path(suite_dir, dataset)
    if not path.is_file():
        return None
    return _load_yaml_path(path)


def parameter_source(cfg: Dict[str, Any]) -> Optional[str]:
    """Return ablation.parameter_source when set (e.g. ``bayesian_selected``)."""
    raw = cfg.get("ablation", {}).get("parameter_source")
    return str(raw) if raw else None


def _placeholder_selected_allowed() -> bool:
    import os

    return (os.environ.get(ALLOW_PLACEHOLDER_SELECTED_ENV) or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def selection_generation(meta_block: Optional[Dict[str, Any]]) -> int:
    """Return ``selection_meta.generation`` (missing → 0 = placeholder)."""
    if not isinstance(meta_block, dict):
        return 0
    raw = meta_block.get("generation", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def search_method_from_selected_subdir(selected_subdir: str) -> Optional[str]:
    """``selected/tpe`` → ``tpe``; bare ``tpe`` → ``tpe``; ``selected`` → None."""
    parts = normalize_selected_subdir(selected_subdir).strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "selected" and parts[1] in PROSTATE_SAM_SEARCH_METHODS:
        return parts[1]
    if len(parts) == 1 and parts[0] in PROSTATE_SAM_SEARCH_METHODS:
        return parts[0]
    return None


def apply_prostate_sam_inner_run_root(
    cfg: Dict[str, Any],
    *,
    suite_dir: str,
    selected_subdir: str,
    meta_block: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Isolate prostate SAM production runs by search method.

    Without this, ``VERSION=bapmos`` + different ``SELECTED_SUBDIR`` values would all
    write ``runs/prostate/bapmos/inner_loop/pooled_seed*`` and overwrite each other.
    """
    if suite_dir != "bapmos":
        return cfg
    method: Optional[str] = None
    if isinstance(meta_block, dict):
        raw = meta_block.get("search_method")
        if raw is not None and str(raw).strip() in PROSTATE_SAM_SEARCH_METHODS:
            method = str(raw).strip()
    if method is None:
        method = search_method_from_selected_subdir(selected_subdir)
    if method is None:
        method = "tpe"
    cfg["run_root"] = f"runs/prostate/bapmos/inner_loop/{method}"
    cfg.setdefault("meta", {})
    cfg["meta"]["inner_search_method"] = method
    return cfg


def apply_selected_hyperparameters(
    cfg: Dict[str, Any],
    suite_dir: str,
    dataset: str,
) -> Dict[str, Any]:
    """
    Merge ``selected/<dataset>.yaml`` when the suite uses Bayesian-selected params.

    ``selection_meta`` is kept on ``cfg['meta']['selection']`` for traceability.

    Refuses ``generation: 0`` (placeholder) selected files unless
    ``BAPMOS_ALLOW_PLACEHOLDER_SELECTED=1``.
    """
    if suite_dir not in SUITES_WITH_SELECTED_PARAMS:
        return cfg
    if parameter_source(cfg) != "bayesian_selected":
        return cfg

    selected_subdir = resolve_selected_subdir(suite_dir)
    path = selected_hyperparameters_path(suite_dir, dataset)
    selected = load_selected_hyperparameters(suite_dir, dataset)
    if not selected:
        raise FileNotFoundError(
            f"bayesian_selected requires hyperparameters at {path}. "
            "Export outer-loop winners first, or fix BAPMOS_SELECTED_SUBDIR "
            "(no silent fallback to selected/). "
            "Bare names like 'tpe' are accepted and normalized to 'selected/tpe'."
        )

    meta_block = selected.get("selection_meta")
    gen = selection_generation(meta_block if isinstance(meta_block, dict) else None)
    if gen <= 0 and not _placeholder_selected_allowed():
        raise RuntimeError(
            f"Refusing placeholder selected hyperparameters at {path} "
            f"(selection_meta.generation={gen}). "
            "Run outer-loop export first (export writes generation>=1), or set "
            f"{ALLOW_PLACEHOLDER_SELECTED_ENV}=1 only for intentional dry-runs."
        )

    hp = {k: v for k, v in selected.items() if k != "selection_meta"}
    out = deep_merge(cfg, hp)
    if meta_block:
        out.setdefault("meta", {})
        out["meta"]["selection"] = copy.deepcopy(meta_block)
    out = apply_prostate_sam_inner_run_root(
        out,
        suite_dir=suite_dir,
        selected_subdir=selected_subdir,
        meta_block=meta_block if isinstance(meta_block, dict) else None,
    )
    return out


def load_version_fragment(version: str) -> Dict[str, Any]:
    """Load ``configs/versions/<version>.yaml`` (optional local fragments).

    Prefer suite keys (``bapmos``, ``bapmos_outer_loop``, …). Standalone BAPMOS
    ships suite trees under ``experiments/``, not optional ``configs/versions/``.
    """
    key = version.lower().replace("-", "_")
    path = package_config_dir() / "versions" / f"{key}.yaml"
    if not path.is_file():
        known = ", ".join(sorted(SUITE_BY_VERSION))
        raise FileNotFoundError(
            f"No config fragment for version={version!r} at {path}. "
            f"Use a suite key ({known}) or --config <path/to/version.yaml>."
        )
    return _load_yaml_path(path)


def load_protocol_defaults() -> Dict[str, Any]:
    """Floor defaults from ``configs/common/protocol.yaml`` (lowest merge priority)."""
    path = package_config_dir() / "common" / "protocol.yaml"
    if not path.is_file():
        return {}
    raw = _load_yaml_path(path)
    floor: Dict[str, Any] = {}
    if isinstance(raw.get("reward"), dict):
        floor["reward"] = copy.deepcopy(raw["reward"])
    if isinstance(raw.get("training"), dict):
        floor["training"] = copy.deepcopy(raw["training"])
    # Map checkpoint_objective → evaluation.* used by the trainer / compose YAMLs.
    co = raw.get("checkpoint_objective") or {}
    eval_floor: Dict[str, Any] = {}
    if co.get("metric"):
        eval_floor["objective_metric"] = co["metric"]
    if co.get("kfold_n") is not None:
        eval_floor["kfold_n"] = int(co["kfold_n"])
    if co.get("kfold_seed") is not None:
        eval_floor["kfold_seed"] = int(co["kfold_seed"])
    if eval_floor:
        floor["evaluation"] = eval_floor
    return floor


def compose_config(version: str, dataset: str) -> Dict[str, Any]:
    """Merge protocol → site → suite ``version.yaml`` (later layers win on conflicts)."""
    protocol = load_protocol_defaults()
    suite_dir = suite_dir_for_version(version)
    if suite_dir is not None:
        ver = load_suite_version_fragment(suite_dir)
        ds = load_dataset_fragment(dataset)
        # Protocol floor, then site common, then suite version (backbone / schedule).
        cfg = deep_merge(protocol, deep_merge(ds, ver))
        if suite_dir in SUITE_DATASET_OVERRIDE_DIRS:
            override = load_suite_dataset_fragment(suite_dir, dataset)
            if override:
                cfg = deep_merge(cfg, override)
        cfg = apply_selected_hyperparameters(cfg, suite_dir, dataset)
        cfg.setdefault("meta", {})
        cfg["meta"]["version"] = str(ver.get("ablation", {}).get("version", version))
        cfg["meta"]["dataset"] = ds.get("dataset", {}).get("name", dataset)
        cfg["meta"]["config_suite"] = suite_dir
        cfg["meta"]["protocol_merged"] = bool(protocol)
        return cfg

    ver = load_version_fragment(version)
    ds = load_dataset_fragment(dataset)
    cfg = deep_merge(protocol, deep_merge(ds, ver))
    cfg.setdefault("meta", {})
    cfg["meta"]["version"] = str(ver.get("ablation", {}).get("version", version))
    cfg["meta"]["dataset"] = ds.get("dataset", {}).get("name", dataset)
    cfg["meta"]["protocol_merged"] = bool(protocol)
    return cfg


def list_experiments(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    exps = cfg.get("experiments", [])
    if not exps:
        return [{"name": "default_seed42", "seed": 42}]
    return list(exps)


def enrich_wandb_run_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach W&B run tags and group after ``resolve_experiment``.

    Inner / production suites get tag ``production``. Outer-loop / search suites
    get ``hpo`` (never ``production``) so dashboards stay separable.
    """
    wandb_cfg = cfg.setdefault("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return cfg

    ablation_id = str(cfg.get("ablation", {}).get("id", "unknown"))
    dataset = str(
        cfg.get("meta", {}).get("dataset") or cfg.get("dataset", {}).get("name", "unknown")
    )
    experiment = str(cfg.get("experiment_name", "run"))

    tags = list(wandb_cfg.get("tags", []))
    # Drop opposite loop tags then add the correct one.
    if _cfg_is_outer_loop(cfg):
        tags = [t for t in tags if t != "production"]
        loop_tag = "hpo"
        if "outer_loop" not in tags:
            tags.append("outer_loop")
    else:
        tags = [t for t in tags if t != "hpo"]
        loop_tag = "production"
    for tag in (f"dataset={dataset}", f"experiment={experiment}", loop_tag):
        if tag not in tags:
            tags.append(tag)
    wandb_cfg["tags"] = list(dict.fromkeys(tags))

    if not wandb_cfg.get("group"):
        wandb_cfg["group"] = f"{ablation_id}_{dataset}"

    return cfg


def _cfg_is_outer_loop(cfg: Dict[str, Any]) -> bool:
    """True for Optuna search / outer_loop configs (not production inner_loop)."""
    suite = str(cfg.get("meta", {}).get("config_suite") or "").lower()
    version = str(cfg.get("ablation", {}).get("version") or "").lower()
    run_root = str(cfg.get("run_root") or "").lower()
    tags = [str(t).lower() for t in (cfg.get("wandb", {}).get("tags") or [])]
    if "outer_loop" in suite or "outer_loop" in version or "outer_loop" in run_root:
        return True
    if "outer_loop" in tags:
        return True
    return False


def resolve_experiment(
    cfg: Dict[str, Any],
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a fully merged config for one experiment (applies ``overrides`` block).

    Sets ``experiment_name`` and ``experiment_seed`` on the returned dict.
    """
    experiments = list_experiments(cfg)
    if experiment_name is None:
        exp = experiments[0]
    else:
        matches = [e for e in experiments if e.get("name") == experiment_name]
        if not matches:
            names = [e.get("name") for e in experiments]
            raise ValueError(f"Unknown experiment {experiment_name!r}; available: {names}")
        exp = matches[0]

    resolved = copy.deepcopy(cfg)
    overrides = exp.get("overrides", {})
    if overrides:
        resolved = deep_merge(resolved, overrides)

    resolved["experiment_name"] = str(exp.get("name", "default_seed42"))
    resolved["experiment_seed"] = int(exp.get("seed", 42))
    return enrich_wandb_run_metadata(resolved)


def run_directory_layout(cfg: Dict[str, Any]) -> str:
    """
    Legacy-style checkpoint parent under ``runs/<bundle>/Optimization/bapmos_<ablation_id>/``.

    Implemented in :mod:`bapmos.method.paths`.
    """
    from bapmos.method.paths import run_directory_layout as _layout

    return _layout(cfg)


def load_composed_config(
    *,
    version: Optional[str] = None,
    dataset: Optional[str] = None,
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load config from (--version + --dataset) or --config file, then apply experiment."""
    if config_path:
        cfg = load_yaml_config(config_path)
        if version or dataset:
            raise ValueError("Use either --config alone or --version with --dataset, not both")
    else:
        if not version or not dataset:
            raise ValueError("Both --version and --dataset are required when --config is omitted")
        cfg = compose_config(version, dataset)
    return resolve_experiment(cfg, experiment_name)


def load_config_path(path: str) -> Dict[str, Any]:
    """Load a single YAML from repo-relative or absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return _load_yaml_path(p)
