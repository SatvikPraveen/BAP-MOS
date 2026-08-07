"""Path helpers for BAP-MOS method runs and flat test output under ``output/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from bapmos.method.data_adapter import repo_root


def _project_root() -> Path:
    return repo_root()


def runs_bundle_from_data_root(data_root: str | Path) -> str:
    """Map preprocessing path to ``runs/<bundle>/`` slug (case_1, case_2, simulation)."""
    from bapmos.paths import dataset_bundle_tag

    return dataset_bundle_tag(str(data_root))


def runs_bundle_from_config(cfg: Dict[str, Any]) -> str:
    common = cfg.get("common", cfg)
    return runs_bundle_from_data_root(common["data_root"])


def optimization_strategy_dir(cfg: Dict[str, Any]) -> str:
    """Subdirectory under ``runs/<bundle>/Optimization/`` for this ablation."""
    ablation_id = str(cfg.get("ablation", {}).get("id", "bapmos"))
    if ablation_id.startswith("bapmos_"):
        return ablation_id
    return f"bapmos_{ablation_id}"


def run_directory_layout(cfg: Dict[str, Any]) -> str:
    """
    Checkpoint parent (timestamped run name appended by trainer)::

        runs/<bundle>/Optimization/bapmos_<ablation_id>/
    """
    bundle = runs_bundle_from_config(cfg)
    strategy = optimization_strategy_dir(cfg)
    return f"runs/{bundle}/Optimization/{strategy}"


def method_slug_from_config(cfg: Dict[str, Any]) -> str:
    """
    Flat folder name under ``output/<dataset-layout>/``.

    Examples: ``bapmos_production_seed42``, ``bapmos_pooled_seed43``.
    """
    version = str(cfg.get("ablation", {}).get("version", "vx"))
    exp = str(cfg.get("experiment_name", "default_seed42"))
    suffix = exp.replace("_seed42", "")
    if suffix in ("default", ""):
        return f"bapmos_{version}"
    return f"bapmos_{version}_{suffix}"


def bapmos_output_layout_root(data_root: str | Path) -> Path:
    """
    Method test-export root::

        inference_output/prostate/pooled/
        inference_output/bladder/pfus1/
        output/simulation/   (legacy site corpora)
    """
    from bapmos.paths import inference_output_layout_root

    return inference_output_layout_root(data_root, repo_root=_project_root()).resolve()


def bapmos_method_test_output_dir(
    data_root: str | Path,
    method_slug: str,
) -> Path:
    """``output/<layout>/<method_slug>/``."""
    slug = method_slug.strip().replace("-", "_")
    if not slug:
        raise ValueError("method_slug must be non-empty")
    return (bapmos_output_layout_root(data_root) / slug).resolve()


def bapmos_method_test_output_dir_from_config(cfg: Dict[str, Any]) -> Path:
    common = cfg.get("common", cfg)
    return bapmos_method_test_output_dir(
        common["data_root"],
        method_slug_from_config(cfg),
    )


def write_bapmos_evaluation_meta(
    output_dir: Path,
    *,
    checkpoint: str | Path,
    data_root: str | Path,
    method_slug: str,
    cfg: Optional[Dict[str, Any]] = None,
    split: str = "test",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``evaluation_meta.json`` at the method output root."""
    import json
    from datetime import datetime, timezone

    from bapmos.paths import resolve_under_project

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = resolve_under_project(checkpoint)
    meta: Dict[str, Any] = {
        "method_slug": method_slug,
        "split": split,
        "checkpoint": str(ckpt.resolve()),
        "data_root": str(resolve_under_project(data_root).resolve()),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if cfg:
        meta["ablation_id"] = cfg.get("ablation", {}).get("id")
        meta["ablation_version"] = cfg.get("ablation", {}).get("version")
        meta["experiment_name"] = cfg.get("experiment_name")
        meta["experiment_seed"] = cfg.get("experiment_seed")
    if extra:
        meta.update(extra)
    path = output_dir / "evaluation_meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path
