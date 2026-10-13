"""Public inference-output protocol (seed isolation + panel export).

Rules
-----
1. Training checkpoints live under isolated ``runs/.../<run_name>/`` folders
   (one folder per seed: ``*_seed42``, ``*_seed43_rep2``, ``*_seed44_rep3``).
2. Stratified ``inference_output/`` (and canonical flat ``output/`` mirrors) are
   written for the **primary seed-42 run only**.
3. Overlay / difference panels: PNG and PDF are separate matplotlib ``savefig``
   calls from in-memory RGB (never PNG→PDF wrap), both at ``PDF_EXPORT_DPI`` (350).
4. Disk images (prediction PNGs + overlay / difference PNG+PDF) for the **first**
   ``DEFAULT_VIZ_PDF_MAX`` (10) evaluated slices only — same stems across methods
   when loaders share order. Metrics still cover the full test split.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from bapmos.pdf_export import PDF_EXPORT_DPI

# Primary training / reporting seed (docs/SEEDS.md).
PRIMARY_INFERENCE_SEED = 42

# First N evaluated slices get prediction PNGs + matplotlib panel PNG/PDF siblings.
DEFAULT_VIZ_PDF_MAX = 10

# Uniform panel DPI (PNG + PDF).
PANEL_EXPORT_DPI = PDF_EXPORT_DPI

_REP_SUFFIX = re.compile(r"_rep\d+$")
_SEED_EMBEDDED = re.compile(r"seed(\d+)")


def is_primary_inference_seed_run(
    *,
    experiment_seed: Optional[int] = None,
    run_name: Optional[str] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Return True only for the primary seed-42 run (not ``*_rep2`` / ``*_rep3``).

    Requires training seed **42**. If both ``experiment_seed`` and an embedded
    ``seedN`` in ``run_name`` are present, they must agree (no ``seed42`` name
    with training seed 43).
    """
    if cfg is not None:
        if experiment_seed is None and cfg.get("experiment_seed") is not None:
            experiment_seed = int(cfg["experiment_seed"])
        if run_name is None:
            run_name = (
                cfg.get("run_name")
                or cfg.get("experiment_name")
                or (cfg.get("common") or {}).get("run_name")
            )

    embedded: Optional[int] = None
    if run_name:
        name = str(run_name).strip()
        if _REP_SUFFIX.search(name):
            return False
        m = _SEED_EMBEDDED.search(name)
        if m:
            embedded = int(m.group(1))

    if experiment_seed is not None:
        seed = int(experiment_seed)
    elif embedded is not None:
        seed = embedded
    else:
        return False

    if seed != int(PRIMARY_INFERENCE_SEED):
        return False
    # Name embeds a different seed than the primary → reject.
    if embedded is not None and embedded != int(PRIMARY_INFERENCE_SEED):
        return False
    return True


def require_primary_inference_seed_run(
    *,
    experiment_seed: Optional[int] = None,
    run_name: Optional[str] = None,
    cfg: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> None:
    """Raise unless this is the primary seed-42 run (or ``force``)."""
    if force:
        return
    if is_primary_inference_seed_run(
        experiment_seed=experiment_seed, run_name=run_name, cfg=cfg
    ):
        return
    raise ValueError(
        "inference_output / stratified test export is restricted to the primary "
        f"seed-{PRIMARY_INFERENCE_SEED} run "
        "(e.g. pooled_seed42 / pfus1_seed42) with matching experiment_seed. "
        "Replicate runs (*_rep2 / *_rep3) keep isolated checkpoints under runs/ only. "
        "Pass force=True / --force-inference-output to override."
    )


def export_viz_protocol_kwargs(
    *,
    viz_pdf_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Kwargs for ``export_stratified_test_bundle``: first-N images + PDF, DPI=350.

    ``viz_pdf_max`` (default :data:`DEFAULT_VIZ_PDF_MAX`) caps prediction PNGs,
    overlay/difference PNGs, and PDFs to the same first-N loader-order slices.
    Pass a larger value (or ``0`` via CLI only if wired) when you need more panels.
    """
    n = DEFAULT_VIZ_PDF_MAX if viz_pdf_max is None else int(viz_pdf_max)
    return {
        "save_visualizations": True,
        "save_viz_pdf": True,
        "viz_selection": "all",
        "viz_max": None,
        "viz_per_patient_max": None,
        "viz_pdf_max": n,
        "save_prediction_images_max": n,
        "viz_pdf_dpi": PANEL_EXPORT_DPI,
    }
