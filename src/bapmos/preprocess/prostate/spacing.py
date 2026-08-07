"""Pixel-spacing contract for the pooled prostate corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from bapmos.training_taxonomy import PIXEL_SPACING_CLINICAL_MM, PIXEL_SPACING_SIMULATION_MM

from bapmos.paths import pooled_prostate_dataset_dir

SPACING_CONTRACT_FILE = "spacing_contract.json"


def pool_pixel_spacing_mm() -> Tuple[float, float]:
    """Spacing (mm/px) for every image/mask under ``data/prostate/pooled/images``."""
    return PIXEL_SPACING_CLINICAL_MM


def spacing_contract_path() -> Path:
    return pooled_prostate_dataset_dir() / SPACING_CONTRACT_FILE


def build_spacing_contract_dict() -> Dict[str, Any]:
    clin = list(PIXEL_SPACING_CLINICAL_MM)
    sim = list(PIXEL_SPACING_SIMULATION_MM)
    scale = sim[0] / clin[0]
    return {
        "pool_pixel_spacing_mm": clin,
        "contract": (
            "Every slice under data/prostate/pooled/images/ is defined at "
            f"pool_pixel_spacing_mm. Simulation sources at {sim[0]} mm/px are "
            f"resampled at build time; clinical case1/case2 slices are already at "
            f"{clin[0]} mm/px (symlinked unchanged). A single spacing value in "
            "training configs is only valid because of that resample."
        ),
        "simulation_resample_scale": scale,
        "cohorts": {
            "simulation": {
                "native_pixel_spacing_mm": sim,
                "effective_pixel_spacing_mm": clin,
                "resampled_at_build": True,
            },
            "case1": {
                "native_pixel_spacing_mm": clin,
                "effective_pixel_spacing_mm": clin,
                "resampled_at_build": False,
            },
            "case2": {
                "native_pixel_spacing_mm": clin,
                "effective_pixel_spacing_mm": clin,
                "resampled_at_build": False,
            },
        },
        "metrics_rule": (
            "MSD/HD95 on pooled train/val/test use pool_pixel_spacing_mm. "
            "Simulation site tests read resampled pooled images, not native "
            f"simulation sources — do not apply {sim[0]} mm/px there."
        ),
    }


def load_spacing_contract() -> Dict[str, Any]:
    path = spacing_contract_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Rebuild with: python -m bapmos.preprocess.prostate"
        )
    return json.loads(path.read_text(encoding="utf-8"))


SITES = ("simulation", "case1", "case2")


def effective_spacing_for_site(site: str) -> Tuple[float, float]:
    """Spacing to use when evaluating a site test list from the pooled bundle."""
    if site not in SITES:
        raise ValueError(f"Unknown site {site!r}")
    contract = load_spacing_contract()
    cohort = contract["cohorts"][site]
    eff = tuple(cohort["effective_pixel_spacing_mm"])
    if len(eff) != 2:
        raise ValueError(f"Invalid effective_pixel_spacing_mm for {site}: {eff}")
    return float(eff[0]), float(eff[1])


def validate_spacing_contract() -> None:
    """Fail fast if the pool was not built with simulation resampling."""
    contract = load_spacing_contract()
    pool = tuple(contract["pool_pixel_spacing_mm"])
    if pool != PIXEL_SPACING_CLINICAL_MM:
        raise ValueError(
            f"spacing_contract pool_pixel_spacing_mm {pool} != "
            f"taxonomy clinical {PIXEL_SPACING_CLINICAL_MM}"
        )
    sim = contract["cohorts"]["simulation"]
    if not sim.get("resampled_at_build"):
        raise ValueError(
            "Simulation cohort in spacing_contract is not marked resampled_at_build. "
            "Cannot use a single clinical spacing for MSD/HD95 on mixed native sim+clinical pixels."
        )
    if tuple(sim["effective_pixel_spacing_mm"]) != PIXEL_SPACING_CLINICAL_MM:
        raise ValueError("Simulation effective spacing must match clinical pool spacing.")
