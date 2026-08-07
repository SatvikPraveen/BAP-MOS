"""
Dataset taxonomy for baseline SAM training (multi-organ and single-organ).

Canonical training corpora live under ``BAPMOS/data/``::

    data/prostate/pooled/          # pooled clinical (4 organs)
    data/prostate/case1|case2/     # single-site clinical
    data/prostate/simulation/      # simulation (3 organs)
    data/bladder/pfus1/            # PFUS1 (8 organs)

Split folders default to ``splits_stratified`` (prostate) or
``splits_patient_70_15_15_seed42`` (PFUS1). Optional parent-layout fallbacks
may also resolve legacy ``preprocessing/`` paths via ``bapmos.paths``.

Background-only *slices* are excluded from stratified splits; class 0 (background
pixels) remains in multiclass masks for every structure-containing slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from bapmos.data.organ_registry import (
    OrganDefinition,
    PFUS1_EIGHT_ORGANS,
    PFUS1_EVAL_CLASS_MAPPING,
    PFUS1_ORGAN_KEYS,
    PFUS1_ORGAN_TO_CLASS,
    REAL_CLINICAL_EVAL_CLASS_MAPPING,
    REAL_CLINICAL_ORGAN_KEYS,
    REAL_CLINICAL_ORGAN_TO_CLASS,
    REAL_CLINICAL_ORGANS,
    SIMULATION_EVAL_CLASS_MAPPING,
    SIMULATION_ORGAN_KEYS,
    SIMULATION_ORGAN_TO_CLASS,
    SIMULATION_THREE_ORGANS,
    pfus1_metrics_organ_list,
    real_clinical_metrics_organ_list,
    simulation_metrics_organ_list,
)
from bapmos.paths import (
    is_pfus1_advanced_training_root,
    is_pfus1_training_root,
    is_pooled_prostate_training_root,
    pfus1_advanced_bundle_dir,
    pfus1_bundle_dir,
    pooled_prostate_dataset_dir,
    project_root,
    real_case_dataset_dir,
    simulation_dataset_dir,
)


# Defaults aligned with ``configs/*/common.pixel_spacing`` (mm/pixel).
PIXEL_SPACING_CLINICAL_MM: Tuple[float, float] = (0.159072, 0.159072)
PIXEL_SPACING_SIMULATION_MM: Tuple[float, float] = (0.14971, 0.14971)
# PFUS1 freehand ultrasound — no DICOM spacing in bundle; use 1.0 mm (pixel-native metrics).
PIXEL_SPACING_PFUS1_MM: Tuple[float, float] = (1.0, 1.0)

# Default split subdirectory under each dataset bundle.
PFUS1_SPLITS_SUBDIR = "splits_patient_70_15_15_seed42"


@dataclass(frozen=True)
class BaselineTaxonomyProfile:
    """Everything a baseline trainer needs to match masks, logits, and metrics."""

    taxonomy_name: str
    is_simulation: bool
    num_classes: int
    organ_keys: Tuple[str, ...]
    organ_to_class: Dict[str, int]
    multiclass_eval_mapping: Dict[int, str]
    evaluator_organ_labels: Tuple[str, ...]
    pixel_spacing_mm: Tuple[float, float]
    data_root_resolved: Path
    organ_definitions: Tuple[OrganDefinition, ...]


def resolve_training_data_root_path(data_root) -> Path:
    p = Path(data_root)
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def is_simulation_training_root(data_root) -> bool:
    """True when ``data_root`` resolves to the canonical simulation bundle."""
    return resolve_training_data_root_path(data_root) == simulation_dataset_dir().resolve()


def is_real_clinical_training_root(data_root) -> bool:
    """True when ``data_root`` resolves to Case 1 or Case 2 preprocessing bundles."""
    resolved = resolve_training_data_root_path(data_root)
    return resolved in (
        real_case_dataset_dir("case1").resolve(),
        real_case_dataset_dir("case2").resolve(),
    )


def default_splits_subdir(data_root) -> str:
    """Default split folder name for a canonical training ``data_root``."""
    if is_pfus1_training_root(data_root) or is_pfus1_advanced_training_root(data_root):
        return PFUS1_SPLITS_SUBDIR
    return "splits_stratified"


def get_baseline_taxonomy_profile(data_root) -> BaselineTaxonomyProfile:
    """
    Return taxonomy for multi-organ / single-organ baseline training.

    Raises ``ValueError`` for unknown roots (no silent default to clinical).
    """
    resolved = resolve_training_data_root_path(data_root)
    if is_pfus1_advanced_training_root(resolved):
        return BaselineTaxonomyProfile(
            taxonomy_name="pfus1_advanced_eight_organ",
            is_simulation=False,
            num_classes=len(PFUS1_ORGAN_TO_CLASS) + 1,
            organ_keys=tuple(PFUS1_ORGAN_KEYS),
            organ_to_class=dict(PFUS1_ORGAN_TO_CLASS),
            multiclass_eval_mapping=dict(PFUS1_EVAL_CLASS_MAPPING),
            evaluator_organ_labels=tuple(pfus1_metrics_organ_list()),
            pixel_spacing_mm=PIXEL_SPACING_PFUS1_MM,
            data_root_resolved=pfus1_advanced_bundle_dir().resolve(),
            organ_definitions=PFUS1_EIGHT_ORGANS,
        )
    if is_pfus1_training_root(resolved):
        return BaselineTaxonomyProfile(
            taxonomy_name="pfus1_eight_organ",
            is_simulation=False,
            num_classes=len(PFUS1_ORGAN_TO_CLASS) + 1,
            organ_keys=tuple(PFUS1_ORGAN_KEYS),
            organ_to_class=dict(PFUS1_ORGAN_TO_CLASS),
            multiclass_eval_mapping=dict(PFUS1_EVAL_CLASS_MAPPING),
            evaluator_organ_labels=tuple(pfus1_metrics_organ_list()),
            pixel_spacing_mm=PIXEL_SPACING_PFUS1_MM,
            data_root_resolved=pfus1_bundle_dir().resolve(),
            organ_definitions=PFUS1_EIGHT_ORGANS,
        )
    if is_simulation_training_root(resolved):
        return BaselineTaxonomyProfile(
            taxonomy_name="simulation_three_organ",
            is_simulation=True,
            num_classes=len(SIMULATION_ORGAN_TO_CLASS) + 1,
            organ_keys=tuple(SIMULATION_ORGAN_KEYS),
            organ_to_class=dict(SIMULATION_ORGAN_TO_CLASS),
            multiclass_eval_mapping=dict(SIMULATION_EVAL_CLASS_MAPPING),
            evaluator_organ_labels=tuple(simulation_metrics_organ_list()),
            pixel_spacing_mm=PIXEL_SPACING_SIMULATION_MM,
            data_root_resolved=resolved,
            organ_definitions=SIMULATION_THREE_ORGANS,
        )
    if is_real_clinical_training_root(resolved):
        return BaselineTaxonomyProfile(
            taxonomy_name="real_clinical_four_organ",
            is_simulation=False,
            num_classes=len(REAL_CLINICAL_ORGAN_TO_CLASS) + 1,
            organ_keys=tuple(REAL_CLINICAL_ORGAN_KEYS),
            organ_to_class=dict(REAL_CLINICAL_ORGAN_TO_CLASS),
            multiclass_eval_mapping=dict(REAL_CLINICAL_EVAL_CLASS_MAPPING),
            evaluator_organ_labels=tuple(real_clinical_metrics_organ_list()),
            pixel_spacing_mm=PIXEL_SPACING_CLINICAL_MM,
            data_root_resolved=resolved,
            organ_definitions=REAL_CLINICAL_ORGANS,
        )
    if is_pooled_prostate_training_root(resolved):
        return BaselineTaxonomyProfile(
            taxonomy_name="real_clinical_four_organ",
            is_simulation=False,
            num_classes=len(REAL_CLINICAL_ORGAN_TO_CLASS) + 1,
            organ_keys=tuple(REAL_CLINICAL_ORGAN_KEYS),
            organ_to_class=dict(REAL_CLINICAL_ORGAN_TO_CLASS),
            multiclass_eval_mapping=dict(REAL_CLINICAL_EVAL_CLASS_MAPPING),
            evaluator_organ_labels=tuple(real_clinical_metrics_organ_list()),
            pixel_spacing_mm=PIXEL_SPACING_CLINICAL_MM,
            data_root_resolved=pooled_prostate_dataset_dir().resolve(),
            organ_definitions=REAL_CLINICAL_ORGANS,
        )
    allowed = (
        simulation_dataset_dir(),
        real_case_dataset_dir("case1"),
        real_case_dataset_dir("case2"),
        pooled_prostate_dataset_dir(),
        pfus1_bundle_dir(),
        pfus1_advanced_bundle_dir(),
    )
    allowed_str = ", ".join(str(p.resolve()) for p in allowed)
    raise ValueError(
        f"Unknown training data_root {data_root!r} (resolved {resolved}). "
        f"Expected one of: {allowed_str}"
    )


def log_baseline_taxonomy_startup(profile: BaselineTaxonomyProfile, *, prefix: str = "") -> None:
    """Single structured print line for run logs (stdout)."""
    pfx = f"{prefix} " if prefix else ""
    print(
        f"{pfx}taxonomy={profile.taxonomy_name} | data_root={profile.data_root_resolved} | "
        f"num_classes={profile.num_classes} | organ_to_class={profile.organ_to_class} | "
        f"pixel_spacing_mm={profile.pixel_spacing_mm} | evaluator_organs={list(profile.evaluator_organ_labels)}"
    )
