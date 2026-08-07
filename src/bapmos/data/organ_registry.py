"""
Single source of truth for organ definitions (keys, class IDs, evaluator labels, colors).

Add a new clinical organ by appending one OrganDefinition to REAL_CLINICAL_ORGANS
and updating training heads / num_classes if needed.

Simulation (3 foreground classes, different ID order) lives in SIMULATION_THREE_ORGANS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class OrganDefinition:
    """One foreground structure in a multi-class mask."""

    key: str
    """Snake_case key used in code and YAML (e.g. 'bladder', 'ptv1')."""

    class_id: int
    """Integer label in combined multiclass PNG / logits channel index (1..K)."""

    evaluator_label: str
    """String stored in MetricsEvaluator CSVs (e.g. 'Bladder', 'PTV1')."""

    color_bgr: Tuple[int, int, int]
    """BGR triplet for OpenCV overlays."""


# --- Real clinical prostate-planning dataset (pooled / case1 / case2) ---
# Class IDs must stay stable for existing masks on disk.

REAL_CLINICAL_ORGANS: Tuple[OrganDefinition, ...] = (
    OrganDefinition("bladder", 1, "Bladder", (0, 255, 255)),
    OrganDefinition("ptv", 2, "PTV", (0, 255, 0)),
    OrganDefinition("rectum", 3, "Rectum", (0, 0, 255)),
    OrganDefinition("urethra", 4, "Urethra", (255, 0, 0)),
)

# --- Simulation / optimization pipeline (Rectum=1, Bladder=2, PTV1=3) ---

SIMULATION_THREE_ORGANS: Tuple[OrganDefinition, ...] = (
    OrganDefinition("rectum", 1, "Rectum", (0, 0, 255)),
    OrganDefinition("bladder", 2, "Bladder", (0, 255, 0)),
    OrganDefinition("ptv1", 3, "PTV1", (255, 0, 0)),
)

# --- PFUS1 pelvic-floor ultrasound (8 foreground classes; ids match combined masks) ---

PFUS1_EIGHT_ORGANS: Tuple[OrganDefinition, ...] = (
    OrganDefinition("pubis", 1, "Pubis", (180, 120, 80)),
    OrganDefinition("urethra", 2, "Urethra", (0, 255, 255)),
    OrganDefinition("bladder", 3, "Bladder", (0, 165, 255)),
    OrganDefinition("uterus", 4, "Uterus", (147, 20, 255)),
    OrganDefinition("vagina", 5, "Vagina", (255, 0, 180)),
    OrganDefinition("anus", 6, "Anus", (60, 76, 40)),
    OrganDefinition("rectum", 7, "Rectum", (255, 0, 0)),
    OrganDefinition("levator_ani", 8, "Levator ani muscle", (0, 255, 0)),
)


def _keys(organs: Tuple[OrganDefinition, ...]) -> List[str]:
    return [o.key for o in organs]


def _to_class_map(organs: Tuple[OrganDefinition, ...]) -> Dict[str, int]:
    return {o.key: o.class_id for o in organs}


def _eval_class_mapping(organs: Tuple[OrganDefinition, ...]) -> Dict[int, str]:
    return {o.class_id: o.evaluator_label for o in organs}


def _colors_bgr(organs: Tuple[OrganDefinition, ...]) -> Dict[int, Tuple[int, int, int]]:
    return {o.class_id: o.color_bgr for o in organs}


def _class_id_names(organs: Tuple[OrganDefinition, ...]) -> Dict[int, str]:
    """Short display names for viz (same as evaluator_label for simplicity)."""
    return {o.class_id: o.evaluator_label for o in organs}


def _evaluator_organ_list(organs: Tuple[OrganDefinition, ...]) -> List[str]:
    return [o.evaluator_label for o in organs]


# Real clinical: primary multiorgan path
REAL_CLINICAL_ORGAN_KEYS: List[str] = _keys(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_ORGAN_TO_CLASS: Dict[str, int] = _to_class_map(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_CLASS_TO_KEY: Dict[int, str] = {o.class_id: o.key for o in REAL_CLINICAL_ORGANS}
REAL_CLINICAL_EVAL_CLASS_MAPPING: Dict[int, str] = _eval_class_mapping(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = _colors_bgr(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_CLASS_ID_TO_DISPLAY: Dict[int, str] = _class_id_names(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_NUM_FOREGROUND: int = len(REAL_CLINICAL_ORGANS)
REAL_CLINICAL_NUM_CLASSES: int = REAL_CLINICAL_NUM_FOREGROUND + 1  # + background

# Simulation
SIMULATION_ORGAN_KEYS: List[str] = _keys(SIMULATION_THREE_ORGANS)
SIMULATION_ORGAN_TO_CLASS: Dict[str, int] = _to_class_map(SIMULATION_THREE_ORGANS)
SIMULATION_EVAL_CLASS_MAPPING: Dict[int, str] = _eval_class_mapping(SIMULATION_THREE_ORGANS)
SIMULATION_COLORS_BGR: Dict[int, Tuple[int, int, int]] = _colors_bgr(SIMULATION_THREE_ORGANS)
SIMULATION_CLASS_ID_TO_DISPLAY: Dict[int, str] = _class_id_names(SIMULATION_THREE_ORGANS)

# PFUS1
PFUS1_ORGAN_KEYS: List[str] = _keys(PFUS1_EIGHT_ORGANS)
PFUS1_ORGAN_TO_CLASS: Dict[str, int] = _to_class_map(PFUS1_EIGHT_ORGANS)
PFUS1_CLASS_TO_KEY: Dict[int, str] = {o.class_id: o.key for o in PFUS1_EIGHT_ORGANS}
PFUS1_EVAL_CLASS_MAPPING: Dict[int, str] = _eval_class_mapping(PFUS1_EIGHT_ORGANS)
PFUS1_COLORS_BGR: Dict[int, Tuple[int, int, int]] = _colors_bgr(PFUS1_EIGHT_ORGANS)
PFUS1_CLASS_ID_TO_DISPLAY: Dict[int, str] = _class_id_names(PFUS1_EIGHT_ORGANS)
PFUS1_NUM_FOREGROUND: int = len(PFUS1_EIGHT_ORGANS)
PFUS1_NUM_CLASSES: int = PFUS1_NUM_FOREGROUND + 1


def real_clinical_has_flags(mask) -> Dict[str, bool]:
    """Given multiclass HxW mask, return has_<key> booleans for dataset __getitem__."""
    import numpy as np

    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    return {f"has_{o.key}": bool((mask == o.class_id).any()) for o in REAL_CLINICAL_ORGANS}


def real_clinical_key_to_evaluator_label() -> Dict[str, str]:
    return {o.key: o.evaluator_label for o in REAL_CLINICAL_ORGANS}


def real_clinical_metrics_organ_list() -> List[str]:
    """Labels passed to MetricsEvaluator(..., organs=...)."""
    return _evaluator_organ_list(REAL_CLINICAL_ORGANS)


def simulation_metrics_organ_list() -> List[str]:
    return _evaluator_organ_list(SIMULATION_THREE_ORGANS)


def pfus1_has_flags(mask) -> Dict[str, bool]:
    """Given multiclass HxW mask, return has_<key> booleans for PFUS1Dataset / MultiOrganDataset."""
    import numpy as np

    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    return {f"has_{o.key}": bool((mask == o.class_id).any()) for o in PFUS1_EIGHT_ORGANS}


def pfus1_metrics_organ_list() -> List[str]:
    return _evaluator_organ_list(PFUS1_EIGHT_ORGANS)
