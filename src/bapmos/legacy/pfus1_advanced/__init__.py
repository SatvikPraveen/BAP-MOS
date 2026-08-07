"""Historical PFUS1-advanced helpers (not required for main paper results).

Cone crop, letterbox, and scale-aware prompt geometry utilities.
"""

from bapmos.legacy.pfus1_advanced.scale_aware_prompts import (
    apply_box_margin,
    is_scale_aware_prompt_geometry,
    prompt_geometry_summary,
    resolve_ring_width,
    sample_per_organ_points_with_scale_aware_geometry,
)
from bapmos.legacy.pfus1_advanced.ultrasound_preprocess import (
    PreprocessParams,
    letterbox_resize,
    preprocess_ultrasound_frame,
)

__all__ = [
    "PreprocessParams",
    "apply_box_margin",
    "is_scale_aware_prompt_geometry",
    "letterbox_resize",
    "preprocess_ultrasound_frame",
    "prompt_geometry_summary",
    "resolve_ring_width",
    "sample_per_organ_points_with_scale_aware_geometry",
]
