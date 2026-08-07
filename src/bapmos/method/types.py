"""Shared types for ``bapmos.method``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

Action = Literal["box", "point", "both"]
EffectiveMode = Literal["box", "point", "both"]


@dataclass(frozen=True)
class OrganPrompts:
    """Prompt bundle for one organ on one slice.

    ``frozen=True`` only blocks attribute reassignment on the dataclass; the NumPy
    arrays in ``box_xyxy`` / ``points_xy`` / ``labels`` remain mutable in place.
    """

    organ_key: str
    mode: Action
    effective_mode: EffectiveMode
    box_xyxy: Optional[np.ndarray]  # (4,) SAM format [x_min, y_min, x_max, y_max]
    points_xy: Optional[np.ndarray]  # (P, 2) in image (x, y)
    labels: Optional[np.ndarray]  # (P,) int32, 1=positive 0=negative


@dataclass
class BlockState:
    """Arms frozen for the current decision block."""

    arms_by_organ: Dict[str, Action]
    batches_in_block: int = 0


DEFAULT_ARMS: List[Action] = ["box", "point", "both"]
