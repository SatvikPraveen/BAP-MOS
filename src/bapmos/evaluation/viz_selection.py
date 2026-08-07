"""Choose which test slices receive qualitative panel exports."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np

_FRAME_RE = re.compile(r"^(.+)_frame_(\d+)$")

from bapmos.legacy.optimization.metrics import MetricsEvaluator

VisualizationSelectionMode = Literal[
    "all", "random", "worst_msd", "best_msd", "per_patient_even"
]


@dataclass(frozen=True)
class SliceVizRecord:
    """One test slice, ranked using evaluator rows already appended for ``sample_id``."""

    sample_stem: str
    sample_id: str
    mean_dice: float
    mean_msd: float
    mean_hd95: float


def aggregate_slice_metrics_for_image(evaluator: MetricsEvaluator, image_id: str) -> Optional[SliceVizRecord]:
    """Mean Dice / MSD / HD95 across organ-rows for this slice (matches MetricsEvaluator CSV semantics)."""
    rows = [m for m in evaluator.per_slice_metrics if str(m["image_id"]) == str(image_id)]
    if not rows:
        return None
    stem = Path(str(image_id)).stem

    dices = [float(m["dice"]) for m in rows]
    msds = [float(m["msd_mm"]) for m in rows if m.get("msd_mm") is not None]
    hd95s = [float(m["hd95_mm"]) for m in rows if m.get("hd95_mm") is not None]

    mean_dice = float(np.mean(dices))
    mean_msd = float(np.mean(msds)) if msds else float("nan")
    mean_hd95 = float(np.mean(hd95s)) if hd95s else float("nan")

    return SliceVizRecord(
        sample_stem=stem,
        sample_id=str(image_id),
        mean_dice=mean_dice,
        mean_msd=mean_msd,
        mean_hd95=mean_hd95,
    )


def _frame_index(stem: str) -> int:
    m = _FRAME_RE.match(stem)
    if not m:
        return 0
    return int(m.group(2))


def _patient_id(stem: str) -> str:
    m = _FRAME_RE.match(stem)
    return m.group(1) if m else stem


def select_stems_evenly_per_patient(stems: Sequence[str], per_patient: int) -> List[str]:
    """Evenly spaced frame stems per patient (PFUS1 ``Pxxx_frame_yyy`` layout)."""
    by_patient: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        by_patient[_patient_id(s)].append(str(s))
    picked: List[str] = []
    for pid in sorted(by_patient):
        ordered = sorted(by_patient[pid], key=_frame_index)
        n = len(ordered)
        k = min(per_patient, n)
        if k == 0:
            continue
        if k == 1:
            picked.append(ordered[0])
            continue
        idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        picked.extend(ordered[i] for i in idxs)
    return picked


def select_slices_per_patient_even(
    records: Sequence[SliceVizRecord],
    per_patient: int,
) -> List[SliceVizRecord]:
    """Up to ``per_patient`` evenly spaced slices per patient bucket."""
    if per_patient <= 0 or not records:
        return []
    stem_to_rec = {r.sample_stem: r for r in records}
    picked_stems = select_stems_evenly_per_patient(list(stem_to_rec), per_patient)
    return [stem_to_rec[s] for s in picked_stems if s in stem_to_rec]


def select_slices_for_visualization(
    records: Sequence[SliceVizRecord],
    mode: VisualizationSelectionMode,
    max_n: Optional[int],
    seed: int,
    *,
    per_patient_max: Optional[int] = None,
) -> List[SliceVizRecord]:
    """
    Return ordered list of records that should receive figure exports.

    ``worst_msd`` / ``best_msd`` sort by mean slice MSD (NaN sinks to the unfavorable tail).
    """
    items = list(records)
    if not items:
        return []

    if mode == "per_patient_even":
        cap = per_patient_max if per_patient_max is not None else max_n
        if cap is None:
            raise ValueError("per_patient_even requires per_patient_max or max_n")
        return select_slices_per_patient_even(items, int(cap))

    rng = random.Random(seed)

    if mode == "random":
        rng.shuffle(items)
        out = items
    elif mode == "worst_msd":
        items.sort(
            key=lambda r: (
                math.isnan(r.mean_msd),
                -(r.mean_msd if not math.isnan(r.mean_msd) else 0.0),
            )
        )
        out = items
    elif mode == "best_msd":
        items.sort(
            key=lambda r: (
                math.isnan(r.mean_msd),
                r.mean_msd if not math.isnan(r.mean_msd) else float("inf"),
            )
        )
        out = items
    else:
        # "all" — preserve dataset order unless capped (then reproducible random subset)
        out = list(items)
        if max_n is not None:
            rng.shuffle(out)

    if max_n is not None and max_n >= 0:
        out = out[: max_n]
    return out
