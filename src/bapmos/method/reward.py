"""
Bandit reward functions for BAP-MOS.

Modes:

- **fixed_clip:** constant τ (mm), ``R = 1 - min(MSD, τ) / τ`` (legacy MSD-only).
- **curriculum:** epoch-dependent ``τ_E``, same MSD-only formula with decaying τ_E.
- **composite_fixed_clip:** fixed τ with Dice + MSD + HD95, λ=3 HD95 scale
  (paper / protocol default)::

  ``R = 1 - [0.3(1-Dice) + 0.5·min(MSD,τ)/τ + 0.2·min(HD95,λτ)/(λτ)]``

  λ is a fixed scale-normalization constant (default 3), not searched in the outer loop.
"""

from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional, TypedDict

import numpy as np

from bapmos.method.curriculum_reward import (
    compute_gamma,
    curriculum_reward,
    rewards_from_validation_msd as _curriculum_rewards,
    tau_at_epoch,
)

RewardMode = Literal["fixed_clip", "curriculum", "composite_fixed_clip"]

# Composite reward weights (fixed protocol constants, not searched).
COMPOSITE_DICE_WEIGHT = 0.3
COMPOSITE_MSD_WEIGHT = 0.5
COMPOSITE_HD95_WEIGHT = 0.2
DEFAULT_HD95_LAMBDA = 3.0


class OrganProbeMetrics(TypedDict):
    msd_mm: float
    hd95_mm: float
    dice: float


def fixed_clip_reward(msd_mm: float, clip_max_mm: float = 20.0) -> float:
    """
    Fixed clipping factor τ (constant across training).

    R = 1 - min(MSD, τ) / τ  ∈ [0, 1]
    """
    tau = max(float(clip_max_mm), 1e-6)
    msd = max(0.0, float(msd_mm))
    return 1.0 - min(msd, tau) / tau


def composite_fixed_clip_reward(
    dice: float,
    msd_mm: float,
    hd95_mm: float,
    clip_max_mm: float,
    *,
    hd95_lambda: float = DEFAULT_HD95_LAMBDA,
) -> float:
    """
    Composite bandit reward with fixed HD95 scale λτ.

    R = 1 - [w_d(1-Dice) + w_m·min(MSD,τ)/τ + w_h·min(HD95,λτ)/(λτ)]
    """
    tau = max(float(clip_max_mm), 1e-6)
    tau_hd95 = max(float(hd95_lambda) * tau, 1e-6)
    dice_term = 1.0 - max(0.0, min(1.0, float(dice)))
    msd_term = min(max(0.0, float(msd_mm)), tau) / tau
    hd95_term = min(max(0.0, float(hd95_mm)), tau_hd95) / tau_hd95
    penalty = (
        COMPOSITE_DICE_WEIGHT * dice_term
        + COMPOSITE_MSD_WEIGHT * msd_term
        + COMPOSITE_HD95_WEIGHT * hd95_term
    )
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def reward_threshold_label(
    reward_mode: RewardMode,
    epoch: int,
    *,
    clip_max_mm: float = 20.0,
    tau_max: float = 15.0,
    tau_min: float = 0.1,
    max_epochs: int = 300,
    decay_fraction: float = 0.8,
    gamma: Optional[float] = None,
) -> float:
    """Scalar MSD threshold used for logging (τ or τ_E).

    For composite mode HD95 scale λτ, use :func:`hd95_reward_threshold_label`.
    """
    if reward_mode == "fixed_clip":
        return clip_max_mm
    if reward_mode == "composite_fixed_clip":
        return clip_max_mm
    return tau_at_epoch(
        epoch,
        tau_max=tau_max,
        tau_min=tau_min,
        max_epochs=max_epochs,
        decay_fraction=decay_fraction,
        gamma=gamma,
    )


def hd95_reward_threshold_label(
    clip_max_mm: float,
    *,
    hd95_lambda: float = DEFAULT_HD95_LAMBDA,
) -> float:
    """HD95 clipping range λτ for logging (composite mode)."""
    return float(hd95_lambda) * float(clip_max_mm)


def rewards_from_validation_msd(
    organ_msd_mm: Mapping[str, float],
    epoch: int,
    *,
    reward_mode: RewardMode = "curriculum",
    clip_max_mm: float = 20.0,
    tau_max: float = 15.0,
    tau_min: float = 0.1,
    max_epochs: int = 300,
    decay_fraction: float = 0.8,
    gamma: Optional[float] = None,
) -> Dict[str, float]:
    """Dispatch per-organ MSD (mm) to bandit rewards in [0, 1]."""
    if reward_mode == "fixed_clip":
        return {organ: fixed_clip_reward(msd, clip_max_mm) for organ, msd in organ_msd_mm.items()}
    if reward_mode == "curriculum":
        if gamma is None:
            gamma = compute_gamma(tau_max, tau_min, max_epochs, decay_fraction)
        return _curriculum_rewards(
            organ_msd_mm,
            epoch,
            tau_max=tau_max,
            tau_min=tau_min,
            max_epochs=max_epochs,
            decay_fraction=decay_fraction,
            gamma=gamma,
        )
    if reward_mode == "composite_fixed_clip":
        raise ValueError(
            "composite_fixed_clip requires Dice/MSD/HD95; use rewards_from_validation_metrics"
        )
    raise ValueError(f"Unknown reward_mode: {reward_mode!r}")


def rewards_from_validation_metrics(
    organ_metrics: Mapping[str, OrganProbeMetrics],
    *,
    clip_max_mm: float = 20.0,
    hd95_lambda: float = DEFAULT_HD95_LAMBDA,
) -> Dict[str, float]:
    """Composite rewards from per-organ probe metrics (Dice, MSD, HD95)."""
    out: Dict[str, float] = {}
    for organ, metrics in organ_metrics.items():
        msd = float(metrics["msd_mm"])
        hd95 = float(metrics["hd95_mm"])
        dice = float(metrics["dice"])
        if not all(np.isfinite(x) for x in (msd, hd95, dice)):
            continue
        out[organ] = composite_fixed_clip_reward(
            dice,
            msd,
            hd95,
            clip_max_mm,
            hd95_lambda=hd95_lambda,
        )
    return out
