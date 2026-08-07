"""
Curriculum-driven boundary reward for BAP-MOS.

The allowable error threshold decays exponentially across training epochs::

    τ_E = max(τ_min, τ_max · γ^E)

where γ is chosen so τ reaches τ_min at ``decay_fraction`` of ``max_epochs``::

    γ = (τ_min / τ_max)^(1 / (decay_fraction · max_epochs))

Given validation MSD for organ o at block ending at epoch E, the bandit reward is::

    R_{t,o}^{(E)} = 1 - min(MSD_{t,o}, τ_E) / τ_E

which lies in the closed interval [0, 1] when MSD ≥ 0
(equals 1 at MSD = 0; equals 0 when MSD ≥ τ_E).
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional


def compute_gamma(
    tau_max: float,
    tau_min: float,
    max_epochs: int,
    decay_fraction: float = 0.8,
) -> float:
    """
  Derive γ so τ_E hits τ_min at epoch ``int(decay_fraction * max_epochs)``.

    γ = (τ_min / τ_max)^(1 / (decay_fraction · max_epochs))
    """
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if not (0.0 < decay_fraction <= 1.0):
        raise ValueError("decay_fraction must be in (0, 1]")
    if tau_max <= 0 or tau_min <= 0:
        raise ValueError("tau_max and tau_min must be positive")
    if tau_min > tau_max:
        raise ValueError("tau_min must be <= tau_max")

    steps = decay_fraction * float(max_epochs)
    if steps <= 0:
        return 1.0
    return (tau_min / tau_max) ** (1.0 / steps)


def tau_at_epoch(
    epoch: int,
    *,
    tau_max: float = 15.0,
    tau_min: float = 0.1,
    max_epochs: int = 300,
    decay_fraction: float = 0.8,
    gamma: Optional[float] = None,
) -> float:
    """
    Curriculum threshold at epoch E (0-based or 1-based — use consistently).

    τ_E = max(τ_min, τ_max · γ^E)
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if gamma is None:
        gamma = compute_gamma(tau_max, tau_min, max_epochs, decay_fraction)
    return max(tau_min, tau_max * (gamma ** float(epoch)))


def curriculum_reward(
    msd_mm: float,
    epoch: int,
    *,
    tau_max: float = 15.0,
    tau_min: float = 0.1,
    max_epochs: int = 300,
    decay_fraction: float = 0.8,
    gamma: Optional[float] = None,
) -> float:
    """
    R_{t,o}^{(E)} = 1 - min(MSD, τ_E) / τ_E  ∈ [0, 1].
    """
    tau_e = tau_at_epoch(
        epoch,
        tau_max=tau_max,
        tau_min=tau_min,
        max_epochs=max_epochs,
        decay_fraction=decay_fraction,
        gamma=gamma,
    )
    msd = max(0.0, float(msd_mm))
    clipped = min(msd, tau_e)
    return 1.0 - clipped / tau_e


def rewards_from_validation_msd(
    organ_msd_mm: Mapping[str, float],
    epoch: int,
    **kwargs,
) -> Dict[str, float]:
    """Map per-organ validation MSD (mm) to curriculum rewards."""
    return {
        organ: curriculum_reward(msd, epoch, **kwargs)
        for organ, msd in organ_msd_mm.items()
    }
