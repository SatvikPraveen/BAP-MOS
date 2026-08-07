"""Loss-mode policy for BAPMOS vs non-method paths.

**BAP-MOS method** (``bapmos.method`` — Meta SAM or MedSAM init)::

    training.loss_mode = kervadec   # scheduled CE+Dice + boundary

**Everything else** (external baselines, multiorgan SAM/MedSAM box/points,
legacy fixed-ratio prompting ``box`` / ``point`` / ``box_point`` /
``boxpoint_box_point``, legacy bandit policies, nnU-Net)::

    training.loss_mode = ce_dice    # regional CE + mean foreground Dice
    # nnU-Net keeps its own stock Dice+CE trainer loss
"""

from __future__ import annotations

from typing import Any, Dict

BAPMOS_METHOD_LOSS_MODE = "kervadec"
NON_METHOD_LOSS_MODE = "ce_dice"

_DEFAULT_KERVADEC = {
    "alpha_start": 1.0,
    "alpha_end": 0.5,
    "alpha_schedule": "linear",
}


def force_bapmos_method_kervadec_loss(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pin BAP-MOS method configs to Kervadec-style decoder loss."""
    training = config.setdefault("training", {})
    training["loss_mode"] = BAPMOS_METHOD_LOSS_MODE
    kerv = training.setdefault("kervadec", {})
    if not isinstance(kerv, dict):
        training["kervadec"] = dict(_DEFAULT_KERVADEC)
    else:
        for key, val in _DEFAULT_KERVADEC.items():
            kerv.setdefault(key, val)
    return config


def force_non_method_ce_dice_loss(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pin non-method / baseline configs to regional CE+Dice (never Kervadec)."""
    training = config.setdefault("training", {})
    training["loss_mode"] = NON_METHOD_LOSS_MODE
    training.pop("kervadec", None)
    return config


# Back-compat alias used by external_baselines / multiorgan entrypoints.
force_external_baseline_ce_dice_loss = force_non_method_ce_dice_loss
