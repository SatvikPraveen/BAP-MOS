"""Shared decoder fine-tuning losses."""

from bapmos.losses.kervadec_style import (
    BoundaryDistMapCache,
    DecoderLossComputer,
    DecoderLossConfig,
    KervadecSchedule,
    alpha_schedule,
    batch_multiclass_distance_maps,
    boundary_loss,
    ce_dice_regional_loss,
    class_distance_map,
    kervadec_total_loss,
    multiclass_distance_maps,
)
from bapmos.losses.loss_policy import (
    BAPMOS_METHOD_LOSS_MODE,
    NON_METHOD_LOSS_MODE,
    force_bapmos_method_kervadec_loss,
    force_external_baseline_ce_dice_loss,
    force_non_method_ce_dice_loss,
)

__all__ = [
    "BAPMOS_METHOD_LOSS_MODE",
    "NON_METHOD_LOSS_MODE",
    "BoundaryDistMapCache",
    "DecoderLossComputer",
    "DecoderLossConfig",
    "KervadecSchedule",
    "alpha_schedule",
    "batch_multiclass_distance_maps",
    "boundary_loss",
    "ce_dice_regional_loss",
    "class_distance_map",
    "force_bapmos_method_kervadec_loss",
    "force_external_baseline_ce_dice_loss",
    "force_non_method_ce_dice_loss",
    "kervadec_total_loss",
    "multiclass_distance_maps",
]
