"""Fine-tuning scope for the fixed-prompt SAM trainers (decoder-only vs full model).

``decoder`` (default, the paper's protocol): image and prompt encoders frozen, mask decoder
trained. ``full``: every SAM ViT-B parameter is trained, which is the "Full Model" arm of the
decoder-only vs full-model comparison (Table I).

Checkpoints store ``model_state = {"mask_decoder": ...}`` for ``decoder`` and additionally
``image_encoder`` / ``prompt_encoder`` for ``full``; :func:`load_sam_model_state` restores
whichever parts are present, so decoder-only checkpoints load unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Dict

import torch

FINETUNE_SCOPES = ("decoder", "full")
_SAM_PARTS = ("image_encoder", "prompt_encoder", "mask_decoder")


def check_finetune_scope(scope: str) -> str:
    scope = str(scope or "decoder")
    if scope not in FINETUNE_SCOPES:
        raise ValueError(f"finetune_scope must be one of {FINETUNE_SCOPES}, got {scope!r}")
    return scope


def apply_finetune_scope(model: torch.nn.Module, scope: str) -> None:
    """Set ``requires_grad`` on the SAM parts trained under ``scope``."""
    full = check_finetune_scope(scope) == "full"
    for p in model.image_encoder.parameters():
        p.requires_grad = full
    for p in model.prompt_encoder.parameters():
        p.requires_grad = full
    for p in model.mask_decoder.parameters():
        p.requires_grad = True


def encoder_grad(scope: str, is_train: bool):
    """Context for encoder forwards: gradients only when training the full model."""
    if is_train and check_finetune_scope(scope) == "full":
        return contextlib.nullcontext()
    return torch.no_grad()


def sam_model_state(model: torch.nn.Module, scope: str) -> Dict[str, dict]:
    """``model_state`` checkpoint entry: decoder only, or every SAM part for ``full``."""
    parts = _SAM_PARTS if check_finetune_scope(scope) == "full" else ("mask_decoder",)
    return {name: getattr(model, name).state_dict() for name in parts}


def load_sam_model_state(model: torch.nn.Module, model_state: Dict[str, dict]) -> None:
    """Restore the SAM parts present in a ``model_state`` checkpoint entry."""
    if "mask_decoder" not in model_state:
        raise KeyError("model_state has no mask_decoder weights")
    for name in _SAM_PARTS:
        if name in model_state:
            getattr(model, name).load_state_dict(model_state[name])
