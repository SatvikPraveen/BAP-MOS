"""Load MedSAM (or compatible) weights into a ``segment_anything`` ViT-B model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


# Canonical ``medsam_vit_b.pth`` (Zenodo) should load with very few missing keys.
DEFAULT_MAX_MISSING_KEYS = int(os.environ.get("MEDSAM_MAX_MISSING_KEYS", "50"))
# Warn when overlap drops below this ratio (matched / model keys).
DEFAULT_MIN_OVERLAP_RATIO = float(os.environ.get("MEDSAM_MIN_OVERLAP_RATIO", "0.95"))


@dataclass(frozen=True)
class MedsamLoadReport:
    """Structured summary of a MedSAM→SAM ViT-B weight overlay."""

    checkpoint_path: str
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    model_keys: int
    checkpoint_keys: int
    matched_keys: int

    @property
    def missing_n(self) -> int:
        return len(self.missing_keys)

    @property
    def unexpected_n(self) -> int:
        return len(self.unexpected_keys)

    @property
    def overlap_ratio(self) -> float:
        return self.matched_keys / self.model_keys if self.model_keys else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medsam_checkpoint": self.checkpoint_path,
            "medsam_missing_keys": self.missing_n,
            "medsam_unexpected_keys": self.unexpected_n,
            "medsam_matched_keys": self.matched_keys,
            "medsam_model_keys": self.model_keys,
            "medsam_checkpoint_keys": self.checkpoint_keys,
            "medsam_overlap_ratio": round(self.overlap_ratio, 6),
        }

    def log_summary(self) -> None:
        name = Path(self.checkpoint_path).name
        print(
            f"MedSAM load {name}: overlap={self.overlap_ratio:.1%} "
            f"({self.matched_keys}/{self.model_keys} model keys matched); "
            f"missing={self.missing_n}, unexpected={self.unexpected_n}"
        )
        if self.missing_keys and self.missing_n <= 20:
            print("  missing:", list(self.missing_keys))
        elif self.missing_keys:
            print("  first missing:", list(self.missing_keys[:20]), "...")
        if self.unexpected_keys and self.unexpected_n <= 20:
            print("  unexpected:", list(self.unexpected_keys))

    def assert_acceptable(
        self,
        *,
        max_missing_keys: int = DEFAULT_MAX_MISSING_KEYS,
        min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
    ) -> None:
        if self.missing_n > max_missing_keys:
            raise ValueError(
                f"MedSAM weight overlap too low: {self.missing_n} missing keys "
                f"(limit {max_missing_keys}). Check checkpoint path and SAM ViT-B architecture."
            )
        if self.overlap_ratio < min_overlap_ratio:
            raise ValueError(
                f"MedSAM overlap ratio {self.overlap_ratio:.1%} below minimum "
                f"{min_overlap_ratio:.1%} ({self.matched_keys}/{self.model_keys} keys matched)."
            )


def _pick_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
        for key in ("model", "state_dict", "network_weights", "weights"):
            inner = raw.get(key)
            if isinstance(inner, dict) and inner and isinstance(next(iter(inner.values())), torch.Tensor):
                return inner
    raise ValueError("Checkpoint did not contain a recognizable state dict.")


def _load_state_dict_file(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MedSAM / SAM checkpoint not found: {path}")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    return _pick_state_dict(raw)


def is_medsam_init_config(config: dict) -> bool:
    """True when checkpoint config denotes the MedSAM-init external baseline."""
    baseline = str(config.get("baseline", ""))
    return "medsam_init" in baseline or bool(config.get("medsam_checkpoint"))


def preview_medsam_overlap(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> MedsamLoadReport:
    """Analyze key overlap without mutating ``model`` weights."""
    state = _load_state_dict_file(checkpoint_path)
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    matched = model_keys & state_keys
    missing = tuple(sorted(model_keys - state_keys))
    unexpected = tuple(sorted(state_keys - model_keys))
    return MedsamLoadReport(
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        missing_keys=missing,
        unexpected_keys=unexpected,
        model_keys=len(model_keys),
        checkpoint_keys=len(state_keys),
        matched_keys=len(matched),
    )


def load_medsam_weights_into_sam_vit_b(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    max_missing_keys: int = DEFAULT_MAX_MISSING_KEYS,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> MedsamLoadReport:
    """
    Load weights into a SAM ``vit_b`` model (``sam_model_registry['vit_b']``).

    MedSAM releases typically ship a state dict compatible with SAM ViT-B.
    ``strict=False`` tolerates minor key mismatches; raises when overlap is too low.
    """
    state = _load_state_dict_file(checkpoint_path)
    preview = preview_medsam_overlap(model, checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report = MedsamLoadReport(
        checkpoint_path=preview.checkpoint_path,
        missing_keys=tuple(sorted(missing)),
        unexpected_keys=tuple(sorted(unexpected)),
        model_keys=preview.model_keys,
        checkpoint_keys=preview.checkpoint_keys,
        matched_keys=preview.matched_keys,
    )
    report.log_summary()
    report.assert_acceptable(
        max_missing_keys=max_missing_keys,
        min_overlap_ratio=min_overlap_ratio,
    )
    return report


def apply_medsam_encoder_init(
    model: torch.nn.Module,
    config: dict,
    *,
    checkpoint_path: Optional[str] = None,
    max_missing_keys: int = DEFAULT_MAX_MISSING_KEYS,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> MedsamLoadReport:
    """
    Overlay MedSAM weights on a SAM ViT-B model before loading a fine-tuned decoder.

    Used at fresh training, resume, and standalone inference so encoder state matches
    the MedSAM-init training definition.
    """
    from bapmos.paths import resolve_model_checkpoint

    path = checkpoint_path or config.get("medsam_checkpoint")
    if not path:
        raise ValueError(
            "MedSAM-init baseline requires config['medsam_checkpoint'] "
            "(MedSAM encoder must be applied before decoder restore)."
        )
    resolved = resolve_model_checkpoint(str(path))
    if not resolved.is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {path!r}")
    return load_medsam_weights_into_sam_vit_b(
        model,
        str(resolved),
        max_missing_keys=max_missing_keys,
        min_overlap_ratio=min_overlap_ratio,
    )


def verify_medsam_checkpoint(
    sam_checkpoint: str,
    medsam_checkpoint: str,
    *,
    max_missing_keys: int = DEFAULT_MAX_MISSING_KEYS,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> MedsamLoadReport:
    """
    Quick confidence check: build SAM ViT-B, preview+load MedSAM, return overlap report.

    Does not require a trained run checkpoint — only foundation weight files.
    """
    from segment_anything import sam_model_registry

    from bapmos.paths import resolve_model_checkpoint

    sam_path = resolve_model_checkpoint(sam_checkpoint)
    medsam_path = resolve_model_checkpoint(medsam_checkpoint)
    if not sam_path.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint!r}")
    if not medsam_path.is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {medsam_checkpoint!r}")

    model = sam_model_registry["vit_b"](checkpoint=str(sam_path))
    report = load_medsam_weights_into_sam_vit_b(
        model,
        str(medsam_path),
        max_missing_keys=max_missing_keys,
        min_overlap_ratio=min_overlap_ratio,
    )
    return report


def format_report_json(report: MedsamLoadReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
