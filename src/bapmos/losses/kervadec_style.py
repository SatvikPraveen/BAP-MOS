"""
Kervadec-style scheduled decoder loss (Eq. 7):

    L_total = α(t) [L_CE + (1/O) Σ_o L_Dice_o] + (1 - α(t)) L_BL

Boundary loss follows Kervadec et al., MICCAI 2019 / LIVIAETS boundary-loss:
signed distance map φ_G × softmax probability (SurfaceLoss / BoundaryLoss).
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# One-shot guard so online EDT fallback does not spam every step.
_ONLINE_DIST_MAP_WARNED = False


def _gt_long(gt_multi: torch.Tensor) -> torch.Tensor:
    if gt_multi.dim() == 4:
        return gt_multi.squeeze(1).long()
    return gt_multi.long()


def ce_dice_regional_loss(
    logits: torch.Tensor,
    gt_multi: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Cross-entropy plus mean foreground Dice loss (classes 1 … C-1)."""
    gt_flat = _gt_long(gt_multi)
    ce = F.cross_entropy(logits, gt_flat, reduction="mean")
    probs = torch.softmax(logits, dim=1)
    dice_losses = []
    for c in range(1, num_classes):
        pred_c = probs[:, c]
        gt_c = (gt_flat == c).float()
        inter = (pred_c * gt_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + gt_c.sum(dim=(1, 2))
        dice = (2.0 * inter + 1e-7) / (union + 1e-7)
        dice_losses.append(1.0 - dice.mean())
    if not dice_losses:
        return ce
    return ce + torch.stack(dice_losses).mean()


def class_distance_map(mask_binary: np.ndarray) -> np.ndarray:
    """
    Signed distance to contour φ_G for a binary foreground mask (H, W).

    Matches LIVIAETS ``one_hot2dist`` / readme: positive outside the object,
    negative inside. Empty masks return zeros.
    """
    posmask = mask_binary.astype(bool)
    if not posmask.any():
        return np.zeros(mask_binary.shape, dtype=np.float32)

    negmask = ~posmask
    # Positive outside, negative inside (official boundary-loss formulation).
    dist_map = (
        distance_transform_edt(negmask) * negmask
        - (distance_transform_edt(posmask) - 1.0) * posmask
    )
    return dist_map.astype(np.float32)


def multiclass_distance_maps(
    label_hw: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Build per-class signed distance maps for a single label image.

    Returns ``(C, H, W)`` float32. Background channel (0) is left at zero;
    empty foreground classes are zero (LIVIAETS style).
    """
    label = np.asarray(label_hw)
    if label.ndim != 2:
        raise ValueError(f"label_hw must be (H, W), got shape {label.shape}")
    h, w = label.shape
    out = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(1, num_classes):
        out[c] = class_distance_map(label == c)
    return out


def batch_multiclass_distance_maps(
    labels_bhw: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build ``(B, C, H, W)`` signed distance maps from integer labels ``(B, H, W)``."""
    labels = np.asarray(labels_bhw)
    if labels.ndim != 3:
        raise ValueError(f"labels_bhw must be (B, H, W), got shape {labels.shape}")
    maps = [multiclass_distance_maps(labels[b], num_classes) for b in range(labels.shape[0])]
    return np.stack(maps, axis=0)


class BoundaryDistMapCache:
    """
    Cache signed φ_G maps for fixed GT labels (LIVIAETS-style precompute).

    Trainers resize masks to a fixed decoder resolution (e.g. 256×256) with
    nearest-neighbor; the same sample therefore yields the same φ_G every epoch.
    Caching avoids repeated CPU ``distance_transform_edt`` on the training critical path.

    Entries are LRU-bounded (``max_entries``) so host RAM cannot grow without limit
    across large datasets or multi-fold jobs. Call ``clear()`` at fold boundaries
    when you want a hard reset.
    """

    def __init__(self, num_classes: int, *, max_entries: int = 20_000) -> None:
        self.num_classes = int(num_classes)
        self.max_entries = int(max_entries)
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def _store(self, cache_key: str, dist_cpu: torch.Tensor) -> None:
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self._cache[cache_key] = dist_cpu
            return
        while len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        self._cache[cache_key] = dist_cpu

    def get_or_compute(
        self,
        label_hw: np.ndarray,
        *,
        cache_key: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Return ``(C, H, W)`` signed distance maps for one label image.

        If ``cache_key`` is set, CPU tensors are stored and moved to ``device`` on hit.
        """
        label = np.asarray(label_hw)
        if label.ndim != 2:
            raise ValueError(f"label_hw must be (H, W), got shape {label.shape}")

        if cache_key is not None and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            dist_cpu = self._cache[cache_key]
            if device is None:
                return dist_cpu
            return dist_cpu.to(device=device, dtype=dtype, non_blocking=True)

        dist_np = multiclass_distance_maps(label, self.num_classes)
        dist_cpu = torch.from_numpy(dist_np)  # float32 CPU
        if cache_key is not None:
            self._store(cache_key, dist_cpu)

        if device is None:
            return dist_cpu
        return dist_cpu.to(device=device, dtype=dtype, non_blocking=True)

    def get_or_compute_batched(
        self,
        labels_bhw: np.ndarray,
        *,
        cache_keys: Optional[list] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ``(B, C, H, W)`` maps, optionally caching each sample by key."""
        labels = np.asarray(labels_bhw)
        if labels.ndim != 3:
            raise ValueError(f"labels_bhw must be (B, H, W), got shape {labels.shape}")
        keys = cache_keys if cache_keys is not None else [None] * labels.shape[0]
        if len(keys) != labels.shape[0]:
            raise ValueError("cache_keys length must match batch size")
        maps = [
            self.get_or_compute(labels[b], cache_key=keys[b], device=None, dtype=dtype)
            for b in range(labels.shape[0])
        ]
        stacked = torch.stack(maps, dim=0)
        if device is None:
            return stacked
        return stacked.to(device=device, dtype=dtype, non_blocking=True)


def _as_dist_maps_tensor(
    dist_maps: Union[torch.Tensor, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_classes: int,
    spatial: Tuple[int, int],
) -> torch.Tensor:
    """Normalize dist maps to ``(B, C, H, W)`` on ``device``."""
    if isinstance(dist_maps, np.ndarray):
        # Host ndarray → tensor, then async H2D when pinned / CUDA available.
        dist_t = torch.as_tensor(dist_maps, dtype=dtype).to(
            device=device, non_blocking=True
        )
    else:
        dist_t = dist_maps.to(device=device, dtype=dtype, non_blocking=True)

    if dist_t.dim() == 3:
        # (C, H, W) → single-item batch
        dist_t = dist_t.unsqueeze(0)
    if dist_t.dim() != 4:
        raise ValueError(
            f"dist_maps must be (C,H,W) or (B,C,H,W), got shape {tuple(dist_t.shape)}"
        )
    if dist_t.shape[0] != batch_size:
        raise ValueError(
            f"dist_maps batch {dist_t.shape[0]} != logits batch {batch_size}"
        )
    if dist_t.shape[1] != num_classes:
        raise ValueError(
            f"dist_maps classes {dist_t.shape[1]} != num_classes {num_classes}"
        )
    if tuple(dist_t.shape[-2:]) != spatial:
        raise ValueError(
            f"dist_maps spatial {tuple(dist_t.shape[-2:])} != logits spatial {spatial}"
        )
    return dist_t


def boundary_loss(
    logits: torch.Tensor,
    gt_multi: torch.Tensor,
    num_classes: int,
    *,
    dist_maps: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """
    Mean boundary loss over batch and foreground classes (LIVIAETS SurfaceLoss).

    L_BL = mean_{b,k∈fg,x,y}[ φ_G(b,k,x,y) · s_θ(b,k,x,y) ]

    Prefer passing precomputed ``dist_maps`` ``(B, C, H, W)`` or ``(C, H, W)``
    (signed φ_G). If omitted, maps are built from ``gt_multi`` on CPU once per call.
    Empty classes contribute zeros so the mean denominator stays fixed.
    """
    global _ONLINE_DIST_MAP_WARNED

    probs = F.softmax(logits, dim=1)
    bsz, n_cls, height, width = probs.shape
    if n_cls != num_classes:
        raise ValueError(f"logits classes {n_cls} != num_classes {num_classes}")

    if dist_maps is None:
        if not _ONLINE_DIST_MAP_WARNED:
            warnings.warn(
                "boundary_loss: dist_maps is None; computing signed φ_G online on "
                "CPU (synchronous). Prefer BoundaryDistMapCache / precomputed maps "
                "on the training path to avoid GPU stalls.",
                stacklevel=2,
            )
            _ONLINE_DIST_MAP_WARNED = True
        gt_flat = _gt_long(gt_multi)
        gt_np = gt_flat.detach().cpu().numpy().astype(np.int64, copy=False)
        dist_np = batch_multiclass_distance_maps(gt_np, num_classes)
        dist_t = torch.as_tensor(dist_np, dtype=logits.dtype).to(
            device=logits.device, non_blocking=True
        )
    else:
        dist_t = _as_dist_maps_tensor(
            dist_maps,
            device=logits.device,
            dtype=logits.dtype,
            batch_size=bsz,
            num_classes=num_classes,
            spatial=(height, width),
        )

    # Foreground only (class 0 left unused / zero in dist maps).
    pc = probs[:, 1:, ...]
    dc = dist_t[:, 1:, ...]
    return (pc * dc).mean()


def alpha_schedule(
    epoch: int,
    max_epochs: int,
    *,
    start: float = 1.0,
    end: float = 0.5,
    mode: str = "linear",
) -> float:
    """Return α(t) for epoch index ``epoch`` in ``[0, max_epochs-1]``."""
    if max_epochs <= 1:
        return float(start)
    t = min(max(int(epoch), 0), max_epochs - 1) / float(max_epochs - 1)
    if mode == "linear":
        return float(start + (end - start) * t)
    if mode == "cosine":
        return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t)))
    raise ValueError(f"Unknown alpha schedule {mode!r}; use 'linear' or 'cosine'")


def kervadec_total_loss(
    logits: torch.Tensor,
    gt_multi: torch.Tensor,
    num_classes: int,
    *,
    epoch: int,
    max_epochs: int,
    alpha_start: float = 1.0,
    alpha_end: float = 0.5,
    alpha_mode: str = "linear",
    dist_maps: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Scheduled CE+Dice + boundary loss blend."""
    regional = ce_dice_regional_loss(logits, gt_multi, num_classes)
    boundary = boundary_loss(logits, gt_multi, num_classes, dist_maps=dist_maps)
    alpha = alpha_schedule(
        epoch,
        max_epochs,
        start=alpha_start,
        end=alpha_end,
        mode=alpha_mode,
    )
    alpha_t = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
    total = alpha_t * regional + (1.0 - alpha_t) * boundary
    return total, {
        "alpha": alpha,
        "regional_loss": float(regional.detach().item()),
        "boundary_loss": float(boundary.detach().item()),
    }


@dataclass
class KervadecSchedule:
    alpha_start: float = 1.0
    alpha_end: float = 0.5
    alpha_schedule: str = "linear"


@dataclass
class DecoderLossConfig:
    """Decoder fine-tuning loss configuration."""

    mode: str = "ce_dice"
    kervadec: KervadecSchedule = field(default_factory=KervadecSchedule)
    max_epochs: int = 300

    @classmethod
    def from_training_config(cls, cfg: dict, *, max_epochs: int) -> "DecoderLossConfig":
        training = cfg.get("training") or {}
        kervadec_raw = training.get("kervadec") or {}
        return cls(
            mode=str(training.get("loss_mode", "ce_dice")).lower(),
            kervadec=KervadecSchedule(
                alpha_start=float(kervadec_raw.get("alpha_start", 1.0)),
                alpha_end=float(kervadec_raw.get("alpha_end", 0.5)),
                alpha_schedule=str(kervadec_raw.get("alpha_schedule", "linear")),
            ),
            max_epochs=int(max_epochs),
        )


class DecoderLossComputer:
    """Callable loss used by SAM decoder trainers."""

    def __init__(self, config: DecoderLossConfig) -> None:
        self.config = config
        self._last_components: Dict[str, float] = {}

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def last_components(self) -> Dict[str, float]:
        return dict(self._last_components)

    def __call__(
        self,
        logits: torch.Tensor,
        gt_multi: torch.Tensor,
        num_classes: int,
        *,
        epoch: int = 0,
        dist_maps: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        if self.config.mode == "ce_dice":
            loss = ce_dice_regional_loss(logits, gt_multi, num_classes)
            self._last_components = {"regional_loss": float(loss.detach().item())}
            return loss

        if self.config.mode == "kervadec":
            loss, components = kervadec_total_loss(
                logits,
                gt_multi,
                num_classes,
                epoch=epoch,
                max_epochs=self.config.max_epochs,
                alpha_start=self.config.kervadec.alpha_start,
                alpha_end=self.config.kervadec.alpha_end,
                alpha_mode=self.config.kervadec.alpha_schedule,
                dist_maps=dist_maps,
            )
            self._last_components = components
            return loss

        raise ValueError(
            f"Unknown training.loss_mode {self.config.mode!r}; "
            "expected 'ce_dice' or 'kervadec'"
        )
