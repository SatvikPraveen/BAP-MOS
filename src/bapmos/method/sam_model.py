"""
SAM ViT-B wrapper: frozen encoders, trainable mask decoder, single-forward 'both' prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from bapmos.method.types import OrganPrompts

DEFAULT_META_SAM_CHECKPOINT = "models/sam_base/sam_vit_b_01ec64.pth"


class FrozenSAMMultiOrgan(nn.Module):
    """
    Segment Anything Model (ViT-B) with frozen image_encoder and prompt_encoder.

    Only ``mask_decoder`` parameters are trainable (requires_grad=True).
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        *,
        init: str = "meta",
        architecture_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        ckpt = str(Path(checkpoint).expanduser())
        init_key = str(init).lower().strip()
        from bapmos.paths import resolve_model_checkpoint

        if init_key == "medsam":
            meta_ckpt = resolve_model_checkpoint(
                architecture_checkpoint or DEFAULT_META_SAM_CHECKPOINT
            )
            if not meta_ckpt.is_file():
                raise FileNotFoundError(
                    f"Meta SAM architecture checkpoint not found for MedSAM init: {meta_ckpt}"
                )
            medsam_ckpt = resolve_model_checkpoint(ckpt)
            if not medsam_ckpt.is_file():
                raise FileNotFoundError(f"MedSAM checkpoint not found: {ckpt}")
            from bapmos.external_baselines.medsam_init.weight_loader import (
                load_medsam_weights_into_sam_vit_b,
            )

            self.sam = sam_model_registry[model_type](checkpoint=str(meta_ckpt))
            load_medsam_weights_into_sam_vit_b(self.sam, str(medsam_ckpt))
        elif init_key in ("meta", "sam"):
            sam_path = resolve_model_checkpoint(ckpt)
            if not sam_path.is_file():
                raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
            self.sam = sam_model_registry[model_type](checkpoint=str(sam_path))
        else:
            raise ValueError(
                f"Unknown SAM init {init!r}; use 'meta' (default) or 'medsam'"
            )
        self.img_size = self.sam.image_encoder.img_size
        self.transform = ResizeLongestSide(self.img_size)

        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

    @property
    def mask_decoder(self) -> nn.Module:
        return self.sam.mask_decoder

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.sam.mask_decoder.parameters()

    def prepare_image(self, image_rgb: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resize and normalize image; return (1, 3, H, W) tensor and original (H, W).

        Expects a channel-last RGB NumPy array of shape ``(H, W, 3)``.
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(f"image_rgb must be np.ndarray, got {type(image_rgb)}")
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise ValueError(
                f"image_rgb must have shape (H, W, 3), got {getattr(image_rgb, 'shape', None)}"
            )
        original_size = image_rgb.shape[:2]
        x = self.transform.apply_image(image_rgb)
        x = torch.as_tensor(x, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)
        x = self.sam.preprocess(x)
        return x, original_size

    @torch.no_grad()
    def encode_image(self, image_rgb: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x, original_size = self.prepare_image(image_rgb, device)
        emb = self.sam.image_encoder(x)
        return emb, original_size

    @torch.no_grad()
    def encode_prompts(
        self,
        prompt: OrganPrompts,
        original_size: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode box and/or points in one prompt_encoder call when mode is 'both'.
        """
        mode = prompt.effective_mode
        box_t: Optional[torch.Tensor] = None
        points_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if prompt.box_xyxy is not None and mode in ("box", "both"):
            box = prompt.box_xyxy.astype(np.float32)
            box_t = self.transform.apply_boxes(box[None, :], original_size)
            box_t = torch.as_tensor(box_t, device=device, dtype=torch.float32)

        if prompt.points_xy is not None and prompt.labels is not None and mode in ("point", "both"):
            pts = self.transform.apply_coords(prompt.points_xy, original_size)
            pts_t = torch.as_tensor(pts, device=device, dtype=torch.float32).unsqueeze(0)
            lab_t = torch.as_tensor(prompt.labels, device=device, dtype=torch.int32).unsqueeze(0)
            points_tuple = (pts_t, lab_t)

        if mode == "box":
            assert box_t is not None
            sparse, dense = self.sam.prompt_encoder(
                points=None,
                boxes=box_t.unsqueeze(1),
                masks=None,
            )
        elif mode == "point":
            assert points_tuple is not None
            sparse, dense = self.sam.prompt_encoder(
                points=points_tuple,
                boxes=None,
                masks=None,
            )
        elif mode == "both":
            assert box_t is not None and points_tuple is not None
            sparse, dense = self.sam.prompt_encoder(
                points=points_tuple,
                boxes=box_t.unsqueeze(1),
                masks=None,
            )
        else:
            raise ValueError(f"Unknown effective_mode: {mode}")

        return sparse, dense

    def decode_mask(
        self,
        image_embeddings: torch.Tensor,
        sparse: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        """Single mask_decoder forward; returns low-res logits (1, 1, 256, 256)."""
        low_res, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return low_res

    def forward_organ(
        self,
        image_embeddings: torch.Tensor,
        prompt: OrganPrompts,
        original_size: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode prompts (no grad) + decode (grad through mask_decoder)."""
        sparse, dense = self.encode_prompts(prompt, original_size, device)
        return self.decode_mask(image_embeddings, sparse, dense)
