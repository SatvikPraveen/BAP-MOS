"""Run inference for a single-organ (binary) SAM decoder checkpoint."""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from tqdm import tqdm

from bapmos.multiorgan.dataset_multi_organ import compute_bounding_box
from bapmos.paths import resolve_model_checkpoint, resolve_training_data_root, resolve_under_project
from .dataset_single_organ import SingleOrganDataset


def stable_rng_from_id(sample_id: str, base_seed: int):
    h = 2166136261
    for c in (sample_id + str(base_seed)):
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.default_rng(h)


def dice_coefficient(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return (2.0 * inter) / union


class SingleOrganInference:
    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.organ = ckpt.get("organ") or self.config.get("organ")
        if self.organ is None:
            raise ValueError("Checkpoint missing organ; train with bapmos.legacy.singleorgan trainers.")
        if ckpt.get("num_classes", 2) != 2:
            raise ValueError(f"Expected single-organ checkpoint num_classes=2, got {ckpt.get('num_classes')}")
        self.prompt_type = ckpt.get("prompt_type") or self.config.get("prompt_type", "points")
        self.seed = int(self.config.get("seed", 42))

        sam_path = self.config.get("sam_checkpoint", "models/sam_base/sam_vit_b_01ec64.pth")
        sam_path = str(resolve_model_checkpoint(sam_path))
        self.model = sam_model_registry["vit_b"](checkpoint=sam_path).to(self.device)
        self.model.mask_decoder.load_state_dict(ckpt["model_state"]["mask_decoder"])
        self.model.eval()
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def _prepare_image(self, image_rgb):
        original_size = image_rgb.shape[:2]
        image_resized = self.transform.apply_image(image_rgb)
        x = torch.as_tensor(image_resized, device=self.device).float()
        x = x.permute(2, 0, 1).contiguous()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(3, 1, 1)
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x.unsqueeze(0), original_size

    def _points_from_gt(self, mask_bin, original_size, image_id):
        from bapmos.multiorgan.dataset_multi_organ import sample_negative_points_from_ring, sample_positive_points

        rng = stable_rng_from_id(image_id, self.seed)
        organ_u8 = (mask_bin > 0).astype(np.uint8)
        if organ_u8.sum() == 0:
            return None
        pos = sample_positive_points(
            organ_u8,
            num_points=self.config.get("num_pos_per_organ", 1),
            rng=rng,
        )
        if pos is None:
            return None
        neg = sample_negative_points_from_ring(
            organ_u8,
            ring_width=self.config.get("ring_width", 20),
            num_neg=self.config.get("num_neg_points", 3),
            rng=rng,
        )
        if neg is not None:
            points = np.vstack([pos, neg])
            labels = np.concatenate(
                [np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)]
            )
        else:
            points = pos
            labels = np.ones(len(pos), dtype=np.int32)
        pts_t = self.transform.apply_coords(points, original_size)
        pts_t = torch.as_tensor(pts_t, device=self.device, dtype=torch.float32).unsqueeze(0)
        lab_t = torch.as_tensor(labels, device=self.device, dtype=torch.int32).unsqueeze(0)
        return pts_t, lab_t

    def _box_from_gt(self, mask_bin, original_size):
        organ_u8 = (mask_bin > 0).astype(np.uint8)
        box = compute_bounding_box(organ_u8)
        if box is None:
            return None
        boxes_t = self.transform.apply_boxes(np.array([box]), original_size)
        boxes_t = torch.as_tensor(boxes_t, device=self.device, dtype=torch.float32)
        return boxes_t[0:1, :]

    def predict_logits(self, image_rgb, mask_bin_gt, image_id: str):
        """Uses GT mask only to build prompts (same as training)."""
        x, original_size = self._prepare_image(image_rgb)
        with torch.no_grad():
            img_emb = self.model.image_encoder(x)
        if self.prompt_type == "box":
            box_t = self._box_from_gt(mask_bin_gt, original_size)
            if box_t is None:
                return None
            sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t.unsqueeze(1), masks=None)
        else:
            pl = self._points_from_gt(mask_bin_gt, original_size, image_id)
            if pl is None:
                return None
            pts_t, lab_t = pl
            sparse, dense = self.model.prompt_encoder(points=(pts_t, lab_t), boxes=None, masks=None)
        low_res_mask, _ = self.model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = torch.zeros(1, 2, 256, 256, device=self.device, dtype=torch.float32)
        logits[0, 1, :, :] = low_res_mask[0, 0]
        organ_probs = torch.sigmoid(logits[:, 1:, :, :])
        background_prob = 1.0 - organ_probs.max(dim=1, keepdim=True)[0]
        logits[:, 0:1, :, :] = torch.logit(background_prob, eps=1e-7)
        pred_small = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.uint8)
        pred_full = cv2.resize(pred_small, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return pred_full

    def run_split(self, data_root: str, split: str, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        ds = SingleOrganDataset(data_root, organ=self.organ, split=split)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        rows = []
        for batch in tqdm(loader, desc=f"{split}"):
            image = batch["image"][0].numpy()
            mask_gt = batch["mask"][0].numpy()
            fn = batch["filename"][0]
            image_id = Path(fn).stem
            if mask_gt.max() == 0:
                continue
            pred = self.predict_logits(image, mask_gt, image_id)
            if pred is None:
                continue
            gt_full = mask_gt
            d = dice_coefficient(pred > 0, gt_full > 0)
            rows.append({"filename": fn, "dice": d})
            cv2.imwrite(str(pred_dir / f"{image_id}_pred.png"), pred * 255)
        df = pd.DataFrame(rows)
        if len(df):
            df.to_csv(out_dir / "metrics.csv", index=False)
        print(f"Saved {len(rows)} predictions to {pred_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    data_root = args.data_root or str(resolve_training_data_root("case1"))
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ckpt_path.parent / f"inference_{args.split}"

    eng = SingleOrganInference(ckpt_path, device=args.device)
    print(f"Organ={eng.organ}, prompt={eng.prompt_type}, split={args.split}")
    eng.run_split(data_root, args.split, out_dir)


if __name__ == "__main__":
    main()
