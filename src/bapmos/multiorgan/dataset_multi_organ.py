"""
Multi-organ SAM dataset (simulation, clinical case1/2 / pooled, PFUS1).

Loads PNG slices + multiclass combined masks and returns taxonomy-aware
``has_<organ>`` flags. Class IDs follow ``bapmos.data.organ_registry``:

- Clinical (4 fg): Bladder=1, PTV=2, Rectum=3, Urethra=4
- Simulation (3 fg): Rectum=1, Bladder=2, PTV1=3
- PFUS1 (8 fg): see ``PFUS1_EIGHT_ORGANS``
"""

import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset

from typing import Any, Dict, List, Optional

from ..data.organ_registry import (
    PFUS1_ORGAN_KEYS,
    REAL_CLINICAL_ORGAN_TO_CLASS,
    real_clinical_has_flags,
    pfus1_has_flags,
)
from ..paths import (
    find_combined_masks_dir,
    find_training_images_dir,
    is_pfus1_advanced_training_root,
    is_pfus1_training_root,
    pfus1_advanced_image_root,
    pfus1_image_root,
    project_root,
)
from ..pfus1.dataset_pfus1 import parse_sample_line


def multi_organ_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Do not stack raw image/mask arrays (variable H×W for PFUS1 and others).

    SAM / optimization trainers iterate ``batch["image"][i]`` per sample internally.
    """
    if not batch:
        return {}
    return {k: [sample[k] for sample in batch] for k in batch[0]}


class MultiOrganDataset(Dataset):
    """
    Dataset for multi-organ segmentation (5 classes) - real clinical data.

    Args:
        data_root (str): Dataset root, e.g. ``data/prostate/case1`` with
            ``case1_dicom_png/`` or ``images/``, ``masks/``, and ``splits_stratified/``.
        split (str): One of ['train', 'val', 'test']
        transform (callable, optional): Transformations to apply
    
    Returns multi-class masks with all organs.
    Class IDs: Background=0, Bladder=1, PTV=2, Rectum=3, Urethra=4
    """
    
    CLASS_NAMES = {
        0: 'Background',
        1: 'Bladder',
        2: 'PTV',
        3: 'Rectum',
        4: 'Urethra'
    }
    
    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        splits_subdir: str = "splits_stratified",
        image_embedding_dir: Optional[str] = None,
    ):
        """
        Args:
            data_root (str): Dataset root (see class docstring for layout).
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Transformations to apply
            splits_subdir (str): Subdirectory of ``data_root`` with ``train.txt`` etc.
                Default ``splits_stratified`` matches ``utils.create_stratified_splits``
                output (protocol seed ``STRATIFICATION_SEED``, default 42).
            image_embedding_dir (str, optional): If set **and** ``data_root`` is the PFUS1
                bundle, each sample includes ``embedding_path`` pointing to
                ``{patient}_{frame}.pt`` under this directory (flat layout).
        """
        dr = Path(data_root)
        if not dr.is_absolute():
            dr = project_root() / dr
        self.data_root = dr.resolve()
        self.split = split
        self.transform = transform
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Must be one of ['train', 'val', 'test']"
            )
        
        splits_dir = self.data_root / splits_subdir
        if not splits_dir.exists():
            raise FileNotFoundError(
                f"Expected {splits_subdir}/ with train.txt, val.txt, test.txt under {self.data_root}. "
                f"Found: {list(self.data_root.iterdir()) if self.data_root.is_dir() else 'missing'}"
            )

        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.image_files = []
        self.mask_files = []
        self.sample_ids: list[str] = []
        self._is_pfus1 = is_pfus1_training_root(self.data_root)
        self._is_pfus1_advanced = is_pfus1_advanced_training_root(self.data_root)
        self._is_pfus1_family = self._is_pfus1 or self._is_pfus1_advanced
        self._embedding_dir: Optional[Path] = None
        if image_embedding_dir is not None:
            if not self._is_pfus1_family:
                raise ValueError(
                    "image_embedding_dir is only supported when data_root resolves to "
                    "data/bladder/pfus1 or data/bladder/pfus1_advanced "
                    "(or resolvable PFUS1 family roots via bapmos.paths)."
                )
            ed = Path(image_embedding_dir)
            if not ed.is_absolute():
                ed = project_root() / ed
            self._embedding_dir = ed.resolve()

        with open(split_file, "r") as f:
            split_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if self._is_pfus1_family:
            images_dir = (
                pfus1_advanced_image_root()
                if self._is_pfus1_advanced
                else pfus1_image_root()
            )
            masks_dir = find_combined_masks_dir(self.data_root)
            bundle_label = "PFUS1-advanced" if self._is_pfus1_advanced else "PFUS1"
            for sample_key in split_lines:
                patient, stem = parse_sample_line(sample_key)
                img_path = images_dir / patient / f"{stem}.png"
                mask_path = masks_dir / f"{patient}_{stem}_combined_mask.png"
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)
                self.sample_ids.append(f"{patient}_{stem}")
            n_fg = len(PFUS1_ORGAN_KEYS)
            organ_line = ", ".join(PFUS1_ORGAN_KEYS)
            print(f"\n{'='*60}")
            print(f"MultiOrganDataset initialized - {bundle_label}")
            print(f"{'='*60}")
            print(f"Split: {split}")
            print(f"Frames: {len(self.image_files)}")
            print(f"Classes: {n_fg + 1} (background + {n_fg} organs)")
            print(f"Organs: {organ_line}")
            print(f"Images: {images_dir}")
            print(f"Masks: {masks_dir}")
            print(f"Split lists: {splits_dir.name}/")
            print(f"{'='*60}\n")
        else:
            images_dir = find_training_images_dir(self.data_root)
            masks_dir = find_combined_masks_dir(self.data_root)
            for img_name in split_lines:
                img_path = images_dir / img_name
                img_id = img_name.replace(".png", "")
                mask_path = masks_dir / f"{img_id}_combined_mask.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                self.image_files.append(img_path)
                self.mask_files.append(mask_path)
                self.sample_ids.append(img_id)
            print(f"\n{'='*60}")
            print("MultiOrganDataset initialized - clinical / simulation bundle")
            print(f"{'='*60}")
            print(f"Split: {split}")
            print(f"Images: {len(self.image_files)}")
            print(f"Split lists: {splits_dir.name}/")
            print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'image': RGB image as numpy array (H, W, 3), uint8
                - 'mask': Multi-class mask (H, W), uint8 {0, 1, 2, 3, 4}
                - 'filename': Image filename for tracking
                - 'embedding_path': (PFUS1 only, optional) path to precomputed SAM ``.pt``
                - 'sam_cached_pack': (PFUS1 only, optional) dict with ``image_embedding``, ``original_size``, ``checkpoint``
                  (loaded in DataLoader workers so disk I/O overlaps GPU work)
                - 'has_bladder': Boolean, True if Bladder present
                - 'has_ptv': Boolean, True if PTV present
                - 'has_rectum': Boolean, True if Rectum present
                - 'has_urethra': Boolean, True if Urethra present
        """
        # Load image (grayscale PNG)
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
        
        # Convert to RGB (SAM expects 3-channel input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Load multi-class mask
        mask_path = self.mask_files[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise IOError(f"Failed to load mask: {mask_path}")
        
        if getattr(self, "_is_pfus1_family", False):
            has_flags = pfus1_has_flags(mask)
            filename = self.sample_ids[idx]
        else:
            has_flags = real_clinical_has_flags(mask)
            filename = image_path.name

        sample = {
            "image": image_rgb,
            "mask": mask,
            "filename": filename,
            **has_flags,
        }
        if self._embedding_dir is not None:
            emb_pt = self._embedding_dir / f"{filename}.pt"
            sample["embedding_path"] = str(emb_pt)
            pack = torch.load(emb_pt, map_location="cpu", weights_only=False)
            if "image_embedding" not in pack or "original_size" not in pack:
                raise KeyError(f"Invalid embedding file (missing keys): {emb_pt}")
            sample["sam_cached_pack"] = {
                "image_embedding": pack["image_embedding"],
                "original_size": pack["original_size"],
                "checkpoint": pack.get("checkpoint"),
            }

        if self.transform:
            sample = self.transform(sample)

        return sample


def has_any_organ(mask):
    """
    Check if mask contains any organ pixels (class > 0).
    
    Args:
        mask: (H, W) numpy array with class IDs
    
    Returns:
        bool: True if any organ present, False otherwise
    """
    return (mask > 0).any()


def compute_bounding_box(binary_mask):
    """
    Compute tight bounding box around binary mask.
    
    Args:
        binary_mask: (H, W) numpy array, values in {0, 1}
    
    Returns:
        box: [x_min, y_min, x_max, y_max] or None if mask is empty
    """
    if binary_mask.sum() == 0:
        return None
    
    # Find coordinates of non-zero pixels
    rows, cols = np.where(binary_mask > 0)
    
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    # SAM format: [x_min, y_min, x_max, y_max]
    box = [x_min, y_min, x_max, y_max]
    
    return box


def compute_union_bounding_box(multi_class_mask):
    """
    Compute bounding box around ALL organs (union of all non-background pixels).
    
    Args:
        multi_class_mask: (H, W) numpy array with values {0, 1, 2, 3, 4}
    
    Returns:
        box: [x_min, y_min, x_max, y_max] or None if no organs present
    """
    # Create binary mask of all organs (any non-zero value)
    organs_mask = (multi_class_mask > 0).astype(np.uint8)
    
    return compute_bounding_box(organs_mask)


def compute_per_organ_boxes(
    multi_class_mask,
    organ_to_class: Optional[Dict[str, int]] = None,
):
    """
    Compute separate bounding boxes for each organ.
    
    Args:
        multi_class_mask: (H, W) numpy array with class IDs (layout depends on organ_to_class)
        organ_to_class: Maps organ key to class id (default: real clinical four organs)
    
    Returns:
        dict: Maps organ name to box [x_min, y_min, x_max, y_max] or None
    """
    if organ_to_class is None:
        organ_to_class = REAL_CLINICAL_ORGAN_TO_CLASS
    boxes = {}
    for organ_name, class_id in organ_to_class.items():
        organ_mask = (multi_class_mask == class_id).astype(np.uint8)
        boxes[organ_name] = compute_bounding_box(organ_mask)
    
    return boxes


def sample_positive_points(binary_mask, num_points=1, rng=None):
    """
    Sample random positive points from binary mask.
    
    Args:
        binary_mask: (H, W) numpy array, values in {0, 1}
        num_points: Number of points to sample
        rng: numpy random generator for determinism (if None, uses np.random)
    
    Returns:
        points: numpy array of shape (num_points, 2) in [x, y] format
                or None if mask is empty
    """
    if rng is None:
        rng = np.random.default_rng()
    
    coords = np.argwhere(binary_mask > 0)  # (N, 2) in [y, x]
    
    if len(coords) == 0:
        return None
    
    if len(coords) < num_points:
        # If not enough positive pixels, sample with replacement
        indices = rng.choice(len(coords), size=num_points, replace=True)
    else:
        indices = rng.choice(len(coords), size=num_points, replace=False)
    
    sampled_coords = coords[indices]
    
    # Convert to [x, y] format
    points = sampled_coords[:, [1, 0]]  # Swap columns: [y, x] -> [x, y]
    
    return points


def sample_multi_organ_points(multi_class_mask, num_points_per_organ=1):
    """
    Sample positive points from each organ present in the mask.
    
    Args:
        multi_class_mask: (H, W) numpy array with class IDs {0,1,2,3,4}
        num_points_per_organ: Number of points to sample per organ
    
    Returns:
        dict: Maps organ name to points array (N, 2) in [x, y] format
              Keys: 'bladder', 'ptv', 'rectum', 'urethra'
              Values are None if organ not present
    """
    points_dict = {}
    for organ_name, class_id in REAL_CLINICAL_ORGAN_TO_CLASS.items():
        organ_mask = (multi_class_mask == class_id).astype(np.uint8)
        points_dict[organ_name] = sample_positive_points(
            organ_mask, 
            num_points=num_points_per_organ
        )
    
    return points_dict


def sample_negative_points_from_ring(organ_mask, ring_width=20, num_neg=3, rng=None):
    """
    Sample negative points from ring around organ mask.
    
    Args:
        organ_mask: (H, W) binary mask of the organ
        ring_width: Width of ring in pixels
        num_neg: Number of negative points to sample
        rng: numpy random generator for determinism (if None, uses np.random)
    
    Returns:
        numpy array (num_neg, 2) in [x, y] format, or None if ring is empty
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Dilate organ mask to create outer boundary
    # Use (2*ring_width+1, 2*ring_width+1) for proper ring width
    k = 2 * ring_width + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(organ_mask, kernel, iterations=1)
    
    # Ring = dilated - original
    ring = dilated - organ_mask
    
    if ring.sum() == 0:
        return None
    
    # Sample from ring
    coords = np.argwhere(ring > 0)  # (N, 2) in [y, x]
    
    if len(coords) == 0:
        return None
    
    if len(coords) < num_neg:
        indices = rng.choice(len(coords), size=num_neg, replace=True)
    else:
        indices = rng.choice(len(coords), size=num_neg, replace=False)
    
    sampled_coords = coords[indices]
    
    # Convert to [x, y] format
    points = sampled_coords[:, [1, 0]]
    
    return points


def sample_per_organ_points_with_negatives(
    multi_class_mask,
    num_pos_per_organ=1,
    num_neg=3,
    ring_width=20,
    rng=None,
    organ_to_class: Optional[Dict[str, int]] = None,
):
    """
    Sample positive + negative points separately for each organ.
    Negatives are sampled from a ring around EACH organ independently.
    
    Args:
        multi_class_mask: (H, W) numpy array with class IDs (layout depends on organ_to_class)
        num_pos_per_organ: Number of positive points per organ
        num_neg: Number of negative points per organ (from ring around that organ)
        ring_width: Width of ring around each organ for negative sampling
        rng: numpy random generator for determinism (if None, uses np.random)
        organ_to_class: Maps organ key to class id (default: real clinical four organs)
    
    Returns:
        dict keyed by organ name (same keys as organ_to_class).
        Each value is a dict with 'points', 'labels', or None if organ not present.
    """
    if rng is None:
        rng = np.random.default_rng()
    if organ_to_class is None:
        organ_to_class = REAL_CLINICAL_ORGAN_TO_CLASS
    
    point_sets = {}
    class_to_key = {cid: key for key, cid in organ_to_class.items()}

    for class_id in sorted(class_to_key.keys()):
        organ_name = class_to_key[class_id]
        organ_mask = (multi_class_mask == class_id).astype(np.uint8)
        
        if organ_mask.sum() == 0:
            point_sets[organ_name] = None
        else:
            # Sample positive points
            pos_points = sample_positive_points(organ_mask, num_pos_per_organ, rng=rng)
            if pos_points is None:
                point_sets[organ_name] = None
                continue
            
            # Sample negative points from ring
            neg_points = sample_negative_points_from_ring(organ_mask, ring_width, num_neg, rng=rng)
            
            if neg_points is not None:
                # Combine positive and negative
                all_points = np.vstack([pos_points, neg_points])
                all_labels = np.concatenate([
                    np.ones(len(pos_points), dtype=np.int32),
                    np.zeros(len(neg_points), dtype=np.int32)
                ])
            else:
                # Only positive points
                all_points = pos_points
                all_labels = np.ones(len(pos_points), dtype=np.int32)
            
            point_sets[organ_name] = {
                'points': all_points,
                'labels': all_labels
            }
    
    return point_sets
