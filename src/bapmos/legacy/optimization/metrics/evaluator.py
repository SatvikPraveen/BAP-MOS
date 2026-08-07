"""
Metrics Evaluator: Per-Slice Computation and Aggregation

Workflow:
1. Compute metrics per organ per slice (2D)
2. Aggregate across slices using:
   - mean ± std
   - median [IQR]
   - worst-10% (tail) for boundary metrics
3. Handle empty masks appropriately
4. Export to CSV for analysis

Distance units follow ``pixel_spacing`` (mm for DICOM-derived clinical/simulation;
pixel-native when spacing is ``(1.0, 1.0)`` e.g. PFUS1).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from .boundary_metrics import compute_all_metrics


class MetricsEvaluator:
    """
    Evaluator for boundary-aware metrics aggregation.
    
    Args:
        pixel_spacing (Tuple[float, float]): Pixel spacing in mm/pixel (x, y)
        organs (List[str]): List of organ names to evaluate
    """
    
    def __init__(
        self,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        organs: List[str] = None
    ):
        if organs is None:
            organs = ["Rectum", "Bladder", "PTV1"]
        
        self.pixel_spacing = pixel_spacing
        self.organs = organs
        
        # Storage for per-slice metrics
        self.per_slice_metrics = []  # List of dicts
    
    def evaluate_slice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        organ_name: str,
        slice_idx: int,
        image_id: str
    ):
        """
        Evaluate metrics for a single slice and organ.
        
        Args:
            pred_mask: (H, W) binary prediction mask {0, 1}
            gt_mask: (H, W) binary ground truth mask {0, 1}
            organ_name: Name of organ
            slice_idx: Slice index (for tracking)
            image_id: Image identifier (filename)
        """
        # Compute all metrics
        metrics = compute_all_metrics(
            pred_mask, gt_mask, self.pixel_spacing, organ_name
        )
        
        # Add metadata
        metrics["slice_idx"] = slice_idx
        metrics["image_id"] = image_id
        
        # Store
        self.per_slice_metrics.append(metrics)
    
    def evaluate_multiclass_slice(
        self,
        pred_multiclass: np.ndarray,
        gt_multiclass: np.ndarray,
        slice_idx: int,
        image_id: str,
        class_mapping: Dict[int, str] = None
    ):
        """
        Evaluate metrics for a multi-class segmentation slice.
        
        Args:
            pred_multiclass: (H, W) multi-class prediction {0, 1, 2, 3}
            gt_multiclass: (H, W) multi-class ground truth {0, 1, 2, 3}
            slice_idx: Slice index
            image_id: Image identifier
            class_mapping: Mapping from class ID to organ name
                Default: {1: "Rectum", 2: "Bladder", 3: "PTV1"}
        """
        if class_mapping is None:
            class_mapping = {1: "Rectum", 2: "Bladder", 3: "PTV1"}
        
        # Evaluate each organ separately
        for class_id, organ_name in class_mapping.items():
            pred_mask = (pred_multiclass == class_id).astype(np.uint8)
            gt_mask = (gt_multiclass == class_id).astype(np.uint8)
            
            self.evaluate_slice(
                pred_mask, gt_mask, organ_name, slice_idx, image_id
            )
    
    def aggregate_metrics(self, organ_name: Optional[str] = None) -> Dict:
        """
        Aggregate per-slice metrics across slices.
        
        Args:
            organ_name: Specific organ to aggregate (None = all organs)
        
        Returns:
            dict: {
                "organ": str,
                "n_slices": int,
                "n_valid_boundary": int,
                "empty_pred_rate": float,
                "false_positive_rate": float,
                
                # Primary metrics (boundary-aware)
                "hd95_mm_mean": float,
                "hd95_mm_std": float,
                "hd95_mm_median": float,
                "hd95_mm_iqr": Tuple[float, float],
                "hd95_mm_worst10": float,
                
                "msd_mm_mean": float,
                "msd_mm_std": float,
                "msd_mm_median": float,
                "msd_mm_iqr": Tuple[float, float],
                "msd_mm_worst10": float,
                
                # Secondary metrics (reporting only)
                "dice_mean": float,
                "dice_std": float,
                "dice_median": float,
                
                "iou_mean": float,
                "iou_std": float,
                "iou_median": float
            }
        """
        # Filter metrics by organ
        if organ_name is not None:
            metrics_list = [
                m for m in self.per_slice_metrics if m["organ"] == organ_name
            ]
        else:
            metrics_list = self.per_slice_metrics
        
        if len(metrics_list) == 0:
            return None
        
        # Extract arrays
        valid_boundary = [m for m in metrics_list if m["valid_boundary"]]
        
        hd95_values = [m["hd95_mm"] for m in valid_boundary if m["hd95_mm"] is not None]
        msd_values = [m["msd_mm"] for m in valid_boundary if m["msd_mm"] is not None]
        
        dice_values = [m["dice"] for m in metrics_list]
        iou_values = [m["iou"] for m in metrics_list]
        
        # Compute failure rates
        n_slices = len(metrics_list)
        n_valid_boundary = len(valid_boundary)
        n_empty_pred = sum(m["empty_pred_failure"] for m in metrics_list)
        n_false_positive = sum(m["false_positive_empty_gt"] for m in metrics_list)
        
        empty_pred_rate = n_empty_pred / n_slices if n_slices > 0 else 0.0
        false_positive_rate = n_false_positive / n_slices if n_slices > 0 else 0.0
        
        # Aggregate boundary metrics
        result = {
            "organ": organ_name if organ_name else "all",
            "n_slices": n_slices,
            "n_valid_boundary": n_valid_boundary,
            "empty_pred_rate": empty_pred_rate,
            "false_positive_rate": false_positive_rate
        }
        
        # HD95
        if len(hd95_values) > 0:
            result["hd95_mm_mean"] = float(np.mean(hd95_values))
            result["hd95_mm_std"] = float(np.std(hd95_values))
            result["hd95_mm_median"] = float(np.median(hd95_values))
            result["hd95_mm_iqr"] = (
                float(np.percentile(hd95_values, 25)),
                float(np.percentile(hd95_values, 75))
            )
            result["hd95_mm_worst10"] = float(np.percentile(hd95_values, 90))
        else:
            result["hd95_mm_mean"] = None
            result["hd95_mm_std"] = None
            result["hd95_mm_median"] = None
            result["hd95_mm_iqr"] = (None, None)
            result["hd95_mm_worst10"] = None
        
        # MSD
        if len(msd_values) > 0:
            result["msd_mm_mean"] = float(np.mean(msd_values))
            result["msd_mm_std"] = float(np.std(msd_values))
            result["msd_mm_median"] = float(np.median(msd_values))
            result["msd_mm_iqr"] = (
                float(np.percentile(msd_values, 25)),
                float(np.percentile(msd_values, 75))
            )
            result["msd_mm_worst10"] = float(np.percentile(msd_values, 90))
        else:
            result["msd_mm_mean"] = None
            result["msd_mm_std"] = None
            result["msd_mm_median"] = None
            result["msd_mm_iqr"] = (None, None)
            result["msd_mm_worst10"] = None
        
        # Dice (secondary)
        result["dice_mean"] = float(np.mean(dice_values))
        result["dice_std"] = float(np.std(dice_values))
        result["dice_median"] = float(np.median(dice_values))
        
        # IoU (secondary)
        result["iou_mean"] = float(np.mean(iou_values))
        result["iou_std"] = float(np.std(iou_values))
        result["iou_median"] = float(np.median(iou_values))
        
        return result
    
    def export_per_slice_csv(self, output_path: Path):
        """
        Export per-slice metrics to CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        df = pd.DataFrame(self.per_slice_metrics)
        
        # Reorder columns for readability
        column_order = [
            "image_id", "slice_idx", "organ",
            "hd95_mm", "msd_mm", "boundary_dist_mm",
            "dice", "iou",
            "gt_empty", "pred_empty", "valid_boundary",
            "false_positive_empty_gt", "empty_pred_failure"
        ]
        
        # Keep only existing columns
        column_order = [c for c in column_order if c in df.columns]
        df = df[column_order]
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Per-slice metrics saved to: {output_path}")
    
    def export_summary_csv(self, output_path: Path):
        """
        Export aggregated summary metrics to CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        # Aggregate for each organ
        summaries = []
        
        for organ in self.organs:
            agg = self.aggregate_metrics(organ)
            if agg is not None:
                summaries.append(agg)
        
        # Also add overall aggregation
        overall = self.aggregate_metrics(organ_name=None)
        if overall is not None:
            summaries.append(overall)
        
        df = pd.DataFrame(summaries)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Summary metrics saved to: {output_path}")
    
    def export_failure_analysis_csv(self, output_path: Path, top_n: int = 20):
        """
        Export worst slices by HD95 for failure analysis.
        
        Args:
            output_path: Path to output CSV file
            top_n: Number of worst slices to export per organ
        """
        failure_cases = []
        
        for organ in self.organs:
            # Filter by organ
            organ_metrics = [
                m for m in self.per_slice_metrics
                if m["organ"] == organ and m["valid_boundary"]
            ]
            
            # Sort by HD95 (descending)
            organ_metrics_sorted = sorted(
                organ_metrics,
                key=lambda x: x["hd95_mm"] if x["hd95_mm"] is not None else -1,
                reverse=True
            )
            
            # Take top N
            worst_cases = organ_metrics_sorted[:top_n]
            
            for case in worst_cases:
                failure_cases.append({
                    "organ": case["organ"],
                    "image_id": case["image_id"],
                    "slice_idx": case["slice_idx"],
                    "hd95_mm": case["hd95_mm"],
                    "msd_mm": case["msd_mm"],
                    "dice": case["dice"],
                    "iou": case["iou"]
                })
        
        df = pd.DataFrame(failure_cases)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Failure analysis saved to: {output_path}")
    
    def reset(self):
        """Clear all stored metrics."""
        self.per_slice_metrics = []


if __name__ == "__main__":
    # Test the evaluator
    print("\n=== Metrics Evaluator Test ===\n")
    
    evaluator = MetricsEvaluator(
        pixel_spacing=(0.5, 0.5),
        organs=["Rectum", "Bladder", "PTV1"]
    )
    
    # Simulate evaluating 10 slices for Rectum
    np.random.seed(42)
    for i in range(10):
        # Create fake masks
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[30:70, 30:70] = 1
        
        # Add some noise to prediction
        pred = gt.copy()
        if i > 0:  # Keep first one perfect
            noise = np.random.randint(-5, 5, size=2)
            pred = np.roll(pred, noise, axis=(0, 1))
        
        evaluator.evaluate_slice(pred, gt, "Rectum", slice_idx=i, image_id=f"slice_{i:03d}")
    
    # Aggregate
    summary = evaluator.aggregate_metrics("Rectum")
    
    print("Rectum metrics (10 slices):")
    print(f"  Valid boundary slices: {summary['n_valid_boundary']}/{summary['n_slices']}")
    print(f"  MSD: {summary['msd_mm_mean']:.3f} ± {summary['msd_mm_std']:.3f} mm")
    print(f"  HD95: {summary['hd95_mm_mean']:.3f} ± {summary['hd95_mm_std']:.3f} mm")
    print(f"  Dice: {summary['dice_mean']:.3f} ± {summary['dice_std']:.3f}")
    print(f"  Empty pred rate: {summary['empty_pred_rate']:.1%}")
