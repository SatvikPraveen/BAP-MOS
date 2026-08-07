"""
Phase 1: Box vs Point Prompt Sampler

Mutually exclusive per sample:
- box_ratio: probability of selecting box-only prompt
- point_ratio: probability of selecting point-only prompt
- Each sample gets EITHER box OR point (never both)

Example: box_ratio=0.7, point_ratio=0.3
    → 70% of samples get box prompts
    → 30% of samples get point prompts
"""

import hashlib
import numpy as np
from typing import Dict, Tuple, Optional, List


class BoxPointSampler:
    """
    Samples prompt type (box or point) for each training sample.
    
    Args:
        box_ratio (float): Probability of selecting box prompt (0 to 1)
        point_ratio (float): Probability of selecting point prompt (0 to 1)
        seed (int): Random seed for reproducibility
    
    Note:
        box_ratio + point_ratio must equal 1.0
    """
    
    def __init__(self, box_ratio: float, point_ratio: float, seed: int = 0):
        # Validate ratios
        assert 0 <= box_ratio <= 1, "box_ratio must be between 0 and 1"
        assert 0 <= point_ratio <= 1, "point_ratio must be between 0 and 1"
        assert abs(box_ratio + point_ratio - 1.0) < 1e-6, "box_ratio + point_ratio must equal 1.0"
        
        self.box_ratio = box_ratio
        self.point_ratio = point_ratio
        self.seed = seed
        
        # Create RNG for deterministic sampling
        self.rng = np.random.default_rng(seed)
        
        # Statistics tracking
        self.counts = {"box": 0, "point": 0}
        self.total_samples = 0
    
    def sample_prompt_type(self, sample_id: str, epoch: int = 0) -> str:
        """
        Sample prompt type for a given sample.
        
        Args:
            sample_id (str): Unique identifier for the sample (e.g., filename)
            epoch (int): Current epoch (for deterministic epoch-specific randomness)
        
        Returns:
            str: Either "box" or "point"
        """
        # Create deterministic seed based on sample_id and epoch
        seed_str = f"{sample_id}_epoch{epoch}_seed{self.seed}"
        # Use stable hash (MD5) instead of process-salted hash() for reproducibility
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        local_rng = np.random.default_rng(hash_val)
        
        # Sample prompt type
        if local_rng.random() < self.box_ratio:
            prompt_type = "box"
        else:
            prompt_type = "point"
        
        # Update statistics
        self.counts[prompt_type] += 1
        self.total_samples += 1
        
        return prompt_type
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about prompt sampling.
        
        Returns:
            dict: {
                "box_count": int,
                "point_count": int,
                "total": int,
                "box_ratio_actual": float,
                "point_ratio_actual": float
            }
        """
        if self.total_samples == 0:
            return {
                "box_count": 0,
                "point_count": 0,
                "total": 0,
                "box_ratio_actual": 0.0,
                "point_ratio_actual": 0.0
            }
        
        return {
            "box_count": self.counts["box"],
            "point_count": self.counts["point"],
            "total": self.total_samples,
            "box_ratio_actual": self.counts["box"] / self.total_samples,
            "point_ratio_actual": self.counts["point"] / self.total_samples
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.counts = {"box": 0, "point": 0}
        self.total_samples = 0
    
    def __repr__(self):
        return (
            f"BoxPointSampler("
            f"box_ratio={self.box_ratio:.2f}, "
            f"point_ratio={self.point_ratio:.2f}, "
            f"seed={self.seed})"
        )


if __name__ == "__main__":
    # Test the sampler
    sampler = BoxPointSampler(box_ratio=0.7, point_ratio=0.3, seed=42)
    
    # Sample 1000 prompts
    for i in range(1000):
        prompt_type = sampler.sample_prompt_type(f"sample_{i}", epoch=0)
    
    # Check statistics
    stats = sampler.get_statistics()
    print("\nBoxPointSampler Test:")
    print(f"Expected: box=0.70, point=0.30")
    print(f"Actual:   box={stats['box_ratio_actual']:.3f}, point={stats['point_ratio_actual']:.3f}")
    print(f"Counts:   box={stats['box_count']}, point={stats['point_count']}, total={stats['total']}")
