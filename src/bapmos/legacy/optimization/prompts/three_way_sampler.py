"""
Phase 2: Three-way Prompt Sampler (Box + Point + Both)

Categorical sampling with three modes:
- Box-only: Only box prompts
- Point-only: Only point prompts  
- Both: Box AND point prompts simultaneously

Weights control relative frequency of each mode.

Example: box_weight=1, point_weight=1, both_weight=1
    → 33.3% box-only, 33.3% point-only, 33.3% both
"""

import hashlib
import numpy as np
from typing import Dict, List


class ThreeWaySampler:
    """
    Samples prompt mode (box, point, or both) for each training sample.
    
    Args:
        box_weight (float): Relative weight for box-only prompts
        point_weight (float): Relative weight for point-only prompts
        both_weight (float): Relative weight for both prompts simultaneously
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, box_weight: float, point_weight: float, both_weight: float, seed: int = 0):
        # Validate weights
        assert box_weight > 0, "box_weight must be positive"
        assert point_weight > 0, "point_weight must be positive"
        assert both_weight > 0, "both_weight must be positive"
        
        # Normalize weights to probabilities
        total = box_weight + point_weight + both_weight
        self.box_prob = box_weight / total
        self.point_prob = point_weight / total
        self.both_prob = both_weight / total
        
        # Store original weights for logging
        self.box_weight = box_weight
        self.point_weight = point_weight
        self.both_weight = both_weight
        self.seed = seed
        
        # Create RNG for deterministic sampling
        self.rng = np.random.default_rng(seed)
        
        # Statistics tracking
        self.counts = {"box": 0, "point": 0, "both": 0}
        self.total_samples = 0
    
    def sample_prompt_mode(self, sample_id: str, epoch: int = 0) -> str:
        """
        Sample prompt mode for a given sample.
        
        Args:
            sample_id (str): Unique identifier for the sample (e.g., filename)
            epoch (int): Current epoch (for deterministic epoch-specific randomness)
        
        Returns:
            str: One of "box", "point", or "both"
        """
        # Create deterministic seed based on sample_id and epoch
        seed_str = f"{sample_id}_epoch{epoch}_seed{self.seed}"
        # Use stable hash (MD5) instead of process-salted hash() for reproducibility
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        local_rng = np.random.default_rng(hash_val)
        
        # Sample prompt mode using categorical distribution
        rand_val = local_rng.random()
        
        if rand_val < self.box_prob:
            mode = "box"
        elif rand_val < self.box_prob + self.point_prob:
            mode = "point"
        else:
            mode = "both"
        
        # Update statistics
        self.counts[mode] += 1
        self.total_samples += 1
        
        return mode
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about prompt sampling.
        
        Returns:
            dict: {
                "box_count": int,
                "point_count": int,
                "both_count": int,
                "total": int,
                "box_ratio_actual": float,
                "point_ratio_actual": float,
                "both_ratio_actual": float
            }
        """
        if self.total_samples == 0:
            return {
                "box_count": 0,
                "point_count": 0,
                "both_count": 0,
                "total": 0,
                "box_ratio_actual": 0.0,
                "point_ratio_actual": 0.0,
                "both_ratio_actual": 0.0
            }
        
        return {
            "box_count": self.counts["box"],
            "point_count": self.counts["point"],
            "both_count": self.counts["both"],
            "total": self.total_samples,
            "box_ratio_actual": self.counts["box"] / self.total_samples,
            "point_ratio_actual": self.counts["point"] / self.total_samples,
            "both_ratio_actual": self.counts["both"] / self.total_samples
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.counts = {"box": 0, "point": 0, "both": 0}
        self.total_samples = 0
    
    def __repr__(self):
        return (
            f"ThreeWaySampler("
            f"weights=[{self.box_weight}:{self.point_weight}:{self.both_weight}], "
            f"probs=[box={self.box_prob:.3f}, point={self.point_prob:.3f}, both={self.both_prob:.3f}], "
            f"seed={self.seed})"
        )


if __name__ == "__main__":
    # Test the sampler with balanced ratio
    print("\n=== Test 1: Balanced (1:1:1) ===")
    sampler1 = ThreeWaySampler(box_weight=1, point_weight=1, both_weight=1, seed=42)
    
    for i in range(1000):
        mode = sampler1.sample_prompt_mode(f"sample_{i}", epoch=0)
    
    stats1 = sampler1.get_statistics()
    print(f"Expected: box=0.333, point=0.333, both=0.333")
    print(f"Actual:   box={stats1['box_ratio_actual']:.3f}, "
          f"point={stats1['point_ratio_actual']:.3f}, "
          f"both={stats1['both_ratio_actual']:.3f}")
    
    # Test with 2:3:5 ratio
    print("\n=== Test 2: Prof-style (2:3:5) ===")
    sampler2 = ThreeWaySampler(box_weight=2, point_weight=3, both_weight=5, seed=42)
    
    for i in range(1000):
        mode = sampler2.sample_prompt_mode(f"sample_{i}", epoch=0)
    
    stats2 = sampler2.get_statistics()
    print(f"Expected: box=0.200, point=0.300, both=0.500")
    print(f"Actual:   box={stats2['box_ratio_actual']:.3f}, "
          f"point={stats2['point_ratio_actual']:.3f}, "
          f"both={stats2['both_ratio_actual']:.3f}")
