"""
Per-Organ UCB1 Multi-Armed Bandit for Adaptive Prompt Selection

Maintains separate UCB1 bandits for each organ, allowing organ-specific
prompt strategy learning. Each organ independently learns which prompt
type (box, point, or both) works best for its unique characteristics.

Scientific Rationale:
    Different organs have different characteristics:
    - Size: PTV1 (large tumor target) vs Bladder (variable size)
    - Shape: Rectum (tubular) vs Bladder (spherical) vs PTV1 (irregular)
    - Boundary clarity: OARs may have different edge characteristics than tumor
    - Clinical importance: Different precision requirements for PTV vs OARs
    
    Per-organ bandits enable each organ to discover its optimal strategy
    independently, rather than forcing a one-size-fits-all approach.

Architecture:
    - One UCB1Bandit instance per organ
    - Independent learning: organ decisions don't interfere
    - Aggregated statistics for overall monitoring
    - Reuses proven UCB1Bandit implementation

Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). 
Finite-time analysis of the multi-armed bandit problem. 
Machine Learning, 47(2-3), 235-256.
"""

import numpy as np
from typing import List, Dict, Any
from ..ucb1_global_bandit import UCB1Bandit


class UCB1PerOrganBandit:
    """
    Per-organ UCB1 bandit for adaptive prompt selection.
    
    Maintains a separate UCB1Bandit instance for each organ, allowing
    organ-specific learning and adaptation. Each organ independently
    selects arms and receives rewards based on its validation performance.
    
    Arms:
        - "box": Box prompts only
        - "point": Point prompts only
        - "both": Box + Point prompts simultaneously
    
    Design:
        - Independent bandits: organ-specific learning
        - Shared hyperparameters: fair comparison across organs
        - Block-level decisions: arm held for full training block
        - Reward = -MSD (absolute quality, not improvement)
    
    Args:
        organs (List[str]): List of organ names (e.g., ["rectum", "bladder", "ptv1"])
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        exploration_constant (float): UCB exploration parameter c (default: 2.0)
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        organs: List[str],
        arms: List[str] = None,
        exploration_constant: float = 2.0,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        assert len(organs) > 0, "Must provide at least one organ"
        
        if arms is None:
            arms = ["box", "point", "both"]
        
        self.organs = organs
        self.n_organs = len(organs)
        self.arms = arms
        self.n_arms = len(arms)
        
        # Hyperparameters (shared across all organs)
        self.exploration_constant = exploration_constant
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # Create separate bandit for each organ
        self.bandits = {}
        for i, organ in enumerate(organs):
            # Use different seed for each organ to avoid synchronization
            organ_seed = seed + i
            self.bandits[organ] = UCB1Bandit(
                arms=arms,
                exploration_constant=exploration_constant,
                warmup_blocks=warmup_blocks,
                min_pulls_per_arm=min_pulls_per_arm,
                reward_clip_max=reward_clip_max,
                seed=organ_seed
            )
        
        # Global state
        self.total_pulls = 0  # Total across all organs
        self.current_arms = {organ: None for organ in organs}
    
    def select_arm(self, organ: str) -> str:
        """
        Select an arm for the specified organ using its UCB1 bandit.
        
        Args:
            organ (str): Organ name
        
        Returns:
            str: Selected arm name
        """
        assert organ in self.organs, f"Unknown organ: {organ}"
        
        selected_arm = self.bandits[organ].select_arm()
        self.current_arms[organ] = selected_arm
        self.total_pulls += 1
        
        return selected_arm
    
    def update_reward(self, organ: str, arm: str, val_metric: float):
        """
        Update reward for the specified organ's bandit.
        
        Args:
            organ (str): Organ name
            arm (str): Arm that was pulled
            val_metric (float): Validation metric (e.g., MSD in mm)
        """
        assert organ in self.organs, f"Unknown organ: {organ}"
        
        self.bandits[organ].update_reward(arm, val_metric)
    
    def get_organ_statistics(self, organ: str) -> Dict[str, Any]:
        """
        Get statistics for a specific organ's bandit.
        
        Args:
            organ (str): Organ name
        
        Returns:
            dict: Bandit statistics for this organ
        """
        assert organ in self.organs, f"Unknown organ: {organ}"
        return self.bandits[organ].get_statistics()
    
    def get_best_arm(self, organ: str) -> str:
        """
        Get best arm for a specific organ.
        
        Args:
            organ (str): Organ name
        
        Returns:
            str: Best arm name for this organ
        """
        assert organ in self.organs, f"Unknown organ: {organ}"
        return self.bandits[organ].get_best_arm()
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all organs.
        
        Notes:
            - total_pulls: Sum of all per-organ arm selections (not number of blocks)
            - arm_selection_rates: Computed over all organ-level pulls 
              (organ_arm_count / total_pulls across all organs)
        
        Returns:
            dict: {
                "total_pulls": int,  # Total pulls across all organs
                "per_organ": dict,   # Per-organ statistics
                "aggregated": dict   # Aggregated metrics
            }
        """
        per_organ = {}
        for organ in self.organs:
            per_organ[organ] = self.get_organ_statistics(organ)
        
        # Aggregate arm counts across all organs
        aggregated_arm_counts = {arm: 0 for arm in self.arms}
        aggregated_arm_rewards = {arm: [] for arm in self.arms}
        
        for organ in self.organs:
            stats = per_organ[organ]
            for arm in self.arms:
                aggregated_arm_counts[arm] += stats["arm_counts"][arm]
                # Collect all rewards for this arm across organs
                if len(self.bandits[organ].arm_rewards[arm]) > 0:
                    aggregated_arm_rewards[arm].extend(self.bandits[organ].arm_rewards[arm])
        
        # Compute aggregated average rewards
        aggregated_avg_rewards = {}
        for arm in self.arms:
            if len(aggregated_arm_rewards[arm]) > 0:
                aggregated_avg_rewards[arm] = np.mean(aggregated_arm_rewards[arm])
            else:
                aggregated_avg_rewards[arm] = 0.0
        
        # Aggregated selection rates
        if self.total_pulls > 0:
            aggregated_selection_rates = {
                arm: count / self.total_pulls
                for arm, count in aggregated_arm_counts.items()
            }
        else:
            aggregated_selection_rates = {arm: 0.0 for arm in self.arms}
        
        return {
            "total_pulls": self.total_pulls,
            "per_organ": per_organ,
            "aggregated": {
                "arm_counts": aggregated_arm_counts,
                "arm_avg_rewards": aggregated_avg_rewards,
                "arm_selection_rates": aggregated_selection_rates,
                "total_blocks": self.total_pulls  # For trainer compatibility
            }
        }
    
    def get_best_arms_per_organ(self) -> Dict[str, str]:
        """
        Get the best arm for each organ.
        
        Returns:
            dict: {organ_name: best_arm_name}
        """
        return {organ: self.get_best_arm(organ) for organ in self.organs}
    
    def __repr__(self):
        return (
            f"UCB1PerOrganBandit("
            f"organs={len(self.organs)}, "
            f"arms={self.arms}, "
            f"c={self.exploration_constant}, "
            f"warmup_blocks={self.warmup_blocks}, "
            f"total_pulls={self.total_pulls})"
        )
