"""
Phase 3: UCB1 Multi-Armed Bandit for Adaptive Prompt Selection

Adaptively selects prompt strategy (box, point, or both) based on
validation performance (MSD).

Algorithm: UCB1 (Upper Confidence Bound)
- Balances exploration vs exploitation
- Block-level decision making: arm held fixed for eval_frequency batches
- Reward signal: negative validation MSD (absolute quality, not improvement)
- Deterministic validation probe set to reduce noise

Key Design:
- Arm → Training Block → Validation → Reward (clean causal chain)
- Organ-balanced reward computation (equal weight per organ)
- Guardrails: reward clipping, minimum pulls per arm

Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). 
Finite-time analysis of the multi-armed bandit problem. 
Machine Learning, 47(2-3), 235-256.
"""

import numpy as np
from typing import List, Dict
import math


class UCB1Bandit:
    """
    UCB1 (Upper Confidence Bound) algorithm for block-level prompt selection.
    
    Arms:
        - "box": Box prompts only
        - "point": Point prompts only
        - "both": Box + Point prompts simultaneously
    
    Design:
        - Arm is held fixed for an entire evaluation block (e.g., 50 batches)
        - After block completion, validation is run and reward is computed
        - Reward = -MSD (absolute quality, not improvement-based)
        - Deterministic validation probe set ensures low-noise rewards
    
    Args:
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        exploration_constant (float): UCB exploration parameter c (default: 2.0)
        warmup_blocks (int): Number of training blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm even after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping reward (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        arms: List[str] = None,
        exploration_constant: float = 2.0,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        # Default arms
        if arms is None:
            arms = ["box", "point", "both"]
        
        assert len(arms) > 0, "Must have at least one arm"
        assert exploration_constant > 0, "exploration_constant must be positive"
        assert warmup_blocks >= 0, "warmup_blocks must be non-negative"
        assert min_pulls_per_arm >= 0, "min_pulls_per_arm must be non-negative"
        
        self.arms = arms
        self.n_arms = len(arms)
        self.exploration_constant = exploration_constant
        self.warmup_blocks = warmup_blocks  # Blocks per arm during warmup
        self.min_pulls_per_arm = min_pulls_per_arm  # Minimum guaranteed pulls
        self.reward_clip_max = reward_clip_max  # Clip MSD to prevent noise spikes
        self.seed = seed
        
        # State tracking
        self.arm_counts = {arm: 0 for arm in arms}  # N_a(t)
        self.arm_rewards = {arm: [] for arm in arms}  # List of rewards per arm
        self.arm_avg_rewards = {arm: 0.0 for arm in arms}  # Average reward per arm
        
        self.total_pulls = 0
        self.current_arm = None
        
        # History
        self.selection_history = []  # (pull_idx, arm_name)
        self.reward_history = []  # (pull_idx, arm_name, reward)
        
        # RNG for warmup uniform sampling
        self.rng = np.random.default_rng(seed)
    
    def select_arm(self) -> str:
        """
        Select an arm using UCB1 algorithm.
        
        During warmup (first warmup_blocks * n_arms pulls):
            - Uniform round-robin exploration
        
        After warmup:
            - Enforce minimum pulls per arm
            - UCB1: arm* = argmax_a [ Q(a) + c * sqrt(ln(t) / N_a(t)) ]
              where:
                Q(a) = average reward for arm a
                N_a(t) = number of times arm a was pulled
                t = total number of pulls
                c = exploration constant
        
        Returns:
            str: Selected arm name
        """
        # Warmup phase: uniform round-robin
        total_warmup_pulls = self.warmup_blocks * self.n_arms
        if self.total_pulls < total_warmup_pulls:
            # Determine which arm to pull next in round-robin
            arm_idx = self.total_pulls % self.n_arms
            selected_arm = self.arms[arm_idx]
        else:
            # Post-warmup: enforce minimum pulls per arm
            arms_below_min = [arm for arm in self.arms if self.arm_counts[arm] < self.min_pulls_per_arm]
            
            if arms_below_min:
                # Force selection of under-explored arm
                selected_arm = arms_below_min[0]  # Could also random.choice()
            else:
                # UCB1 selection
                ucb_values = {}
                
                for arm in self.arms:
                    # Average reward Q(a)
                    Q_a = self.arm_avg_rewards[arm]
                    
                    # Count N_a(t)
                    N_a = self.arm_counts[arm]
                    
                    # UCB term: c * sqrt(ln(t) / N_a(t))
                    if N_a == 0:
                        # Should not happen after warmup, but handle gracefully
                        ucb_term = float('inf')
                    else:
                        # Defensive: ensure t >= 1 to avoid log(0)
                        t = max(self.total_pulls, 1)
                        ucb_term = self.exploration_constant * math.sqrt(
                            math.log(t) / N_a
                        )
                    
                    # UCB value
                    ucb_values[arm] = Q_a + ucb_term
                
                # Select arm with maximum UCB value (random tie-breaking)
                max_value = max(ucb_values.values())
                best_arms = [arm for arm, value in ucb_values.items() if value == max_value]
                
                if len(best_arms) > 1:
                    # Random tie-breaking using seeded RNG
                    selected_arm = self.rng.choice(best_arms)
                else:
                    selected_arm = best_arms[0]
        
        # Update state
        self.current_arm = selected_arm
        self.total_pulls += 1
        self.arm_counts[selected_arm] += 1
        
        # Record history
        self.selection_history.append((self.total_pulls, selected_arm))
        
        return selected_arm
    
    def update_reward(self, arm: str, val_metric: float):
        """
        Update reward for the given arm.
        
        Uses negative MSD as reward (maximize reward = minimize MSD).
        Clips MSD to prevent noise spikes from dominating learning.
        
        Args:
            arm (str): Arm that was pulled
            val_metric (float): Validation metric (e.g., MSD in mm)
        """
        assert arm in self.arms, f"Unknown arm: {arm}"
        
        # Clip MSD to prevent outliers from dominating
        clipped_metric = min(val_metric, self.reward_clip_max)
        
        # Reward = negative metric (maximize reward = minimize metric)
        reward = -clipped_metric
        
        # Update arm statistics
        self.arm_rewards[arm].append(reward)
        self.arm_avg_rewards[arm] = np.mean(self.arm_rewards[arm])
        
        # Record history
        self.reward_history.append((self.total_pulls, arm, reward))
    
    def get_statistics(self) -> Dict:
        """
        Get current bandit statistics.
        
        Returns:
            dict: {
                "total_pulls": int,
                "arm_counts": dict,
                "arm_avg_rewards": dict,
                "arm_selection_rates": dict,
                "current_arm": str
            }
        """
        if self.total_pulls == 0:
            arm_selection_rates = {arm: 0.0 for arm in self.arms}
        else:
            arm_selection_rates = {
                arm: count / self.total_pulls
                for arm, count in self.arm_counts.items()
            }
        
        return {
            "total_pulls": self.total_pulls,
            "arm_counts": self.arm_counts.copy(),
            "arm_avg_rewards": self.arm_avg_rewards.copy(),
            "arm_selection_rates": arm_selection_rates,
            "current_arm": self.current_arm
        }
    
    def get_best_arm(self) -> str:
        """
        Get arm with highest average reward.
        
        Returns:
            str: Best arm name
        """
        return max(self.arm_avg_rewards, key=self.arm_avg_rewards.get)
    
    def __repr__(self):
        return (
            f"UCB1Bandit("
            f"arms={self.arms}, "
            f"c={self.exploration_constant}, "
            f"warmup_blocks={self.warmup_blocks}, "
            f"min_pulls={self.min_pulls_per_arm}, "
            f"total_pulls={self.total_pulls})"
        )


if __name__ == "__main__":
    # Test the bandit
    print("\n=== UCB1 Bandit Test ===\n")
    
    bandit = UCB1Bandit(
        arms=["box", "point", "both"],
        exploration_constant=2.0,
        warmup_blocks=5,  # 5 blocks per arm during warmup
        min_pulls_per_arm=3,
        reward_clip_max=10.0,
        seed=42
    )
    
    # Simulate 50 pulls with different reward profiles
    # Assume: box → MSD=2.5, point → MSD=3.0, both → MSD=2.0 (both is best)
    arm_true_metrics = {"box": 2.5, "point": 3.0, "both": 2.0}
    
    print("Simulated arm metrics (lower MSD is better):")
    for arm, metric in arm_true_metrics.items():
        print(f"  {arm}: MSD = {metric:.2f} mm")
    print()
    
    for i in range(50):
        # Select arm
        arm = bandit.select_arm()
        
        # Simulate noisy metric observation
        noise = np.random.randn() * 0.2
        observed_metric = arm_true_metrics[arm] + noise
        
        # Clip metric
        observed_metric = min(observed_metric, 10.0)
        
        # Update reward (negative MSD)
        bandit.update_reward(arm, observed_metric)
        
        if (i + 1) % 10 == 0:
            stats = bandit.get_statistics()
            print(f"Pull {i+1}:")
            print(f"  Arm counts: {stats['arm_counts']}")
            print(f"  Avg rewards: {dict((k, f'{v:.3f}') for k, v in stats['arm_avg_rewards'].items())}")
            print(f"  Selection rates: {dict((k, f'{v:.3f}') for k, v in stats['arm_selection_rates'].items())}")
            print()
    
    best_arm = bandit.get_best_arm()
    print(f"Best arm: {best_arm} (highest avg reward)")
    print(f"Expected best: both (lowest MSD=2.0, highest reward=-2.0)")
    print(f"\nNote: Reward = -MSD, so 'both' should have reward ≈ -2.0")
