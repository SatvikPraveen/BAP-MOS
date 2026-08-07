"""
UCB-Tuned Multi-Armed Bandit for Per-Organ Adaptive Prompt Selection

UCB-Tuned is a variance-aware variant of UCB1 that uses empirical variance
estimates to potentially achieve better performance with tighter confidence bounds.

Key Differences from UCB1:
- Uses variance estimates V(a) instead of fixed exploration constant
- Automatically adapts exploration based on reward variance
- Often performs better in practice than UCB1

Per-Organ Design:
- Each organ maintains an independent UCB-Tuned bandit
- Rewards are organ-specific MSDs mapped to [0, 1] for the standard 1/4 variance bound
  (reward = 1 - clipped_msd / reward_clip_max; lower MSD → higher reward)

Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
Finite-time analysis of the multi-armed bandit problem.
Machine Learning, 47(2-3), 235-256.
"""

import numpy as np
from typing import List, Dict, Any
import math


class UCBTunedBandit:
    """
    UCB-Tuned algorithm for single organ prompt selection.
    
    UCB-Tuned formula:
        UCB_tuned(a) = Q(a) + sqrt(ln(n) / N(a) * min(1/4, V(a)))
    
    where V(a) is the variance estimate:
        V(a) = (1/N(a)) * sum(r_i^2) - Q(a)^2 + sqrt(2*ln(n)/N(a))
    
    Args:
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        warmup_blocks (int): Number of training blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm even after warmup (default: 5)
        reward_clip_max (float): Maximum MSD (mm) for clipping before [0, 1] normalization
        seed (int): Random seed for reproducibility

    Rewards are stored in [0, 1] so min(1/4, V(a)) matches the UCB-Tuned analysis for
    bounded rewards. UCB1 per-organ may use unnormalized negative MSD instead.
    """
    
    def __init__(
        self,
        arms: List[str] = None,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        # Default arms
        if arms is None:
            arms = ["box", "point", "both"]
        
        assert len(arms) > 0, "Must have at least one arm"
        assert warmup_blocks >= 0, "warmup_blocks must be non-negative"
        assert min_pulls_per_arm >= 0, "min_pulls_per_arm must be non-negative"
        
        self.arms = arms
        self.n_arms = len(arms)
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # State tracking
        self.arm_counts = {arm: 0 for arm in arms}  # N_a(t)
        self.arm_rewards = {arm: [] for arm in arms}  # List of rewards per arm
        self.arm_avg_rewards = {arm: 0.0 for arm in arms}  # Q(a)
        self.arm_squared_rewards = {arm: [] for arm in arms}  # For variance calculation
        
        self.total_pulls = 0
        
        # History
        self.selection_history = []  # (pull_idx, arm_name)
        self.reward_history = []  # (pull_idx, arm_name, reward)
        
        # RNG for warmup
        self.rng = np.random.default_rng(seed)
    
    def select_arm(self) -> str:
        """
        Select an arm using UCB-Tuned algorithm.
        
        During warmup:
            - Round-robin exploration
        
        After warmup:
            - Enforce minimum pulls per arm
            - UCB-Tuned: arm* = argmax_a [ Q(a) + sqrt(ln(t)/N(a) * min(1/4, V(a))) ]
        
        Returns:
            str: Selected arm name
        """
        # Warmup phase: round-robin
        total_warmup_pulls = self.warmup_blocks * self.n_arms
        if self.total_pulls < total_warmup_pulls:
            arm_idx = self.total_pulls % self.n_arms
            selected_arm = self.arms[arm_idx]
        else:
            # Post-warmup: enforce minimum pulls
            arms_below_min = [arm for arm in self.arms if self.arm_counts[arm] < self.min_pulls_per_arm]
            
            if arms_below_min:
                selected_arm = arms_below_min[0]
            else:
                # UCB-Tuned selection
                ucb_values = {}
                
                for arm in self.arms:
                    Q_a = self.arm_avg_rewards[arm]
                    N_a = self.arm_counts[arm]
                    
                    if N_a == 0:
                        ucb_values[arm] = float('inf')
                    else:
                        # Calculate variance estimate V(a)
                        V_a = self._calculate_variance(arm)
                        
                        # UCB-Tuned formula
                        # sqrt(ln(t)/N(a) * min(1/4, V(a)))
                        exploration_term = math.sqrt(
                            (math.log(self.total_pulls) / N_a) * min(0.25, V_a)
                        )
                        
                        ucb_values[arm] = Q_a + exploration_term
                
                # Select arm with maximum UCB value (random tie-breaking)
                max_value = max(ucb_values.values())
                best_arms = [arm for arm, value in ucb_values.items() if value == max_value]
                
                if len(best_arms) > 1:
                    # Random tie-breaking using seeded RNG
                    selected_arm = self.rng.choice(best_arms)
                else:
                    selected_arm = best_arms[0]
        
        # Update state (UCB1-style lifecycle)
        self.total_pulls += 1
        self.arm_counts[selected_arm] += 1

        # Record selection
        self.selection_history.append((self.total_pulls, selected_arm))
        
        return selected_arm
    
    def _calculate_variance(self, arm: str) -> float:
        """
        Calculate variance estimate V(a) for UCB-Tuned.
        
        V(a) = (1/N(a)) * sum(r_i^2) - Q(a)^2 + sqrt(2*ln(t)/N(a))
        
        Args:
            arm: Arm name
        
        Returns:
            Variance estimate (clamped to be non-negative)
        """
        N_a = self.arm_counts[arm]
        
        if N_a == 0:
            return 0.25  # Default to max variance
        
        Q_a = self.arm_avg_rewards[arm]
        
        # Mean of squared rewards
        if len(self.arm_squared_rewards[arm]) > 0:
            mean_squared = np.mean(self.arm_squared_rewards[arm])
        else:
            mean_squared = 0.0
        
        # Empirical variance: E[R^2] - E[R]^2
        empirical_variance = mean_squared - (Q_a ** 2)
        
        # Variance bound term
        if self.total_pulls > 1:
            variance_bound = math.sqrt(2 * math.log(self.total_pulls) / N_a)
        else:
            variance_bound = 0.0
        
        # V(a) = empirical_variance + variance_bound
        V_a = empirical_variance + variance_bound
        
        # Clamp to [0, 0.25] (max variance for bounded rewards)
        V_a = max(0.0, min(0.25, V_a))
        
        return V_a
    
    def update_reward(self, arm: str, val_metric: float):
        """
        Update bandit state with reward for selected arm.

        MSD (mm) is clipped to [0, reward_clip_max] then mapped to [0, 1]:

            reward = 1 - clipped_msd / reward_clip_max

        so the UCB-Tuned min(1/4, V(a)) term matches bounded-reward theory.
        Lower MSD → higher reward.
        """
        assert arm in self.arms, f"Unknown arm: {arm}"

        clipped_metric = min(max(float(val_metric), 0.0), self.reward_clip_max)
        denom = self.reward_clip_max if self.reward_clip_max > 0 else 1.0
        reward = 1.0 - (clipped_metric / denom)
        
        # Update reward tracking
        self.arm_rewards[arm].append(reward)
        self.arm_squared_rewards[arm].append(reward ** 2)
        
        # Update average reward
        self.arm_avg_rewards[arm] = np.mean(self.arm_rewards[arm])
        
        # Record history
        self.reward_history.append((self.total_pulls, arm, reward))
    
    def get_best_arm(self) -> str:
        """
        Get the arm with highest average reward.
        
        Returns:
            str: Best arm name
        """
        return max(self.arm_avg_rewards, key=self.arm_avg_rewards.get)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current bandit statistics.
        
        Returns:
            dict: Statistics including counts, rewards, selection rates, variance estimates
        """
        total = max(self.total_pulls, 1)
        
        # Calculate variance estimates for all arms
        variance_estimates = {}
        for arm in self.arms:
            variance_estimates[arm] = self._calculate_variance(arm)
        
        return {
            "total_pulls": self.total_pulls,
            "arm_counts": self.arm_counts.copy(),
            "arm_avg_rewards": self.arm_avg_rewards.copy(),
            "arm_selection_rates": {arm: self.arm_counts[arm] / total for arm in self.arms},
            "arm_variance_estimates": variance_estimates.copy(),
            "best_arm": self.get_best_arm(),
        }


class UCBTunedPerOrganBandit:
    """
    Per-organ UCB-Tuned bandit for multi-organ segmentation.
    
    Each organ (rectum, bladder, ptv1) maintains an independent UCB-Tuned bandit
    instance, allowing organ-specific learning of optimal prompt strategies.
    
    Design:
        - Wraps multiple UCBTunedBandit instances (one per organ)
        - Each organ learns independently
        - Shared hyperparameters for fair comparison
        - Per-organ reward updates (organ-specific MSDs)
    
    Args:
        organs (List[str]): List of organ names (e.g., ["rectum", "bladder", "ptv1"])
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum guaranteed pulls per arm (default: 5)
        reward_clip_max (float): Maximum MSD for clipping (default: 20.0 mm)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        organs: List[str],
        arms: List[str] = None,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        if arms is None:
            arms = ["box", "point", "both"]
        
        assert len(organs) > 0, "Must have at least one organ"
        assert len(arms) > 0, "Must have at least one arm"
        
        self.organs = organs
        self.arms = arms
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # Create independent bandit for each organ
        self.bandits = {}
        for i, organ in enumerate(organs):
            self.bandits[organ] = UCBTunedBandit(
                arms=arms,
                warmup_blocks=warmup_blocks,
                min_pulls_per_arm=min_pulls_per_arm,
                reward_clip_max=reward_clip_max,
                seed=seed + i  # Different seed per organ for diversity
            )
    
    def select_arm(self, organ: str) -> str:
        """
        Select arm for specified organ using its UCB-Tuned bandit.
        
        Args:
            organ: Organ name (must be in self.organs)
        
        Returns:
            str: Selected arm name
        """
        assert organ in self.bandits, f"Unknown organ: {organ}"
        return self.bandits[organ].select_arm()
    
    def update_reward(self, organ: str, arm: str, val_metric: float):
        """
        Update reward for specified organ's bandit.
        
        Args:
            organ: Organ name
            arm: Arm that was pulled
            val_metric: Validation metric for this organ (MSD in mm)
        """
        assert organ in self.bandits, f"Unknown organ: {organ}"
        self.bandits[organ].update_reward(arm, val_metric)
    
    def get_best_arms_per_organ(self) -> Dict[str, str]:
        """
        Get best arm for each organ.
        
        Returns:
            dict: {organ_name: best_arm_name}
        """
        return {organ: bandit.get_best_arm() for organ, bandit in self.bandits.items()}
    
    def get_statistics_for_organ(self, organ: str) -> Dict[str, Any]:
        """
        Get statistics for specific organ.
        
        Args:
            organ: Organ name
        
        Returns:
            dict: Bandit statistics for this organ
        """
        assert organ in self.bandits, f"Unknown organ: {organ}"
        return self.bandits[organ].get_statistics()
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all organs.
        
        Returns:
            dict: {
                'per_organ': {organ: stats},
                'aggregated': {overall metrics}
            }
        """
        per_organ_stats = {}
        for organ in self.organs:
            per_organ_stats[organ] = self.get_statistics_for_organ(organ)
        
        # Aggregated statistics
        total_pulls = sum(bandit.total_pulls for bandit in self.bandits.values())
        avg_variance = np.mean([
            stats['arm_variance_estimates'][arm]
            for stats in per_organ_stats.values()
            for arm in self.arms
        ]) if total_pulls > 0 else 0.0
        # Defensive copy and add symmetry keys for runtime safety
        aggregated = {
            "total_organs": len(self.organs),
            "total_blocks": total_pulls // len(self.organs) if len(self.organs) > 0 else 0,
            "avg_pulls_per_organ": total_pulls / len(self.organs) if len(self.organs) > 0 else 0,
            "avg_variance_estimate": avg_variance,
            "best_arms_per_organ": self.get_best_arms_per_organ().copy(),
            # Optional: add symmetry keys for trainer compatibility
            "arm_counts": {organ: per_organ_stats[organ]["arm_counts"].copy() for organ in self.organs},
            "arm_selection_rates": {organ: per_organ_stats[organ]["arm_selection_rates"].copy() for organ in self.organs},
            "arm_avg_rewards": {organ: per_organ_stats[organ]["arm_avg_rewards"].copy() for organ in self.organs},
            "arm_variance_estimates": {organ: per_organ_stats[organ]["arm_variance_estimates"].copy() for organ in self.organs},
        }
        return {
            "per_organ": {k: v.copy() if isinstance(v, dict) else v for k, v in per_organ_stats.items()},
            "aggregated": aggregated
        }


# ============================================================================
# Built-in Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UCB-TUNED PER-ORGAN BANDIT DEMO")
    print("=" * 70)
    
    # Initialize per-organ bandit
    organs = ["rectum", "bladder", "ptv1"]
    arms = ["box", "point", "both"]
    
    bandit = UCBTunedPerOrganBandit(
        organs=organs,
        arms=arms,
        warmup_blocks=3,  # 3 blocks per arm = 9 total warmup pulls per organ
        min_pulls_per_arm=5,
        reward_clip_max=20.0,
        seed=42
    )
    
    print(f"\nInitialized UCB-Tuned per-organ bandit:")
    print(f"  Organs: {organs}")
    print(f"  Arms: {arms}")
    print(f"  Warmup: 3 blocks/arm × 3 arms = 9 pulls per organ")
    print(f"  Min pulls/arm: 5")
    
    # Simulate training with synthetic MSDs
    # Ground truth: rectum prefers "box", bladder prefers "point", ptv1 prefers "both"
    true_msds = {
        "rectum": {"box": 2.0, "point": 4.0, "both": 3.0},
        "bladder": {"box": 4.0, "point": 2.0, "both": 3.0},
        "ptv1": {"box": 3.5, "point": 3.5, "both": 1.5},
    }
    
    print("\n" + "-" * 70)
    print("SIMULATION: 30 pulls per organ")
    print("-" * 70)
    
    np.random.seed(42)
    
    for pull in range(30):
        # Select arms for all organs
        selected_arms = {}
        for organ in organs:
            selected_arms[organ] = bandit.select_arm(organ)
        
        # Simulate validation (noisy MSDs based on true values)
        for organ in organs:
            arm = selected_arms[organ]
            true_msd = true_msds[organ][arm]
            noisy_msd = true_msd + np.random.normal(0, 0.3)  # Add noise
            noisy_msd = max(0.1, noisy_msd)  # Keep positive
            
            bandit.update_reward(organ, arm, noisy_msd)
        
        # Log every 10 pulls
        if (pull + 1) % 10 == 0:
            print(f"\nPull {pull + 1}:")
            stats = bandit.get_all_statistics()
            for organ in organs:
                organ_stats = stats['per_organ'][organ]
                best_arm = organ_stats['best_arm']
                var_estimates = organ_stats['arm_variance_estimates']
                var_str = ', '.join([f'{arm}={var_estimates[arm]:.4f}' for arm in arms])
                print(f"  {organ}: best={best_arm}, variances={var_str}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    stats = bandit.get_all_statistics()
    best_arms = stats['aggregated']['best_arms_per_organ']
    
    print("\nBest arms learned per organ:")
    for organ in organs:
        learned = best_arms[organ]
        expected_arms = {"rectum": "box", "bladder": "point", "ptv1": "both"}
        expected = expected_arms.get(organ, "?")
        match = "✓" if learned == expected else "✗"
        print(f"  {organ}: {learned} (expected: {expected}) {match}")
    
    print("\nPer-organ statistics:")
    for organ in organs:
        organ_stats = stats['per_organ'][organ]
        print(f"\n  {organ}:")
        print(f"    Total pulls: {organ_stats['total_pulls']}")
        print(f"    Arm counts: {organ_stats['arm_counts']}")
        avg_rewards_str = ', '.join([f'{arm}={organ_stats["arm_avg_rewards"][arm]:.3f}' for arm in arms])
        print(f"    Avg rewards: {avg_rewards_str}")
        var_estimates_str = ', '.join([f'{arm}={organ_stats["arm_variance_estimates"][arm]:.4f}' for arm in arms])
        print(f"    Variance estimates: {var_estimates_str}")
    
    print("\n" + "=" * 70)
    print("UCB-Tuned uses variance estimates to adapt exploration automatically!")
    print("Lower variance → tighter confidence bounds → less exploration needed")
    print("=" * 70)
