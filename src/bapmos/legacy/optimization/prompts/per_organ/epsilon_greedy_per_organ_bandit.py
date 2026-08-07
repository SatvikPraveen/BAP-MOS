"""
Per-Organ Epsilon-Greedy Multi-Armed Bandit for Adaptive Prompt Selection

Maintains separate epsilon-greedy bandits for each organ, allowing organ-specific
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

Algorithm: Epsilon-Greedy
    - With probability epsilon: explore (select random arm)
    - With probability 1-epsilon: exploit (select best arm)
    - Simple, interpretable, widely used baseline
    - Often performs well in practice despite theoretical limitations

Architecture:
    - One EpsilonGreedyBandit instance per organ
    - Independent learning: organ decisions don't interfere
    - Aggregated statistics for overall monitoring
    - Epsilon decay support for transitioning from exploration to exploitation

Reference: Sutton, R. S., & Barto, A. G. (2018). 
Reinforcement learning: An introduction (2nd ed.). MIT press.
"""

import numpy as np
from typing import List, Dict, Any


class EpsilonGreedyBandit:
    """
    Epsilon-Greedy bandit for single organ adaptive prompt selection.
    
    Arms:
        - "box": Box prompts only
        - "point": Point prompts only
        - "both": Box + Point prompts simultaneously
    
    Design:
        - Epsilon-greedy exploration/exploitation trade-off
        - Block-level decisions: arm held for full training block
        - Reward = -MSD (absolute quality, not improvement)
        - Optional epsilon decay for adaptive exploration
    
    Args:
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        epsilon (float): Exploration probability (default: 0.1)
        epsilon_decay (float): Decay factor per pull (default: 1.0 = no decay)
        epsilon_min (float): Minimum epsilon value (default: 0.01)
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        arms: List[str] = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        # Default arms
        if arms is None:
            arms = ["box", "point", "both"]
        
        assert len(arms) > 0, "Must have at least one arm"
        assert 0.0 <= epsilon <= 1.0, "epsilon must be in [0, 1]"
        assert 0.0 <= epsilon_min <= 1.0, "epsilon_min must be in [0, 1]"
        assert 0.0 < epsilon_decay <= 1.0, "epsilon_decay must be in (0, 1]"
        assert warmup_blocks >= 0, "warmup_blocks must be non-negative"
        assert min_pulls_per_arm >= 0, "min_pulls_per_arm must be non-negative"
        
        self.arms = arms
        self.n_arms = len(arms)
        self.epsilon_initial = epsilon
        self.epsilon = epsilon  # Current epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.warmup_blocks = warmup_blocks  # Blocks per arm during warmup
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # State tracking
        self.arm_counts = {arm: 0 for arm in arms}  # N_a(t)
        self.arm_rewards = {arm: [] for arm in arms}  # List of rewards per arm
        self.arm_avg_rewards = {arm: 0.0 for arm in arms}  # Average reward per arm
        
        self.total_pulls = 0
        self.current_arm = None
        
        # History
        self.selection_history = []  # (pull_idx, arm_name, was_exploration)
        self.reward_history = []  # (pull_idx, arm_name, reward)
        
        # RNG
        self.rng = np.random.default_rng(seed)
    
    def select_arm(self) -> str:
        """
        Select an arm using epsilon-greedy strategy.
        
        During warmup (first warmup_blocks * n_arms pulls):
            - Uniform round-robin exploration
        
        After warmup:
            - Enforce minimum pulls per arm
            - With probability epsilon: explore (random arm)
            - With probability 1-epsilon: exploit (best arm)
            - Decay epsilon after each selection
        
        Returns:
            str: Selected arm name
        """
        was_exploration = False
        
        # Warmup phase: uniform round-robin
        total_warmup_pulls = self.warmup_blocks * self.n_arms
        if self.total_pulls < total_warmup_pulls:
            arm_idx = self.total_pulls % self.n_arms
            selected_arm = self.arms[arm_idx]
            was_exploration = True
        else:
            # Post-warmup: enforce minimum pulls per arm
            arms_below_min = [arm for arm in self.arms if self.arm_counts[arm] < self.min_pulls_per_arm]
            
            if arms_below_min:
                # Force selection of under-explored arm
                selected_arm = arms_below_min[0]
                was_exploration = True
            else:
                # Epsilon-greedy selection
                if self.rng.random() < self.epsilon:
                    # Explore: select random arm
                    selected_arm = self.rng.choice(self.arms)
                    was_exploration = True
                else:
                    # Exploit: select best arm (highest average reward)
                    selected_arm = max(self.arm_avg_rewards, key=self.arm_avg_rewards.get)
                    was_exploration = False
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update state
        self.current_arm = selected_arm
        self.total_pulls += 1
        self.arm_counts[selected_arm] += 1
        
        # Record history
        self.selection_history.append((self.total_pulls, selected_arm, was_exploration))
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current bandit statistics.
        
        Returns:
            dict: {
                "total_pulls": int,
                "arm_counts": dict,
                "arm_avg_rewards": dict,
                "arm_selection_rates": dict,
                "current_arm": str,
                "current_epsilon": float,
                "exploration_rate": float
            }
        """
        if self.total_pulls == 0:
            selection_rates = {arm: 0.0 for arm in self.arms}
            exploration_rate = 0.0
        else:
            selection_rates = {
                arm: count / self.total_pulls
                for arm, count in self.arm_counts.items()
            }
            # Compute actual exploration rate from history
            n_explorations = sum(1 for _, _, was_exp in self.selection_history if was_exp)
            exploration_rate = n_explorations / self.total_pulls
        
        return {
            "total_pulls": self.total_pulls,
            "arm_counts": dict(self.arm_counts),
            "arm_avg_rewards": dict(self.arm_avg_rewards),
            "arm_selection_rates": selection_rates,
            "current_arm": self.current_arm,
            "current_epsilon": self.epsilon,
            "exploration_rate": exploration_rate
        }
    
    def get_best_arm(self) -> str:
        """
        Get the arm with highest average reward.
        
        Returns:
            str: Best arm name
        """
        if self.total_pulls == 0:
            return self.arms[0]  # Default
        
        return max(self.arm_avg_rewards, key=self.arm_avg_rewards.get)
    
    def __repr__(self):
        return (
            f"EpsilonGreedyBandit("
            f"arms={self.arms}, "
            f"epsilon={self.epsilon:.4f}, "
            f"total_pulls={self.total_pulls})"
        )


class EpsilonGreedyPerOrganBandit:
    """
    Per-organ epsilon-greedy bandit for adaptive prompt selection.
    
    Maintains a separate EpsilonGreedyBandit instance for each organ, allowing
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
        epsilon (float): Initial exploration probability (default: 0.1)
        epsilon_decay (float): Decay factor per pull (default: 1.0 = no decay)
        epsilon_min (float): Minimum epsilon value (default: 0.01)
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        organs: List[str],
        arms: List[str] = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # Create separate bandit for each organ
        self.bandits = {}
        for i, organ in enumerate(organs):
            # Use different seed for each organ to avoid synchronization
            organ_seed = seed + i
            self.bandits[organ] = EpsilonGreedyBandit(
                arms=arms,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
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
        Select an arm for the specified organ using its epsilon-greedy bandit.
        
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
        
        Returns:
            dict: {
                "total_pulls": int,  # Total pulls across all organs
                "per_organ": dict,   # Per-organ statistics
                "aggregated": dict   # Aggregated metrics (includes total_blocks for trainer compatibility)
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
        
        # Average epsilon across organs
        avg_epsilon = np.mean([self.bandits[organ].epsilon for organ in self.organs])
        
        # Aggregate exploration rate
        total_exploration_rate = 0.0
        for organ in self.organs:
            total_exploration_rate += per_organ[organ]["exploration_rate"]
        avg_exploration_rate = total_exploration_rate / self.n_organs if self.n_organs > 0 else 0.0
        
        return {
            "total_pulls": self.total_pulls,
            "per_organ": per_organ,
            "aggregated": {
                "arm_counts": dict(aggregated_arm_counts),
                "arm_avg_rewards": dict(aggregated_avg_rewards),
                "arm_selection_rates": dict(aggregated_selection_rates),
                "avg_epsilon": avg_epsilon,
                "avg_exploration_rate": avg_exploration_rate,
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
        avg_epsilon = np.mean([self.bandits[organ].epsilon for organ in self.organs])
        return (
            f"EpsilonGreedyPerOrganBandit("
            f"organs={len(self.organs)}, "
            f"arms={self.arms}, "
            f"avg_epsilon={avg_epsilon:.4f}, "
            f"warmup_blocks={self.warmup_blocks}, "
            f"total_pulls={self.total_pulls})"
        )


if __name__ == "__main__":
    # Test the per-organ epsilon-greedy bandit
    print("\n=== Epsilon-Greedy Per-Organ Bandit Test ===\n")
    
    # Multi-organ setup (Prostate radiotherapy dataset)
    organs = ["rectum", "bladder", "ptv1"]
    
    bandit = EpsilonGreedyPerOrganBandit(
        organs=organs,
        arms=["box", "point", "both"],
        epsilon=0.3,  # 30% exploration initially
        epsilon_decay=0.99,  # Decay epsilon by 1% per pull
        epsilon_min=0.05,  # Minimum 5% exploration
        warmup_blocks=3,  # 3 blocks per arm during warmup
        min_pulls_per_arm=2,
        reward_clip_max=10.0,
        seed=42
    )
    
    # Simulate organ-specific MSD profiles
    # PTV1 (large tumor target) → "both" works better for coverage
    # OARs (rectum, bladder) → may vary based on boundary characteristics
    organ_arm_metrics = {
        "rectum": {"box": 2.5, "point": 3.0, "both": 2.2},
        "bladder": {"box": 2.8, "point": 3.5, "both": 2.4},
        "ptv1": {"box": 3.2, "point": 2.8, "both": 1.9}
    }
    
    print("Dataset: Prostate Cancer Radiotherapy Planning")
    print("Simulated organ-specific metrics (lower MSD is better):")
    for organ in organs:
        organ_type = "OAR" if organ in ["rectum", "bladder"] else "PTV"
        print(f"  {organ} ({organ_type}):")
        for arm, metric in organ_arm_metrics[organ].items():
            print(f"    {arm}: MSD = {metric:.2f} mm")
    print()
    
    # Simulate 30 pulls per organ (90 total)
    np.random.seed(42)
    for pull in range(30):
        for organ in organs:
            # Select arm for this organ
            arm = bandit.select_arm(organ)
            
            # Simulate noisy metric observation
            true_metric = organ_arm_metrics[organ][arm]
            noise = np.random.randn() * 0.2
            observed_metric = max(0, true_metric + noise)
            
            # Update reward
            bandit.update_reward(organ, arm, observed_metric)
        
        # Print progress every 10 pulls
        if (pull + 1) % 10 == 0:
            print(f"\n=== After {pull + 1} pulls per organ ===")
            stats = bandit.get_all_statistics()
            
            print(f"Total pulls across all organs: {stats['total_pulls']}")
            print(f"Average epsilon: {stats['aggregated']['avg_epsilon']:.4f}")
            print(f"Average exploration rate: {stats['aggregated']['avg_exploration_rate']:.3f}")
            print(f"\nAggregated arm counts: {stats['aggregated']['arm_counts']}")
            print(f"Aggregated selection rates: {dict((k, f'{v:.3f}') for k, v in stats['aggregated']['arm_selection_rates'].items())}")
            
            print("\nPer-organ best arms:")
            best_arms = bandit.get_best_arms_per_organ()
            for organ in organs:
                organ_stats = stats['per_organ'][organ]
                print(f"  {organ}: {best_arms[organ]} "
                      f"(epsilon={organ_stats['current_epsilon']:.4f}, "
                      f"counts: {organ_stats['arm_counts']}, "
                      f"avg rewards: {dict((k, f'{v:.2f}') for k, v in organ_stats['arm_avg_rewards'].items())})")
    
    print("\n=== Final Results ===")
    print("\nExpected best arms per organ:")
    print("  rectum: both (MSD=2.2)")
    print("  bladder: both (MSD=2.4)")
    print("  ptv1: both (MSD=1.9)")
    
    print("\nLearned best arms per organ:")
    best_arms = bandit.get_best_arms_per_organ()
    for organ in organs:
        print(f"  {organ}: {best_arms[organ]}")
    
    print("\n✓ Per-organ epsilon-greedy bandits allow organ-specific strategy learning!")
    print("  Simple exploration/exploitation trade-off with epsilon parameter")
    print("  Epsilon decay enables transition from exploration to exploitation")
