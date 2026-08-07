"""
Per-Organ Epsilon-Decay Multi-Armed Bandit for Adaptive Prompt Selection

Maintains separate epsilon-decay bandits for each organ with multiple decay schedules.
Unlike epsilon-greedy with optional decay, this algorithm is specifically designed
around decaying exploration strategies.

Scientific Rationale:
    Different organs have different characteristics:
    - Size: PTV1 (large tumor target) vs Bladder (variable size)
    - Shape: Rectum (tubular) vs Bladder (spherical) vs PTV1 (irregular)
    - Boundary clarity: OARs may have different edge characteristics than tumor
    - Clinical importance: Different precision requirements for PTV vs OARs
    
    Epsilon decay enables:
    - High initial exploration to discover good strategies
    - Gradual transition to exploitation as learning progresses
    - Multiple decay schedules for different learning dynamics

Decay Schedules:
    - exponential: ε(t) = max(ε_min, ε_0 * decay^t)
    - linear: ε(t) = max(ε_min, ε_0 - decay_rate * t)
    - inverse: ε(t) = max(ε_min, ε_0 / (1 + decay_rate * t))
    - step: ε(t) decreases by factor every N steps
    - cosine: ε(t) follows cosine annealing schedule

Reference: 
- Sutton & Barto (2018). Reinforcement Learning: An Introduction.
- Tokic (2010). Adaptive ε-greedy exploration in reinforcement learning.
"""

import numpy as np
from typing import List, Dict, Any, Literal
import math


class EpsilonDecayBandit:
    """
    Epsilon-decay bandit for single organ with multiple decay schedules.
    
    Arms:
        - "box": Box prompts only
        - "point": Point prompts only
        - "both": Box + Point prompts simultaneously
    
    Decay Schedules:
        - exponential: Fastest initial decay, common in deep learning
        - linear: Steady linear reduction
        - inverse: Slow decay, good for cautious exploration
        - step: Sudden drops at intervals
        - cosine: Smooth cosine annealing
    
    Args:
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        epsilon_start (float): Initial exploration probability (default: 0.3)
        epsilon_end (float): Final/minimum epsilon value (default: 0.01)
        decay_schedule (str): Decay schedule type (default: "exponential")
        decay_steps (int): Total steps for decay (default: 1000)
        decay_rate (float): Decay rate parameter (schedule-specific)
        step_size (int): Steps between drops for 'step' schedule (default: 100)
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        arms: List[str] = None,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        decay_schedule: Literal["exponential", "linear", "inverse", "step", "cosine"] = "exponential",
        decay_steps: int = 1000,
        decay_rate: float = None,
        step_size: int = 100,
        warmup_blocks: int = 10,
        min_pulls_per_arm: int = 5,
        reward_clip_max: float = 20.0,
        seed: int = 0
    ):
        # Default arms
        if arms is None:
            arms = ["box", "point", "both"]
        
        assert len(arms) > 0, "Must have at least one arm"
        assert 0.0 <= epsilon_start <= 1.0, "epsilon_start must be in [0, 1]"
        assert 0.0 <= epsilon_end <= 1.0, "epsilon_end must be in [0, 1]"
        assert epsilon_end <= epsilon_start, "epsilon_end must be <= epsilon_start"
        assert decay_schedule in ["exponential", "linear", "inverse", "step", "cosine"], \
            f"Unknown decay schedule: {decay_schedule}"
        
        self.arms = arms
        self.n_arms = len(arms)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start  # Current epsilon
        self.decay_schedule = decay_schedule
        self.decay_steps = decay_steps
        self.step_size = step_size
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # Compute decay rate if not provided
        if decay_rate is None:
            if decay_schedule == "exponential":
                # decay_rate such that epsilon_start * rate^decay_steps = epsilon_end
                if epsilon_end > 0:
                    self.decay_rate = (epsilon_end / epsilon_start) ** (1 / decay_steps)
                else:
                    self.decay_rate = 0.99  # Default fallback
            elif decay_schedule == "linear":
                # Linear decrease over decay_steps
                self.decay_rate = (epsilon_start - epsilon_end) / decay_steps
            elif decay_schedule == "inverse":
                # Inverse schedule rate
                self.decay_rate = 0.001  # Default
            elif decay_schedule == "step":
                # Step decay factor
                self.decay_rate = 0.5  # Halve epsilon at each step
            else:  # cosine
                self.decay_rate = None  # Not used for cosine
        else:
            self.decay_rate = decay_rate
        
        # State tracking
        self.arm_counts = {arm: 0 for arm in arms}
        self.arm_rewards = {arm: [] for arm in arms}
        self.arm_avg_rewards = {arm: 0.0 for arm in arms}
        
        self.total_pulls = 0
        self.decay_step = 0  # Step counter for decay (post-warmup)
        self.current_arm = None
        
        # History
        self.selection_history = []  # (pull_idx, arm_name, was_exploration, epsilon)
        self.reward_history = []  # (pull_idx, arm_name, reward)
        self.epsilon_history = [epsilon_start]  # Track epsilon over time
        
        # RNG
        self.rng = np.random.default_rng(seed)
    
    def _compute_epsilon(self, step: int) -> float:
        """
        Compute epsilon value for given decay step.
        
        Args:
            step (int): Current decay step (0-indexed, post-warmup)
        
        Returns:
            float: Epsilon value
        """
        if self.decay_schedule == "exponential":
            # ε(t) = max(ε_end, ε_start * decay_rate^t)
            epsilon = self.epsilon_start * (self.decay_rate ** step)
            
        elif self.decay_schedule == "linear":
            # ε(t) = max(ε_end, ε_start - decay_rate * t)
            epsilon = self.epsilon_start - self.decay_rate * step
            
        elif self.decay_schedule == "inverse":
            # ε(t) = max(ε_end, ε_start / (1 + decay_rate * t))
            epsilon = self.epsilon_start / (1.0 + self.decay_rate * step)
            
        elif self.decay_schedule == "step":
            # ε(t) = ε_start * decay_rate^floor(t / step_size)
            num_drops = step // self.step_size
            epsilon = self.epsilon_start * (self.decay_rate ** num_drops)
            
        elif self.decay_schedule == "cosine":
            # Cosine annealing: ε(t) = ε_end + 0.5 * (ε_start - ε_end) * (1 + cos(π * t / T))
            if step >= self.decay_steps:
                epsilon = self.epsilon_end
            else:
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * step / self.decay_steps))
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * cosine_factor
        else:
            epsilon = self.epsilon_start  # Fallback
        
        # Clamp to [epsilon_end, epsilon_start]
        return max(self.epsilon_end, min(self.epsilon_start, epsilon))
    
    def select_arm(self) -> str:
        """
        Select an arm using epsilon-decay strategy.
        
        During warmup (first warmup_blocks * n_arms pulls):
            - Uniform round-robin exploration
        
        After warmup:
            - Enforce minimum pulls per arm
            - With probability epsilon: explore (random arm)
            - With probability 1-epsilon: exploit (best arm)
            - Decay epsilon according to schedule
        
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
                # Epsilon-greedy selection with decaying epsilon
                if self.rng.random() < self.epsilon:
                    # Explore: select random arm
                    selected_arm = self.rng.choice(self.arms)
                    was_exploration = True
                else:
                    # Exploit: select best arm (highest average reward)
                    selected_arm = max(self.arm_avg_rewards, key=self.arm_avg_rewards.get)
                    was_exploration = False
                
                # Update epsilon according to decay schedule
                self.decay_step += 1
                self.epsilon = self._compute_epsilon(self.decay_step)
        
        # Update state
        self.current_arm = selected_arm
        self.total_pulls += 1
        self.arm_counts[selected_arm] += 1
        
        # Record history
        self.selection_history.append((self.total_pulls, selected_arm, was_exploration, self.epsilon))
        self.epsilon_history.append(self.epsilon)
        
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
            dict: Comprehensive statistics including epsilon decay info
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
            n_explorations = sum(1 for _, _, was_exp, _ in self.selection_history if was_exp)
            exploration_rate = n_explorations / self.total_pulls
        
        return {
            "total_pulls": self.total_pulls,
            "decay_step": self.decay_step,
            "arm_counts": dict(self.arm_counts),
            "arm_avg_rewards": dict(self.arm_avg_rewards),
            "arm_selection_rates": dict(selection_rates),
            "current_arm": self.current_arm,
            "current_epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "decay_schedule": self.decay_schedule,
            "exploration_rate": exploration_rate,
            "epsilon_progress": self.decay_step / self.decay_steps if self.decay_steps > 0 else 0.0
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
            f"EpsilonDecayBandit("
            f"arms={self.arms}, "
            f"schedule={self.decay_schedule}, "
            f"ε={self.epsilon:.4f}, "
            f"pulls={self.total_pulls})"
        )


class EpsilonDecayPerOrganBandit:
    """
    Per-organ epsilon-decay bandit for adaptive prompt selection.
    
    Maintains a separate EpsilonDecayBandit instance for each organ, allowing
    organ-specific learning with decaying exploration. Supports multiple
    decay schedules for different exploration dynamics.
    
    Arms:
        - "box": Box prompts only
        - "point": Point prompts only
        - "both": Box + Point prompts simultaneously
    
    Design:
        - Independent bandits: organ-specific learning
        - Shared hyperparameters: fair comparison across organs
        - Multiple decay schedules: exponential, linear, inverse, step, cosine
        - Block-level decisions: arm held for full training block
        - Reward = -MSD (absolute quality, not improvement)
    
    Args:
        organs (List[str]): List of organ names (e.g., ["rectum", "bladder", "ptv1"])
        arms (List[str]): List of arm names (default: ["box", "point", "both"])
        epsilon_start (float): Initial exploration probability (default: 0.3)
        epsilon_end (float): Final/minimum epsilon value (default: 0.01)
        decay_schedule (str): Decay schedule type (default: "exponential")
        decay_steps (int): Total steps for decay (default: 1000)
        decay_rate (float): Decay rate parameter (schedule-specific, auto-computed if None)
        step_size (int): Steps between drops for 'step' schedule (default: 100)
        warmup_blocks (int): Number of blocks per arm during warmup (default: 10)
        min_pulls_per_arm (int): Minimum pulls per arm after warmup (default: 5)
        reward_clip_max (float): Maximum MSD value for clipping (default: 20.0)
        seed (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        organs: List[str],
        arms: List[str] = None,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        decay_schedule: Literal["exponential", "linear", "inverse", "step", "cosine"] = "exponential",
        decay_steps: int = 1000,
        decay_rate: float = None,
        step_size: int = 100,
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
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_schedule = decay_schedule
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_size = step_size
        self.warmup_blocks = warmup_blocks
        self.min_pulls_per_arm = min_pulls_per_arm
        self.reward_clip_max = reward_clip_max
        self.seed = seed
        
        # Create separate bandit for each organ
        self.bandits = {}
        for i, organ in enumerate(organs):
            # Use different seed for each organ to avoid synchronization
            organ_seed = seed + i
            self.bandits[organ] = EpsilonDecayBandit(
                arms=arms,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                decay_schedule=decay_schedule,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                step_size=step_size,
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
        Select an arm for the specified organ using its epsilon-decay bandit.
        
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
            dict: Comprehensive statistics including per-organ and aggregated metrics
                  (includes total_blocks for trainer compatibility)
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
        
        # Average decay progress
        avg_decay_progress = np.mean([
            per_organ[organ]["epsilon_progress"] for organ in self.organs
        ])
        
        return {
            "total_pulls": self.total_pulls,
            "per_organ": per_organ,
            "aggregated": {
                "arm_counts": dict(aggregated_arm_counts),
                "arm_avg_rewards": dict(aggregated_avg_rewards),
                "arm_selection_rates": dict(aggregated_selection_rates),
                "avg_epsilon": avg_epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "decay_schedule": self.decay_schedule,
                "avg_exploration_rate": avg_exploration_rate,
                "avg_decay_progress": avg_decay_progress,
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
            f"EpsilonDecayPerOrganBandit("
            f"organs={len(self.organs)}, "
            f"arms={self.arms}, "
            f"schedule={self.decay_schedule}, "
            f"avg_ε={avg_epsilon:.4f}, "
            f"pulls={self.total_pulls})"
        )


if __name__ == "__main__":
    # Test the per-organ epsilon-decay bandit
    print("\n=== Epsilon-Decay Per-Organ Bandit Test ===\n")
    
    # Multi-organ setup (Prostate radiotherapy dataset)
    organs = ["rectum", "bladder", "ptv1"]
    
    # Test with exponential decay
    bandit = EpsilonDecayPerOrganBandit(
        organs=organs,
        arms=["box", "point", "both"],
        epsilon_start=0.5,  # Start with 50% exploration
        epsilon_end=0.05,   # End with 5% exploration
        decay_schedule="exponential",
        decay_steps=50,  # Decay over 50 steps (post-warmup)
        warmup_blocks=3,
        min_pulls_per_arm=2,
        reward_clip_max=10.0,
        seed=42
    )
    
    # Simulate organ-specific MSD profiles
    organ_arm_metrics = {
        "rectum": {"box": 2.5, "point": 3.0, "both": 2.2},
        "bladder": {"box": 2.8, "point": 3.5, "both": 2.4},
        "ptv1": {"box": 3.2, "point": 2.8, "both": 1.9}
    }
    
    print(f"Decay Schedule: {bandit.decay_schedule}")
    print(f"Epsilon: {bandit.epsilon_start:.2f} → {bandit.epsilon_end:.2f}")
    print(f"Decay Steps: {bandit.decay_steps}")
    print("\nDataset: Prostate Cancer Radiotherapy Planning")
    print("Simulated organ-specific metrics (lower MSD is better):")
    for organ in organs:
        organ_type = "OAR" if organ in ["rectum", "bladder"] else "PTV"
        print(f"  {organ} ({organ_type}):")
        for arm, metric in organ_arm_metrics[organ].items():
            print(f"    {arm}: MSD = {metric:.2f} mm")
    print()
    
    # Simulate 30 pulls per organ
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
        
        # Print progress
        if (pull + 1) in [10, 20, 30]:
            print(f"\n=== After {pull + 1} pulls per organ ===")
            stats = bandit.get_all_statistics()
            
            print(f"Total pulls: {stats['total_pulls']}")
            print(f"Average epsilon: {stats['aggregated']['avg_epsilon']:.4f} "
                  f"(progress: {stats['aggregated']['avg_decay_progress']:.1%})")
            print(f"Exploration rate: {stats['aggregated']['avg_exploration_rate']:.3f}")
            print(f"\nAggregated arm selection: {stats['aggregated']['arm_selection_rates']}")
            
            print("\nPer-organ best arms:")
            best_arms = bandit.get_best_arms_per_organ()
            for organ in organs:
                organ_stats = stats['per_organ'][organ]
                print(f"  {organ}: {best_arms[organ]} "
                      f"(ε={organ_stats['current_epsilon']:.4f}, "
                      f"counts={organ_stats['arm_counts']})")
    
    print("\n=== Final Results ===")
    print("\nExpected best arms:")
    for organ, metrics in organ_arm_metrics.items():
        best_arm = min(metrics, key=metrics.get)
        print(f"  {organ}: {best_arm} (MSD={metrics[best_arm]:.2f})")
    
    print("\nLearned best arms:")
    best_arms = bandit.get_best_arms_per_organ()
    for organ in organs:
        print(f"  {organ}: {best_arms[organ]}")
    
    print("\n✓ Epsilon-decay enables smooth transition from exploration to exploitation!")
    print(f"  Started at {bandit.epsilon_start:.0%} exploration, ended at ~{stats['aggregated']['avg_epsilon']:.0%}")
