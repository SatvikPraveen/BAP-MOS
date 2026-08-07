"""
Variance-bounded UCB-Tuned bandit for per-organ prompt selection.

Action space A = {'box', 'point', 'both'}. Each organ maintains an independent
bandit with a FIFO sliding window of curriculum rewards (default W=50).

For arm a after t block updates (t incremented only in ``commit_block_reward``)::

    σ²_a = Var(rewards in deque)
    V_a  = σ²_a + sqrt(2 ln(t) / N_a)
    Ṽ_a  = min(0.25, V_a)
    Bonus_a = sqrt(ln(t) / N_a · Ṽ_a)
    UCB_a = Q_a + Bonus_a

**Critical:** ``select_arm()`` does NOT increment t or N_a. Only ``commit_block_reward``
increments them when validation rewards are committed at block end.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Mapping, Optional

RewardMemory = Literal["sliding", "cumulative"]

import numpy as np

from bapmos.method.types import Action, BlockState, DEFAULT_ARMS


class VarianceBoundedUCBTuned:
    """
    Single-organ variance-bounded UCB-Tuned bandit.

    ``memory='sliding'`` (default): Q and σ² over the last ``window_size`` rewards.
    ``memory='cumulative'``: Q and σ² over all N committed rewards (legacy core/).
    """

    def __init__(
        self,
        arms: Optional[List[Action]] = None,
        window_size: int = 50,
        warmup_blocks_per_arm: int = 10,
        min_pulls_per_arm: int = 5,
        seed: int = 0,
        *,
        memory: RewardMemory = "sliding",
    ) -> None:
        if arms is None:
            arms = list(DEFAULT_ARMS)
        if len(arms) == 0:
            raise ValueError("arms must be non-empty")
        if memory not in ("sliding", "cumulative"):
            raise ValueError(f"memory must be 'sliding' or 'cumulative', got {memory!r}")
        if memory == "sliding" and window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.arms: List[Action] = list(arms)
        self.memory: RewardMemory = memory
        self.window_size = int(window_size)
        self.warmup_blocks_per_arm = int(warmup_blocks_per_arm)
        self.min_pulls_per_arm = int(min_pulls_per_arm)
        self.seed = int(seed)

        self.t: int = 0  # block updates committed; NOT incremented in select_arm
        self.N_a: Dict[Action, int] = {a: 0 for a in self.arms}
        self.Q_a: Dict[Action, float] = {a: 0.0 for a in self.arms}
        self._rewards: Dict[Action, List[float]] = {a: [] for a in self.arms}
        self._rng = np.random.default_rng(seed)
        self._last_selected: Optional[Action] = None

    def _reward_history(self, arm: Action) -> List[float]:
        if self.memory == "sliding":
            return list(self._rewards[arm])[-self.window_size :]
        return list(self._rewards[arm])

    def _t_eff(self) -> int:
        """Use max(1, t) inside log terms before the first commit."""
        return max(1, self.t)

    def _empirical_variance(self, arm: Action) -> float:
        data = self._reward_history(arm)
        if len(data) < 2:
            return 0.0
        return float(np.var(data, ddof=1))

    def _variance_bound(self, arm: Action) -> float:
        """
        V_a = σ²_a + sqrt(2 ln(t) / N_a),  Ṽ_a = min(0.25, V_a).
        """
        n_a = self.N_a[arm]
        if n_a == 0:
            return 0.25
        t_eff = self._t_eff()
        sigma2 = self._empirical_variance(arm)
        bound_term = math.sqrt(2.0 * math.log(t_eff) / n_a)
        v_a = sigma2 + bound_term
        return min(0.25, max(0.0, v_a))

    def _ucb_score(self, arm: Action) -> float:
        n_a = self.N_a[arm]
        if n_a == 0:
            return float("inf")
        t_eff = self._t_eff()
        v_tilde = self._variance_bound(arm)
        bonus = math.sqrt((math.log(t_eff) / n_a) * v_tilde)
        return self.Q_a[arm] + bonus

    def select_arm(self) -> Action:
        """
        Select arm for the next block. Does NOT increment t or N_a.
        """
        total_warmup = self.warmup_blocks_per_arm * len(self.arms)
        if self.t < total_warmup:
            idx = self.t % len(self.arms)
            selected = self.arms[idx]
        else:
            below_min = [a for a in self.arms if self.N_a[a] < self.min_pulls_per_arm]
            if below_min:
                # Random among under-pulled arms to avoid arm-list order bias.
                selected = self._rng.choice(below_min)
            else:
                scores = {a: self._ucb_score(a) for a in self.arms}
                max_score = max(scores.values())
                best = [a for a, s in scores.items() if s == max_score]
                selected = self._rng.choice(best) if len(best) > 1 else best[0]

        self._last_selected = selected
        return selected

    def commit_block_reward(self, arm: Action, reward: float) -> None:
        """
        Commit reward at block end: update deque, Q_a, N_a, and t.
        """
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")
        r = float(np.clip(reward, 0.0, 1.0))
        self._rewards[arm].append(r)
        if self.memory == "sliding" and len(self._rewards[arm]) > self.window_size:
            del self._rewards[arm][: len(self._rewards[arm]) - self.window_size]
        hist = self._reward_history(arm)
        self.Q_a[arm] = float(np.mean(hist)) if hist else 0.0
        self.N_a[arm] += 1
        self.t += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Statistics aligned with legacy ``bapmos.legacy.optimization`` bandit W&B keys."""
        total = max(self.t, 1)
        return {
            "t": self.t,
            "total_pulls": self.t,
            "N_a": dict(self.N_a),
            "Q_a": dict(self.Q_a),
            "arm_counts": dict(self.N_a),
            "arm_avg_rewards": dict(self.Q_a),
            "arm_selection_rates": {a: self.N_a[a] / total for a in self.arms},
            "arm_variance_estimates": {a: self._variance_bound(a) for a in self.arms},
            "variance_bounds": {a: self._variance_bound(a) for a in self.arms},
            "best_arm": max(self.arms, key=lambda a: self.Q_a.get(a, 0.0)),
            "last_selected": self._last_selected,
        }


class PerOrganBanditPolicy:
    """Independent VarianceBoundedUCBTuned per organ."""

    def __init__(
        self,
        organs: List[str],
        arms: Optional[List[Action]] = None,
        window_size: int = 50,
        warmup_blocks_per_arm: int = 10,
        min_pulls_per_arm: int = 5,
        seed: int = 0,
        *,
        memory: RewardMemory = "sliding",
    ) -> None:
        if len(organs) == 0:
            raise ValueError("organs must be non-empty")
        self.organs = list(organs)
        self.arms = list(arms or DEFAULT_ARMS)
        self.bandits: Dict[str, VarianceBoundedUCBTuned] = {
            organ: VarianceBoundedUCBTuned(
                arms=self.arms,
                window_size=window_size,
                warmup_blocks_per_arm=warmup_blocks_per_arm,
                min_pulls_per_arm=min_pulls_per_arm,
                seed=seed + i,
                memory=memory,
            )
            for i, organ in enumerate(self.organs)
        }

    def select_arm(self, organ: str) -> Action:
        return self.bandits[organ].select_arm()

    def commit_block_reward(self, organ: str, arm: Action, reward: float) -> None:
        self.bandits[organ].commit_block_reward(arm, reward)

    def get_statistics_for_organ(self, organ: str) -> Dict[str, Any]:
        return self.bandits[organ].get_statistics()

    def get_all_statistics(self) -> Dict[str, Any]:
        """Comprehensive stats for W&B (mirrors legacy per-organ UCB bandit layout)."""
        per_organ_stats = {
            organ: self.get_statistics_for_organ(organ) for organ in self.organs
        }
        total_pulls = sum(stats["total_pulls"] for stats in per_organ_stats.values())
        n_organs = max(len(self.organs), 1)
        return {
            "per_organ": per_organ_stats,
            "aggregated": {
                "total_organs": len(self.organs),
                "total_blocks": total_pulls // n_organs if total_pulls else 0,
                "total_pulls": total_pulls,
                "avg_pulls_per_organ": total_pulls / n_organs,
                "best_arms_per_organ": self.get_best_arms_per_organ(),
            },
        }

    def export_state(self) -> Dict[str, Any]:
        """Serializable bandit state for checkpoint resume."""
        organs_state: Dict[str, Any] = {}
        for organ, bandit in self.bandits.items():
            organs_state[organ] = {
                "t": bandit.t,
                "N_a": dict(bandit.N_a),
                "Q_a": dict(bandit.Q_a),
                "rewards": {a: list(bandit._rewards[a]) for a in bandit.arms},
                "last_selected": bandit._last_selected,
                "memory": bandit.memory,
                "rng_state": bandit._rng.bit_generator.state,
            }
        return {
            "organs": organs_state,
            "arms": list(self.arms),
            "organs_list": list(self.organs),
            "memory": self.bandits[self.organs[0]].memory if self.organs else "sliding",
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore bandit state from :meth:`export_state`."""
        if list(state.get("organs_list", [])) != list(self.organs):
            raise ValueError("Checkpoint organ list does not match current policy.")
        if list(state.get("arms", [])) != list(self.arms):
            raise ValueError("Checkpoint arms do not match current policy.")
        for organ, blob in state.get("organs", {}).items():
            if organ not in self.bandits:
                continue
            bandit = self.bandits[organ]
            bandit.t = int(blob.get("t", 0))
            bandit.N_a = {a: int(blob["N_a"][a]) for a in bandit.arms}
            bandit.Q_a = {a: float(blob["Q_a"][a]) for a in bandit.arms}
            saved_rewards = blob.get("rewards", {})
            for arm in bandit.arms:
                bandit._rewards[arm] = [float(r) for r in saved_rewards.get(arm, [])]
                hist = bandit._reward_history(arm)
                bandit.Q_a[arm] = float(np.mean(hist)) if hist else 0.0
            bandit._last_selected = blob.get("last_selected")
            rng_state = blob.get("rng_state")
            if rng_state is not None:
                bandit._rng.bit_generator.state = rng_state

    def get_best_arms_per_organ(self) -> Dict[str, Action]:
        """Arm with highest Q_a per organ (for test-time prompting)."""
        best: Dict[str, Action] = {}
        for organ, bandit in self.bandits.items():
            if not bandit.arms:
                continue
            best[organ] = max(bandit.arms, key=lambda a: bandit.Q_a.get(a, 0.0))
        return best


class BlockScheduler:
    """
    Hold bandit arms fixed for K training batches; commit rewards only at block end.

    ``begin_block`` selects arms (no t/N_a increment). ``end_block`` commits curriculum
    rewards and advances bandit time steps.
    """

    def __init__(
        self,
        policy: PerOrganBanditPolicy,
        block_size_batches: int = 50,
    ) -> None:
        if block_size_batches < 1:
            raise ValueError("block_size_batches must be >= 1")
        self.policy = policy
        self.block_size_batches = int(block_size_batches)
        self.state = BlockState(arms_by_organ={}, batches_in_block=0)

    @property
    def current_arms(self) -> Dict[str, Action]:
        return dict(self.state.arms_by_organ)

    def begin_block(self, organs_present: List[str]) -> Dict[str, Action]:
        """Select and freeze one arm per present organ for the upcoming block."""
        arms: Dict[str, Action] = {}
        for organ in organs_present:
            if organ in self.policy.bandits:
                arms[organ] = self.policy.select_arm(organ)
        self.state = BlockState(arms_by_organ=arms, batches_in_block=0)
        return dict(arms)

    def register_batch(self, n_batches: int = 1) -> bool:
        """
        Increment the in-block **dataloader-step** counter by ``n_batches``.

        Counts loader iterations, not samples. Passing ``len(batch)`` would treat
        ``block_size_batches`` as a sample budget (wrong for ``batch_size > 1``).
        Returns True if the block just completed.
        """
        self.state.batches_in_block += int(n_batches)
        return self.state.batches_in_block >= self.block_size_batches

    def end_block(self, rewards: Mapping[str, float]) -> None:
        """
        Commit rewards for organs in the current block; reset block state.
        """
        for organ, arm in self.state.arms_by_organ.items():
            if organ in rewards:
                self.policy.commit_block_reward(organ, arm, float(rewards[organ]))
        self.state = BlockState(arms_by_organ={}, batches_in_block=0)

    def needs_new_block(self) -> bool:
        return len(self.state.arms_by_organ) == 0

    def export_state(self) -> Dict[str, Any]:
        return {
            "arms_by_organ": dict(self.state.arms_by_organ),
            "batches_in_block": int(self.state.batches_in_block),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.state = BlockState(
            arms_by_organ=dict(state.get("arms_by_organ", {})),
            batches_in_block=int(state.get("batches_in_block", 0)),
        )
