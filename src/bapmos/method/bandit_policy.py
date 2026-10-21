"""
Variance-bounded UCB-Tuned bandit for per-organ prompt selection.

Action space A = {'box', 'point', 'both'}. Each organ maintains an independent
bandit with a FIFO sliding window of block rewards in [0, 1] (default W=50).

For arm a after t block updates (t incremented only in ``commit_block_reward``)::

    σ²_a = Var(rewards in window)   (sample variance, ddof=1)
    V_a  = σ²_a + sqrt(2 ln(max(1,t)) / N_a)
    Ṽ_a  = min(0.25, V_a)
    Bonus_a = sqrt(ln(max(1,t)) / N_a · Ṽ_a)
    UCB_a = Q_a + Bonus_a

Arms with N_a = 0 score +inf. Selection is round-robin for the first
``warmup_blocks_per_arm * len(arms)`` blocks, then uniform-random among arms with
fewer than ``min_pulls_per_arm`` commits, then argmax UCB (random tie-break).

**Critical:** ``select_arm()`` does NOT increment t or N_a. Only ``commit_block_reward``
increments them when validation rewards are committed at block end.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

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
        Commit reward (clipped to [0, 1]) at block end: update the window, Q_a, N_a, and t.
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
        """Select the next arm for ``organ`` (does not advance t or N_a)."""
        return self.bandits[organ].select_arm()

    def commit_block_reward(self, organ: str, arm: Action, reward: float) -> None:
        """Commit one block reward for ``organ``'s ``arm`` (advances that organ's t and N_a)."""
        self.bandits[organ].commit_block_reward(arm, reward)

    def select_block_arms(self, organs: List[str]) -> Dict[str, Action]:
        """Arms for one block: an independent draw per organ (``organs`` order)."""
        return {organ: self.select_arm(organ) for organ in organs if organ in self.bandits}

    def commit_block_rewards(
        self,
        arms_by_organ: Mapping[str, Action],
        rewards: Mapping[str, float],
    ) -> None:
        """Commit each organ's block reward to that organ's bandit."""
        for organ, arm in arms_by_organ.items():
            if organ in rewards:
                self.commit_block_reward(organ, arm, float(rewards[organ]))

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


class GlobalBanditPolicy(PerOrganBanditPolicy):
    """
    One UCB-Tuned bandit shared by every organ (global-selector ablation).

    Each block selects a single arm for all organs, and the committed reward is the
    mean of the per-organ block rewards. Every organ key maps to the same bandit
    object, so test-time prompting (highest Q) uses one arm for all organs.
    """

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
        self.shared = VarianceBoundedUCBTuned(
            arms=self.arms,
            window_size=window_size,
            warmup_blocks_per_arm=warmup_blocks_per_arm,
            min_pulls_per_arm=min_pulls_per_arm,
            seed=seed,
            memory=memory,
        )
        self.bandits = {organ: self.shared for organ in self.organs}

    def select_block_arms(self, organs: List[str]) -> Dict[str, Action]:
        """One draw per block, applied to every listed organ."""
        organs = [o for o in organs if o in self.bandits]
        if not organs:
            return {}
        arm = self.shared.select_arm()
        return {organ: arm for organ in organs}

    def commit_block_rewards(
        self,
        arms_by_organ: Mapping[str, Action],
        rewards: Mapping[str, float],
    ) -> None:
        """Commit the mean reward over organs that received one (one bandit step)."""
        got = [float(rewards[o]) for o in arms_by_organ if o in rewards]
        if not got:
            return
        arm = next(iter(arms_by_organ.values()))
        self.shared.commit_block_reward(arm, float(np.mean(got)))

    def get_all_statistics(self) -> Dict[str, Any]:
        stats = self.shared.get_statistics()
        return {
            "per_organ": {organ: stats for organ in self.organs},
            "aggregated": {
                "total_organs": len(self.organs),
                "total_blocks": stats["total_pulls"],
                "total_pulls": stats["total_pulls"],
                "avg_pulls_per_organ": float(stats["total_pulls"]),
                "best_arms_per_organ": self.get_best_arms_per_organ(),
            },
        }

    def export_state(self) -> Dict[str, Any]:
        state = super().export_state()
        state["scope"] = "global"
        return state


class FixedSchedulePolicy:
    """
    Fixed prompt-mix schedule with no learning (fixed-ratio control).

    Each block draws one arm per organ with probabilities proportional to ``weights``
    (e.g. Box:Point:Both = 2:3:5). Rewards are ignored. Test-time prompting uses the
    highest-weight arm, the convention of the legacy fixed-ratio runs.
    """

    def __init__(
        self,
        organs: List[str],
        weights: Mapping[str, float],
        seed: int = 0,
    ) -> None:
        if len(organs) == 0:
            raise ValueError("organs must be non-empty")
        arms = [a for a in DEFAULT_ARMS if float(weights.get(a, 0.0)) > 0.0]
        if not arms:
            raise ValueError(f"fixed schedule needs a positive weight; got {dict(weights)!r}")
        unknown = set(weights) - set(DEFAULT_ARMS)
        if unknown:
            raise ValueError(f"Unknown arms in fixed weights: {sorted(unknown)}")
        total = sum(float(weights[a]) for a in arms)
        self.organs = list(organs)
        self.arms: List[Action] = arms
        self.probs = np.array([float(weights[a]) / total for a in arms])
        self.bandits = {organ: None for organ in self.organs}
        self._rng = np.random.default_rng(seed)
        self._counts: Dict[str, Dict[Action, int]] = {
            organ: {a: 0 for a in arms} for organ in self.organs
        }

    def select_arm(self, organ: str) -> Action:
        _ = organ
        return self.arms[int(self._rng.choice(len(self.arms), p=self.probs))]

    def select_block_arms(self, organs: List[str]) -> Dict[str, Action]:
        out: Dict[str, Action] = {}
        for organ in organs:
            if organ in self.bandits:
                out[organ] = self.select_arm(organ)
                self._counts[organ][out[organ]] += 1
        return out

    def commit_block_rewards(
        self,
        arms_by_organ: Mapping[str, Action],
        rewards: Mapping[str, float],
    ) -> None:
        """Rewards do not change a fixed schedule."""
        _ = arms_by_organ, rewards

    def get_best_arms_per_organ(self) -> Dict[str, Action]:
        top = self.arms[int(np.argmax(self.probs))]
        return {organ: top for organ in self.organs}

    def get_all_statistics(self) -> Dict[str, Any]:
        top = self.get_best_arms_per_organ()
        per_organ = {
            organ: {
                "t": sum(self._counts[organ].values()),
                "total_pulls": sum(self._counts[organ].values()),
                "N_a": dict(self._counts[organ]),
                "Q_a": {a: 0.0 for a in self.arms},
                "arm_counts": dict(self._counts[organ]),
                "arm_avg_rewards": {a: 0.0 for a in self.arms},
                "best_arm": top[organ],
            }
            for organ in self.organs
        }
        total = sum(s["total_pulls"] for s in per_organ.values())
        return {
            "per_organ": per_organ,
            "aggregated": {
                "total_organs": len(self.organs),
                "total_blocks": total // max(len(self.organs), 1),
                "total_pulls": total,
                "avg_pulls_per_organ": total / max(len(self.organs), 1),
                "best_arms_per_organ": top,
            },
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "scope": "fixed",
            "arms": list(self.arms),
            "probs": [float(p) for p in self.probs],
            "organs_list": list(self.organs),
            "counts": {o: dict(c) for o, c in self._counts.items()},
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if list(state.get("organs_list", [])) != list(self.organs):
            raise ValueError("Checkpoint organ list does not match current policy.")
        if list(state.get("arms", [])) != list(self.arms):
            raise ValueError("Checkpoint arms do not match current policy.")
        self._counts = {
            o: {a: int(n) for a, n in c.items()} for o, c in state.get("counts", {}).items()
        } or self._counts
        if state.get("rng_state") is not None:
            self._rng.bit_generator.state = state["rng_state"]


PromptPolicy = Union[PerOrganBanditPolicy, FixedSchedulePolicy]

BANDIT_SCOPES = ("per_organ", "global", "fixed")


def build_prompt_policy(
    scope: str,
    organs: List[str],
    *,
    arms: Optional[List[Action]] = None,
    window_size: int = 50,
    warmup_blocks_per_arm: int = 10,
    min_pulls_per_arm: int = 5,
    seed: int = 0,
    memory: RewardMemory = "sliding",
    fixed_weights: Optional[Mapping[str, float]] = None,
) -> PromptPolicy:
    """Policy for ``bandit.scope``: ``per_organ`` (BAP-MOS), ``global`` or ``fixed``."""
    if scope == "fixed":
        if not fixed_weights:
            raise ValueError("bandit.scope=fixed requires bandit.fixed_weights")
        return FixedSchedulePolicy(organs, fixed_weights, seed=seed)
    cls = {"per_organ": PerOrganBanditPolicy, "global": GlobalBanditPolicy}.get(scope)
    if cls is None:
        raise ValueError(f"bandit.scope must be one of {BANDIT_SCOPES}, got {scope!r}")
    return cls(
        organs=organs,
        arms=arms,
        window_size=window_size,
        warmup_blocks_per_arm=warmup_blocks_per_arm,
        min_pulls_per_arm=min_pulls_per_arm,
        seed=seed,
        memory=memory,
    )


class BlockScheduler:
    """
    Hold bandit arms fixed for K training batches; commit rewards only at block end.

    ``begin_block`` selects arms (no t/N_a increment). ``end_block`` commits block
    rewards and advances bandit time steps.
    """

    def __init__(
        self,
        policy: "PromptPolicy",
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
        arms = self.policy.select_block_arms(list(organs_present))
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
        self.policy.commit_block_rewards(self.state.arms_by_organ, rewards)
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
