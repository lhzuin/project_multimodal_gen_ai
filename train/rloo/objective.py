"""REINFORCE leave-one-out over policy trajectories, including KL in returns."""

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from train.rl.contracts import PolicyDecision, Trajectory
from .config import RLOOConfig


@dataclass(frozen=True)
class RLOODecision:
    decision: PolicyDecision
    advantage: float
    weight: float


def leave_one_out_advantages(returns: torch.Tensor) -> torch.Tensor:
    """Each sample uses only the other group members as its reward baseline."""
    if returns.ndim != 1 or returns.numel() < 2 or not torch.isfinite(returns).all():
        raise ValueError("RLOO needs at least two finite scalar returns")
    # Algebraically identical to subtracting the mean of all *other* returns.
    return (returns - returns.mean()) * returns.numel() / (returns.numel() - 1)


def trajectory_sampled_kl(trajectory: Trajectory) -> float:
    """Sum log pi_behavior(a|s) - log pi_reference(a|s) for policy actions.

    Individual sampled values may be negative. Their on-policy expectation is
    the trajectory KL; opponent transition probabilities cancel in the ratio.
    """
    total = 0.0
    for decision in trajectory.decisions:
        if decision.reference_log_probs is None:
            raise ValueError("reference probabilities missing from rollout")
        total += (
            decision.old_log_prob - decision.reference_log_probs[decision.action_index]
        )
    if not math.isfinite(total):
        raise ValueError("nonfinite sampled trajectory KL")
    return total


def training_items(
    trajectories: Sequence[Trajectory],
    config: RLOOConfig,
) -> tuple[list[RLOODecision], dict[str, float]]:
    groups: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id, []).append(trajectory)
    valid = [
        group
        for group in groups.values()
        if (
            len(group) == config.rollouts_per_entry
            and all(
                t.result is not None and t.result.get("status") == "completed"
                for t in group
            )
        )
    ]
    items: list[RLOODecision] = []
    scores, rewards, kls, returns_all = [], [], [], []
    reward_spreads, reward_stds, abs_advantages = [], [], []
    reward_zero_variance = 0
    zero_variance = 0
    for group in valid:
        group_returns = []
        group_rewards = []
        for trajectory in group:
            cp = trajectory.score_cp
            if not math.isfinite(cp):
                raise ValueError("nonfinite centipawn score")
            scaled = cp / config.reward_scale_cp
            reward = math.tanh(scaled) if config.reward_mode == "tanh" else scaled
            kl = trajectory_sampled_kl(trajectory) if config.kl_beta > 0 else 0.0
            value = reward - config.kl_beta * kl
            scores.append(cp)
            rewards.append(reward)
            group_rewards.append(reward)
            kls.append(kl)
            returns_all.append(value)
            group_returns.append(value)
        values = torch.tensor(group_returns, dtype=torch.float64)
        advantages = leave_one_out_advantages(values).tolist()
        abs_advantages.extend(abs(value) for value in advantages)
        spread = max(group_rewards) - min(group_rewards)
        reward_spreads.append(spread)
        reward_zero_variance += int(spread == 0)
        reward_mean = sum(group_rewards) / len(group_rewards)
        reward_stds.append(
            math.sqrt(sum((r - reward_mean) ** 2 for r in group_rewards) / len(group_rewards))
        )
        zero_variance += int(torch.all(values == values[0]))
        # Average groups and trajectories, but SUM policy actions within each
        # trajectory. Dividing by its length would change episodic REINFORCE.
        weight = 1.0 / (len(valid) * len(group))
        for trajectory, advantage in zip(group, advantages, strict=True):
            items.extend(
                RLOODecision(d, advantage, weight) for d in trajectory.decisions
            )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return items, {
        "valid_groups": float(len(valid)),
        "dropped_groups": float(len(groups) - len(valid)),
        "zero_variance_groups": float(zero_variance),
        # Raw environment rewards can be flat while KL makes regularized
        # returns vary. Keep these diagnostics separate from zero_variance_groups.
        "reward_zero_variance_groups": float(reward_zero_variance),
        "reward_low_spread_groups": float(sum(s < 0.02 for s in reward_spreads)),
        "mean_reward_group_spread": mean(reward_spreads),
        "mean_reward_group_std": mean(reward_stds),
        "reward_saturation_fraction": (
            mean([float(abs(r) > 0.99) for r in rewards])
            if config.reward_mode == "tanh" else 0.0
        ),
        "mean_abs_advantage": mean(abs_advantages),
        "mean_abs_kl_penalty": mean([config.kl_beta * abs(kl) for kl in kls]),
        "mean_score_cp": mean(scores),
        "mean_reward": mean(rewards),
        "mean_return": mean(returns_all),
        "mean_sampled_kl": mean(kls),
        "policy_decisions": float(len(items)),
        "no_policy_trajectories": float(
            sum(not t.decisions for group in valid for t in group)
        ),
    }


def reinforce_loss(
    selected_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted trajectory score-function loss, accumulated across micro-batches."""
    return -(selected_log_probs * advantages.detach() * weights).sum()
