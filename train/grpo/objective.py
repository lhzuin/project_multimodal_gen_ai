"""Outcome GRPO: group advantages, clipped policy ratios and exact legal KL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .contracts import PolicyDecision, Trajectory


@dataclass(frozen=True)
class WeightedDecision:
    decision: PolicyDecision
    advantage: float
    weight: float


def group_training_items(
    trajectories: Sequence[Trajectory], epsilon: float
) -> tuple[list[WeightedDecision], dict[str, float]]:
    """Drop entire failed groups; never silently normalize an incomplete group.

    Every group, trajectory, and policy action is averaged separately. This
    preserves GRPO's per-trajectory normalization across mates of unequal length
    and across arbitrary gradient micro-batch boundaries.
    """
    groups: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id, []).append(trajectory)
    valid = [
        group
        for group in groups.values()
        if all(
            t.result is not None
            and t.result.get("status") == "completed"
            and t.decisions
            for t in group
        )
    ]
    items: list[WeightedDecision] = []
    zero_variance = 0
    all_scores: list[float] = []
    for group in valid:
        scores = torch.tensor([t.score_cp for t in group], dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite centipawn score")
        std = scores.std(unbiased=False)
        advantages = (scores - scores.mean()) / (std + epsilon)
        zero_variance += int(std <= epsilon)
        all_scores.extend(scores.tolist())
        for trajectory, advantage in zip(group, advantages.tolist(), strict=True):
            weight = 1.0 / (len(valid) * len(group) * len(trajectory.decisions))
            items.extend(
                WeightedDecision(d, advantage, weight) for d in trajectory.decisions
            )
    return items, {
        "valid_groups": float(len(valid)),
        "dropped_groups": float(len(groups) - len(valid)),
        "zero_variance_groups": float(zero_variance),
        "mean_score_cp": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "policy_decisions": float(len(items)),
    }


def exact_legal_kl(
    log_probs: torch.Tensor, reference_log_probs: torch.Tensor
) -> torch.Tensor:
    """D_KL(current || fixed reference) over the finite set of legal actions."""
    return (log_probs.exp() * (log_probs - reference_log_probs)).sum()


def clipped_grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    kl: torch.Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a weighted sum; callers accumulate it without another division."""
    ratios = (new_log_probs - old_log_probs).exp()
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratios * advantages, clipped * advantages)
    loss = (weights * (-surrogate + kl_beta * kl)).sum()
    return loss, {
        "loss": loss.detach(),
        "kl": (weights * kl).sum().detach(),
        "clip_fraction": (weights * ((ratios - 1).abs() > clip_epsilon)).sum().detach(),
    }
