"""GRPO objective and optimizer loop over the shared on-policy lifecycle."""

from typing import Any, Sequence

import torch

from train.rl.contracts import Trajectory
from train.rl.trainer import OnPolicyTrainer
from train.rl.optimization import clip_gradients
from .config import GRPOConfig
from .objective import (
    WeightedDecision,
    clipped_grpo_loss,
    exact_legal_kl,
    group_training_items,
)


class GRPOTrainer(OnPolicyTrainer):
    algorithm = "grpo"
    checkpoint_format = "chess-grpo-v1"
    config: GRPOConfig

    def __init__(self, *, config: GRPOConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config=config or GRPOConfig(), **kwargs)

    @property
    def updates_per_batch(self) -> int:
        return self.config.update_epochs

    def training_items(
        self, trajectories: Sequence[Trajectory]
    ) -> tuple[list[WeightedDecision], dict[str, float]]:
        return group_training_items(trajectories, self.config.advantage_epsilon)

    def update(self, items: Sequence[WeightedDecision]) -> dict[str, float]:
        """Accumulate the original per-group/per-trajectory GRPO mean."""
        totals = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
        if not items:
            return {**totals, "optimizer_steps": float(self.optimizer_steps)}
        self.policy.model.eval()
        gradient_totals: dict[str, float] = {}
        for _ in range(self.config.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_metrics = dict.fromkeys(totals, 0.0)
            try:
                for start in range(0, len(items), self.config.micro_batch_size):
                    batch = items[start : start + self.config.micro_batch_size]
                    logs = self.policy.log_distributions(
                        [item.decision.observation for item in batch],
                        self.config.temperature,
                    )
                    new_log_probs = torch.stack(
                        [
                            log[item.decision.action_index]
                            for log, item in zip(logs, batch, strict=True)
                        ]
                    )
                    old_log_probs = new_log_probs.new_tensor(
                        [item.decision.old_log_prob for item in batch]
                    )
                    advantages = new_log_probs.new_tensor(
                        [item.advantage for item in batch]
                    )
                    weights = new_log_probs.new_tensor([item.weight for item in batch])
                    kl_terms = []
                    for log, item in zip(logs, batch, strict=True):
                        ref = item.decision.reference_log_probs
                        if self.config.kl_beta > 0:
                            if ref is None:
                                raise ValueError(
                                    "reference probabilities missing from rollout"
                                )
                            kl_terms.append(exact_legal_kl(log, log.new_tensor(ref)))
                        else:
                            kl_terms.append(log.new_zeros(()))
                    loss, metrics = clipped_grpo_loss(
                        new_log_probs,
                        old_log_probs,
                        advantages,
                        weights,
                        torch.stack(kl_terms),
                        clip_epsilon=self.config.clip_epsilon,
                        kl_beta=self.config.kl_beta,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite GRPO loss")
                    loss.backward()
                    for key, value in metrics.items():
                        epoch_metrics[key] += float(value.item())
                gradient_metrics = clip_gradients(
                    self.policy.model.parameters(),
                    self.config.max_grad_norm,
                )
                for key, value in gradient_metrics.items():
                    gradient_totals[key] = (
                        gradient_totals.get(key, 0.0)
                        + value / self.config.update_epochs
                    )
                self.optimizer.step()
                self.optimizer_steps += 1
                for key in totals:
                    totals[key] += epoch_metrics[key] / self.config.update_epochs
            finally:
                self.optimizer.zero_grad(set_to_none=True)
        return {
            **totals,
            **gradient_totals,
            "optimizer_steps": float(self.optimizer_steps),
        }
