"""Separate RLOO optimizer loop, sharing only environment/policy lifecycle."""

from typing import Any, Sequence

import torch

from train.rl.contracts import Trajectory
from train.rl.trainer import OnPolicyTrainer
from train.rl.optimization import clip_gradients
from .config import RLOOConfig
from .objective import RLOODecision, reinforce_loss, training_items


class RLOOTrainer(OnPolicyTrainer):
    algorithm = "rloo"
    checkpoint_format = "chess-rloo-v1"
    config: RLOOConfig

    def __init__(self, *, config: RLOOConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config=config or RLOOConfig(), **kwargs)

    def training_items(
        self,
        trajectories: Sequence[Trajectory],
    ) -> tuple[list[RLOODecision], dict[str, float]]:
        return training_items(trajectories, self.config)

    def update(self, items: Sequence[RLOODecision]) -> dict[str, float]:
        """One REINFORCE step; gradient clipping, but no PPO ratio clipping."""
        if not items:
            return {"loss": 0.0, "optimizer_steps": float(self.optimizer_steps)}
        self.policy.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        try:
            for start in range(0, len(items), self.config.micro_batch_size):
                batch = items[start : start + self.config.micro_batch_size]
                distributions = self.policy.log_distributions(
                    [item.decision.observation for item in batch],
                    self.config.temperature,
                )
                log_probs = torch.stack(
                    [
                        log[item.decision.action_index]
                        for log, item in zip(distributions, batch, strict=True)
                    ]
                )
                behavior = log_probs.new_tensor(
                    [item.decision.old_log_prob for item in batch]
                )
                if not torch.allclose(
                    log_probs.detach(), behavior, atol=1e-4, rtol=1e-4
                ):
                    raise ValueError(
                        "RLOO requires fresh on-policy samples; policy changed since collection"
                    )
                advantages = log_probs.new_tensor([item.advantage for item in batch])
                weights = log_probs.new_tensor([item.weight for item in batch])
                loss = reinforce_loss(log_probs, advantages, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite RLOO loss")
                loss.backward()
                loss_total += float(loss.detach().item())
            gradient_metrics = clip_gradients(
                self.policy.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer_steps += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "loss": loss_total,
            **gradient_metrics,
            "optimizer_steps": float(self.optimizer_steps),
        }
