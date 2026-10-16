"""GRPO-specific configuration, preserving the original public constructor."""

from dataclasses import dataclass
import math

from train.rl.config import RolloutConfig


@dataclass(frozen=True)
class GRPOConfig(RolloutConfig):
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-6
    update_epochs: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if not math.isfinite(self.advantage_epsilon) or self.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be finite and positive")
        if (
            isinstance(self.update_epochs, bool)
            or not isinstance(self.update_epochs, int)
            or self.update_epochs < 1
        ):
            raise ValueError("update_epochs must be a positive integer")
