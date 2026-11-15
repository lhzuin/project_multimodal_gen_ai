"""Algorithm-independent rollout and optimizer settings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RolloutConfig:
    training_batch_size: int = 4  # dataset entries per effective optimizer batch
    rollouts_per_entry: int = 8
    max_concurrent_games: int = 32
    sampling_batch_size: int = 16
    micro_batch_size: int = 8  # policy decisions per backward pass
    ply_depth: int = 10  # newly played plies, both players
    randomize_policy_color: bool = True  # False: policy plays the selected side to move
    temperature: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    kl_beta: float = 0.04
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    poll_timeout_ms: int = 100
    rollout_timeout_s: float = 300.0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.randomize_policy_color, bool):
            raise ValueError("randomize_policy_color must be a boolean")
        for name in (
            "training_batch_size",
            "max_concurrent_games",
            "sampling_batch_size",
            "micro_batch_size",
            "ply_depth",
            "poll_timeout_ms",
            "rollouts_per_entry",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rollouts_per_entry < 2:
            raise ValueError("group baselines need at least two rollouts_per_entry")
        for name in (
            "temperature",
            "learning_rate",
            "max_grad_norm",
            "rollout_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("kl_beta", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.poll_timeout_ms > 300_000:
            raise ValueError("poll_timeout_ms exceeds the API's 300000 ms limit")
