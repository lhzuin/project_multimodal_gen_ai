"""RLOO settings; one fresh on-policy update per collected rollout batch."""

from dataclasses import dataclass
import math
from typing import Literal

from train.rl.config import RolloutConfig


@dataclass(frozen=True)
class RLOOConfig(RolloutConfig):
    rollouts_per_entry: int = 4
    max_concurrent_games: int = 8
    sampling_batch_size: int = 4
    micro_batch_size: int = 2
    reward_mode: Literal["tanh", "linear"] = "tanh"
    reward_scale_cp: float = 400.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reward_mode not in ("tanh", "linear"):
            raise ValueError("reward_mode must be tanh or linear")
        if not math.isfinite(self.reward_scale_cp) or self.reward_scale_cp <= 0:
            raise ValueError("reward_scale_cp must be finite and positive")
