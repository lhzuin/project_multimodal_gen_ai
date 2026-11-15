"""Standalone YAML RLOO launcher, independent of Hydra and GRPO settings."""

from typing import Any

from train.rl.launch import launch_main, run_training
from .config import RLOOConfig
from .trainer import RLOOTrainer


async def run(config: dict[str, Any]) -> None:
    await run_training(
        config, trainer_type=RLOOTrainer, config_type=RLOOConfig, algorithm="rloo"
    )


def main() -> None:
    launch_main(run, "rloo")


if __name__ == "__main__":
    main()
