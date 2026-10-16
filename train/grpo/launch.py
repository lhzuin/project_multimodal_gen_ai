"""GRPO YAML entry point over the shared launcher."""

from typing import Any

from train.rl.launch import launch_main, run_training
from .config import GRPOConfig
from .trainer import GRPOTrainer


async def run(config: dict[str, Any]) -> None:
    await run_training(
        config, trainer_type=GRPOTrainer, config_type=GRPOConfig, algorithm="grpo"
    )


def main() -> None:
    launch_main(run, "grpo")


if __name__ == "__main__":
    main()
