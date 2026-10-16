"""Shared YAML/argparse launcher infrastructure; no algorithm selection magic."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from rl_stockfish_service import (
    AsyncRolloutClient,
    EnginePoolConfig,
    SearchLimit,
    ServiceConfig,
)

from .config import RolloutConfig
from .config_io import load_config
from .metrics import MetricsLogger
from .server import LocalStockfishServer
from .trainer import OnPolicyTrainer


def _database(config: dict[str, Any]) -> Any:
    """Open a preselected prompt manifest or call an upstream database factory.

    This does not implement position/puzzle selection or a new dataset class.
    A factory receives kwargs and returns the documented indexable interface.
    """
    if "factory" in config:
        module, name = config["factory"].split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        return factory(**config.get("kwargs", {}))
    with Path(config["manifest"]).open(encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list):
        raise ValueError("prompt manifest must contain a JSON list")
    return prompts


def _pool(config: dict[str, Any]) -> EnginePoolConfig:
    values = dict(config)
    if "default_limit" in values:
        values["default_limit"] = SearchLimit(**values["default_limit"])
    return EnginePoolConfig(**values)


async def run_training(
    config: dict[str, Any],
    *,
    trainer_type: type[OnPolicyTrainer],
    config_type: type[RolloutConfig],
    algorithm: str,
) -> None:
    database = _database(config["database"])
    training_config = config_type(**config.get(algorithm, {}))
    server_config = config.get("server", {})

    async def train(server: Any) -> None:
        trainer = trainer_type(
            database=database,
            server=server,
            checkpoint_path=config["checkpoint_path"],
            model_type=config["model_type"],
            model_kwargs=config.get("model_kwargs", {}),
            config=training_config,
        )
        if config.get("resume_path"):
            trainer.resume(config["resume_path"])
        output = config.get(
            "output_checkpoint", f"checkpoints/chess_{algorithm}.latest.pt"
        )
        logging_settings = dict(config.get("logging", {}))
        # Persistent local metrics by default, including for older configs.
        # Set jsonl_path=null to explicitly disable the journal.
        logging_settings.setdefault(
            "jsonl_path", str(Path(output).with_suffix(".metrics.jsonl"))
        )
        with MetricsLogger(
            logging_settings,
            run_config=config,
            algorithm=algorithm,
            run_id=trainer.logging_run_id,
            resumed=bool(config.get("resume_path")),
        ) as logger:
            trainer.logging_run_id = logger.run_id
            await trainer.fit(
                epochs=config.get("epochs", 1),
                max_updates=config.get("max_updates"),
                checkpoint_path=output,
                save_every_batches=config.get("save_every_batches", 1),
                on_metrics=logger,
            )
            if config.get("export_weights"):
                trainer.export_weights(config["export_weights"])

    if server_config.get("mode", "remote") == "remote":
        await train(
            AsyncRolloutClient(
                host=server_config.get("host", "127.0.0.1"),
                port=server_config.get("port", 8765),
                timeout_s=server_config.get("timeout_s", 30.0),
            )
        )
    elif server_config["mode"] == "local":
        async with LocalStockfishServer(
            _pool(server_config["opponent"]),
            _pool(server_config["evaluator"]),
            ServiceConfig(**server_config.get("service", {})),
        ) as server:
            await train(server)
    else:
        raise ValueError("server mode must be remote or local")


def launch_main(
    run: Callable[[dict[str, Any]], Awaitable[None]], algorithm: str
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train a chess policy with {algorithm.upper()}"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="YAML experiment configuration"
    )
    args = parser.parse_args()
    asyncio.run(run(load_config(args.config)))
