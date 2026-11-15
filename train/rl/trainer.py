"""Algorithm-independent on-policy lifecycle over the Stockfish v1 API."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
import random
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .checkpoints import atomic_save
from .config import RolloutConfig
from .contracts import (
    PolicyAdapter,
    PositionDatabase,
    PositionPrompt,
    RolloutServer,
    Trajectory,
)
from .policies import load_policy, select_device
from .rollouts import RolloutCollector


class OnPolicyTrainer(ABC):
    """Keep policy, behavior statistics, reference, and environment separate.

    ``database`` is an indexable PositionDatabase, not a raw PGN or SQLite path.
    An upstream dataset adapter chooses the key moments and exposes the prompt
    contract. Pass an AsyncRolloutClient, LocalStockfishServer, or any v1 server.

    Eval mode deliberately disables dropout for both sampling and gradient
    replay. Eval mode does not disable gradients. Changing dropout masks would
    corrupt likelihood ratios even before the first optimizer step.
    """

    algorithm: str
    checkpoint_format: str
    updates_per_batch: int = 1

    def __init__(
        self,
        *,
        database: PositionDatabase,
        server: RolloutServer,
        checkpoint_path: str | Path | None,
        model_type: str,
        config: RolloutConfig,
        model_kwargs: Mapping[str, Any] | None = None,
        policy: PolicyAdapter | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if model_type not in ("encoder", "decoder"):
            raise ValueError("model_type must be encoder or decoder")
        if len(database) < 1:
            raise ValueError("database cannot be empty")
        self.config = config
        self.database = database
        self.server = server
        self.model_type = model_type
        self.model_kwargs = dict(model_kwargs or {})
        self.original_checkpoint_path = (
            None if checkpoint_path is None else str(checkpoint_path)
        )
        self.device = select_device(self.config.device)
        if policy is None:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required unless a policy is injected"
                )
            policy = load_policy(checkpoint_path, model_type, self.model_kwargs)
        self.policy = policy
        self.policy.model.to(self.device).eval()
        self.reference = copy.deepcopy(policy) if self.config.kl_beta > 0 else None
        if self.reference is not None:
            self.reference.model.requires_grad_(False).eval()
        parameters = [p for p in self.policy.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("policy has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.rng = random.Random(self.config.seed)
        self.collector = RolloutCollector(
            server,
            self.policy,
            self.reference,
            self.config,
            self.rng,
        )
        self.optimizer_steps = 0
        self.completed_batches = 0
        self.logging_run_id: str | None = None
        self.epoch = 0
        self._order: list[int] = []
        self._cursor = 0
        self._busy = False

    def _prompts(self, indices: Sequence[int]) -> list[PositionPrompt]:
        prompts = []
        for index in indices:
            item = self.database[index]
            prompt = (
                item
                if isinstance(item, PositionPrompt)
                else PositionPrompt.from_mapping(item)
            )
            prompt.validate(self.model_type)
            prompts.append(prompt)
        return prompts

    @abstractmethod
    def training_items(
        self, trajectories: Sequence[Trajectory]
    ) -> tuple[Sequence[Any], dict[str, float]]:
        """Algorithm extension point for alternative advantage/reward rules."""
        raise NotImplementedError

    @abstractmethod
    def update(self, items: Sequence[Any]) -> dict[str, float]:
        """Accumulate an algorithm-specific objective and update the policy."""
        raise NotImplementedError

    async def train_batch(self, indices: Sequence[int]) -> dict[str, float]:
        if self._busy:
            raise RuntimeError(
                "concurrent training calls would invalidate on-policy statistics"
            )
        if not indices:
            raise ValueError("training indices cannot be empty")
        self._busy = True
        try:
            started = time.monotonic()
            prompts = self._prompts(indices)
            trajectories = await self.collector.collect(prompts)
            health = await self.server.health()
            collected = time.monotonic()
            items, metrics = self.training_items(trajectories)
            for role, kind in (("opponent", "play"), ("evaluator", "evaluate")):
                searches = health.get(role + "_pool", {}).get("searches", {}).get(kind, {})
                for name in ("nodes_mean", "depth_mean"):
                    if name in searches:
                        metrics[f"server_{role}_{name}_cumulative"] = float(searches[name])
            # Training stays synchronous once collection completes: no sampled
            # action can race with optimizer changes to the policy weights.
            metrics.update(self.update(items))
            finished = time.monotonic()
            metrics.update(
                rollout_seconds=collected - started,
                update_seconds=finished - collected,
                batch_seconds=finished - started,
            )
            self.completed_batches += 1
            metrics["completed_batches"] = float(self.completed_batches)
            return metrics
        finally:
            self._busy = False

    async def fit(
        self,
        *,
        epochs: int = 1,
        max_updates: int | None = None,
        checkpoint_path: str | Path | None = None,
        save_every_batches: int = 1,
        export_weights_path: str | Path | None = None,
        export_every_batches: int | None = None,
        on_metrics: Callable[[Mapping[str, float]], None] | None = None,
    ) -> None:
        """Train to an absolute epoch/update target; resume at batch boundaries."""
        if (
            epochs < 1
            or save_every_batches < 1
            or (max_updates is not None and max_updates < 1)
        ):
            raise ValueError(
                "epochs, save_every_batches, and max_updates must be positive"
            )
        if export_every_batches is not None and (
            export_every_batches < 1 or export_weights_path is None
        ):
            raise ValueError("periodic exports need a positive interval and export_weights_path")
        if max_updates is not None and max_updates % self.updates_per_batch:
            raise ValueError("max_updates must align with optimizer batch boundaries")
        while self.epoch < epochs:
            if max_updates is not None and self.optimizer_steps >= max_updates:
                break
            if not self._order:
                self._order = list(range(len(self.database)))
                if self.config.shuffle:
                    self.rng.shuffle(self._order)
                self._cursor = 0
            indices = self._order[
                self._cursor : self._cursor + self.config.training_batch_size
            ]
            metrics = await self.train_batch(indices)
            self._cursor += len(indices)
            if self._cursor >= len(self._order):
                self.epoch += 1
                self._order = []
                self._cursor = 0
            metrics["epoch_progress"] = self.epoch + self._cursor / len(self.database)
            metrics["target_batches"] = float(
                math.ceil(len(self.database) / self.config.training_batch_size) * epochs
            )
            if checkpoint_path and self.completed_batches % save_every_batches == 0:
                self.save_checkpoint(checkpoint_path)
            if export_every_batches and self.completed_batches % export_every_batches == 0:
                path = Path(export_weights_path)
                path = path.with_name(f"{path.stem}.step-{self.optimizer_steps:06d}{path.suffix}")
                self.export_weights(path)
            if on_metrics:
                on_metrics(metrics)
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

    def save_checkpoint(self, path: str | Path) -> None:
        if self._busy:
            raise RuntimeError(
                "checkpoints are saved only at completed batch boundaries"
            )
        atomic_save(
            {
                "format": self.checkpoint_format,
                "model": self.policy.model.state_dict(),
                "reference": (
                    None
                    if self.reference is None
                    else self.reference.model.state_dict()
                ),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "model_type": self.model_type,
                "model_kwargs": self.model_kwargs,
                "original_checkpoint_path": self.original_checkpoint_path,
                "optimizer_steps": self.optimizer_steps,
                "completed_batches": self.completed_batches,
                "logging_run_id": self.logging_run_id,
                "epoch": self.epoch,
                "order": self._order,
                "cursor": self._cursor,
                "database_length": len(self.database),
                "rng": self.rng.getstate(),
                "sampling_rng": self.collector.sampling_rng.get_state(),
            },
            path,
        )

    def resume(self, path: str | Path) -> None:
        if self._busy:
            raise RuntimeError("cannot resume while training is active")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != self.checkpoint_format:
            raise ValueError(f"not a {self.algorithm.upper()} resume checkpoint")
        if (
            state["model_type"] != self.model_type
            or state["model_kwargs"] != self.model_kwargs
        ):
            raise ValueError("resume model architecture differs")
        if state["database_length"] != len(self.database):
            raise ValueError("resume requires the same database snapshot")
        old_config = dict(state["config"])
        # Older v1 checkpoints always randomized color, before this switch
        # existed. Preserve their behavior without requiring a format bump.
        old_config.setdefault("randomize_policy_color", True)
        current_config = asdict(self.config)
        # The same algorithm/data cursor can move to another device or use a
        # different micro-batch size without changing its objective.
        for key in (
            "device",
            "micro_batch_size",
            "sampling_batch_size",
            "max_concurrent_games",
        ):
            old_config.pop(key)
            current_config.pop(key)
        if old_config != current_config:
            raise ValueError("resume training configuration differs")
        self.policy.model.load_state_dict(state["model"], strict=True)
        if self.reference is not None:
            self.reference.model.load_state_dict(state["reference"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_steps = state["optimizer_steps"]
        self.completed_batches = state["completed_batches"]
        self.logging_run_id = state.get("logging_run_id")
        self.epoch, self._order, self._cursor = (
            state["epoch"],
            state["order"],
            state["cursor"],
        )
        self.rng.setstate(state["rng"])
        self.collector.sampling_rng.set_state(state["sampling_rng"])
        self.policy.model.eval()

    def export_weights(self, path: str | Path) -> None:
        """Export weights compatible with existing tournament/model loaders."""
        if self._busy:
            raise RuntimeError("cannot export during training")
        atomic_save(self.policy.model.state_dict(), path)
