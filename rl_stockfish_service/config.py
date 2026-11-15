"""Validated runtime configuration for the rollout service."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import Any

import chess.engine


@dataclass(frozen=True)
class SearchLimit:
    """A reproducible Stockfish search budget.

    At least one of ``movetime_ms``, ``depth``, or ``nodes`` must be set. When
    several are set, python-chess forwards all of them and the first limit
    reached stops the search.
    """

    movetime_ms: int | None = None
    depth: int | None = None
    nodes: int | None = None

    def __post_init__(self) -> None:
        values = (self.movetime_ms, self.depth, self.nodes)
        if all(value is None for value in values):
            raise ValueError("a search limit requires movetime_ms, depth, or nodes")
        for name, value in zip(("movetime_ms", "depth", "nodes"), values):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_chess(self) -> chess.engine.Limit:
        return chess.engine.Limit(
            time=None if self.movetime_ms is None else self.movetime_ms / 1000.0,
            depth=self.depth,
            nodes=self.nodes,
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "movetime_ms": self.movetime_ms,
            "depth": self.depth,
            "nodes": self.nodes,
        }


@dataclass(frozen=True)
class EnginePoolConfig:
    """Configuration for independent, single-owner Stockfish processes."""

    executable: str
    workers: int = 1
    threads_per_engine: int = 1
    hash_mb_per_engine: int = 64
    queue_capacity: int = 4096
    default_limit: SearchLimit = field(
        default_factory=lambda: SearchLimit(movetime_ms=50)
    )
    limit_strength: bool = False
    uci_elo: int | None = None
    skill_level: int | None = None
    restart_attempts: int = 1
    clear_hash_before_search: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.clear_hash_before_search, bool):
            raise ValueError("clear_hash_before_search must be a boolean")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.threads_per_engine < 1:
            raise ValueError("threads_per_engine must be at least 1")
        if self.hash_mb_per_engine < 1:
            raise ValueError("hash_mb_per_engine must be at least 1")
        if self.queue_capacity < self.workers:
            raise ValueError("queue_capacity must be at least the worker count")
        if self.restart_attempts < 0:
            raise ValueError("restart_attempts cannot be negative")
        if self.skill_level is not None and not 0 <= self.skill_level <= 20:
            raise ValueError("skill_level must be in [0, 20]")
        if self.uci_elo is not None and not self.limit_strength:
            raise ValueError("uci_elo requires limit_strength=True")

    def validate_executable(self) -> Path:
        executable = self.executable
        if executable == "stockfish":
            executable = os.environ.get("STOCKFISH_EXECUTABLE", executable)
        path = Path(shutil.which(executable) or executable).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Stockfish executable not found: {path}")
        return path

    def public_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "workers": self.workers,
            "threads_per_engine": self.threads_per_engine,
            "hash_mb_per_engine": self.hash_mb_per_engine,
            "queue_capacity": self.queue_capacity,
            "default_limit": self.default_limit.to_dict(),
            "limit_strength": self.limit_strength,
            "uci_elo": self.uci_elo,
            "skill_level": self.skill_level,
            "restart_attempts": self.restart_attempts,
            "clear_hash_before_search": self.clear_hash_before_search,
        }


@dataclass(frozen=True)
class ServiceConfig:
    """Coordinator and transport settings."""

    host: str = "127.0.0.1"
    port: int = 8765
    policy_queue_capacity: int = 4096
    result_queue_capacity: int = 4096
    max_batch_size: int = 512
    max_request_bytes: int = 8 * 1024 * 1024
    mate_score_cp: int = 100_000
    reward_scale_cp: float = 400.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.policy_queue_capacity < 1 or self.result_queue_capacity < 1:
            raise ValueError("queue capacities must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes is unreasonably small")
        if self.mate_score_cp <= 0 or self.reward_scale_cp <= 0:
            raise ValueError("reward scales must be positive")
