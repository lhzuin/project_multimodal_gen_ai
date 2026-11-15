"""Transport-neutral request and result contract.

The wire server uses JSON, but these types deliberately contain no transport
logic. A future HTTP, gRPC, or distributed-queue adapter can therefore retain
the same policy/rollout contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

import chess


class PolicyColor(StrEnum):
    WHITE = "white"
    BLACK = "black"
    RANDOM = "random"


class GameStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_POLICY = "waiting_policy"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Termination(StrEnum):
    HORIZON = "horizon"
    CHECKMATE = "checkmate"
    DRAW = "draw"
    ILLEGAL_POLICY_MOVE = "illegal_policy_move"
    MAX_PLIES = "max_plies"
    ENGINE_ERROR = "engine_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GameSpec:
    """One rollout requested by the trainer.

    ``horizon_plies`` counts newly played plies after ``opening_moves_uci``.
    The observation always returns ``start_fen`` and complete relative move
    history, which is enough for both board encoders and causal decoders.
    """

    start_fen: str = chess.STARTING_FEN
    opening_moves_uci: tuple[str, ...] = ()
    policy_color: PolicyColor = PolicyColor.RANDOM
    horizon_plies: int = 32
    max_plies: int = 400
    game_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_plies < 0:
            raise ValueError("horizon_plies cannot be negative")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")
        if self.horizon_plies > self.max_plies:
            raise ValueError("horizon_plies cannot exceed max_plies")
        board = chess.Board(self.start_fen)
        for move_uci in self.opening_moves_uci:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError as exc:
                raise ValueError(f"invalid opening UCI move: {move_uci}") from exc
            if move not in board.legal_moves:
                raise ValueError(f"illegal opening move {move_uci} for {board.fen()}")
            board.push(move)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GameSpec":
        try:
            color = PolicyColor(str(value.get("policy_color", PolicyColor.RANDOM)))
        except ValueError as exc:
            raise ValueError("policy_color must be white, black, or random") from exc
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        moves = value.get("opening_moves_uci", ())
        if not isinstance(moves, (list, tuple)) or not all(isinstance(x, str) for x in moves):
            raise ValueError("opening_moves_uci must be a list of strings")
        return cls(
            start_fen=str(value.get("start_fen", chess.STARTING_FEN)),
            opening_moves_uci=tuple(moves),
            policy_color=color,
            horizon_plies=int(value.get("horizon_plies", 32)),
            max_plies=int(value.get("max_plies", 400)),
            game_id=None if value.get("game_id") is None else str(value["game_id"]),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class EngineEvaluation:
    """Stockfish value from the requested player's point of view."""

    centipawns: int
    mate: int | None
    depth: int | None
    nodes: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "centipawns": self.centipawns,
            "mate": self.mate,
            "depth": self.depth,
            "nodes": self.nodes,
        }
