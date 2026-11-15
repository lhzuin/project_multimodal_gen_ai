"""Dataset, policy, rollout and server interfaces shared by RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import chess
import torch
from torch import nn


@dataclass(frozen=True)
class PositionPrompt:
    """A selected game moment; the future puzzle dataset returns this type.

    ``start_fen`` is the history's base position; ``moves_uci`` ends at the
    selected moment. An encoder may supply a FEN alone. A decoder must provide
    real history from the standard initial position, since its input contains
    no board-state embedding. Puzzle selection/database IO belongs upstream.
    """

    entry_id: str
    start_fen: str = chess.STARTING_FEN
    moves_uci: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def board(self) -> chess.Board:
        board = chess.Board(self.start_fen)
        if not board.is_valid():
            raise ValueError(f"entry {self.entry_id}: invalid board")
        for uci in self.moves_uci:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"entry {self.entry_id}: illegal history move {uci}")
            board.push(move)
        return board

    def validate(self, model_type: str) -> None:
        if not self.entry_id:
            raise ValueError("entry_id cannot be empty")
        if self.board().is_game_over(claim_draw=True):
            raise ValueError(
                f"entry {self.entry_id}: selected position is already terminal"
            )
        if model_type == "decoder" and self.start_fen != chess.STARTING_FEN:
            raise ValueError(
                "decoder prompts need actual history from chess.STARTING_FEN; "
                "the existing decoder cannot condition on an arbitrary FEN"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PositionPrompt:
        moves = value.get("moves_uci", ())
        if not isinstance(moves, (list, tuple)) or not all(
            isinstance(x, str) for x in moves
        ):
            raise ValueError("moves_uci must be a sequence of UCI strings")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        return cls(
            entry_id=str(value["entry_id"]),
            start_fen=str(value.get("start_fen", chess.STARTING_FEN)),
            moves_uci=tuple(moves),
            metadata=dict(metadata),
        )


class PositionDatabase(Protocol):
    """Indexable, finite database of already selected positions.

    Items must be PositionPrompt or mappings with the same fields. Access is
    deterministic for a fixed dataset snapshot. No model tensors, tokenizer
    IDs, target human moves, or RL rewards should be returned by the dataset.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> PositionPrompt | Mapping[str, Any]: ...


class RolloutServer(Protocol):
    """The established Stockfish v1 contract (network or in-process adapter)."""

    async def health(self) -> dict[str, Any]: ...
    async def create_games(
        self, games: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...
    async def poll_policy_turns(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]: ...
    async def submit_actions(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]: ...
    async def poll_results(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]: ...
    async def cancel_games(self, game_ids: Sequence[str]) -> list[dict[str, Any]]: ...
    async def release_games(self, game_ids: Sequence[str]) -> list[dict[str, Any]]: ...


class PolicyAdapter(Protocol):
    """Different algorithms can reuse the same differentiable action boundary.

    Return one normalized log-probability vector per observation, ordered like
    legal_moves_uci. Sampling and gradient replay use this identical function.
    """

    model: nn.Module

    def log_distributions(
        self, observations: Sequence[Mapping[str, Any]], temperature: float
    ) -> list[torch.Tensor]: ...


@dataclass(frozen=True)
class PolicyDecision:
    observation: Mapping[str, Any]
    action_index: int
    old_log_prob: float
    reference_log_probs: tuple[float, ...] | None


@dataclass
class Trajectory:
    game_id: str
    group_id: str
    decisions: list[PolicyDecision] = field(default_factory=list)
    result: Mapping[str, Any] | None = None

    @property
    def score_cp(self) -> float:
        if self.result is None or self.result.get("status") != "completed":
            raise ValueError("only successfully completed games have usable rewards")
        reward = self.result["reward"]
        if reward["perspective"] != "policy":
            raise ValueError("Stockfish reward must use the policy perspective")
        return float(reward["centipawns"])
