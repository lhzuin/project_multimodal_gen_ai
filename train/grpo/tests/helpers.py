"""Small differentiable policies and deterministic engines for RL tests."""

from typing import Any, Mapping, Sequence

import chess
import torch
from torch import nn

from rl_stockfish_service.coordinator import RolloutCoordinator
from rl_stockfish_service.config import ServiceConfig
from rl_stockfish_service.protocol import EngineEvaluation, GameSpec


class TinyPolicy:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(0.1)

    def log_distributions(
        self, observations: Sequence[Mapping[str, Any]], temperature: float
    ) -> list[torch.Tensor]:
        device = self.model.weight.device
        return [
            torch.log_softmax(
                self.model.weight.view(())
                * torch.linspace(-1, 1, len(o["legal_moves_uci"]), device=device)
                / temperature,
                dim=0,
            )
            for o in observations
        ]


class FakeEngine:
    async def play(self, board: chess.Board, limit: Any = None) -> chess.Move:
        return next(iter(board.legal_moves))

    async def evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
        *,
        mate_score_cp: int,
        limit: Any = None,
    ) -> EngineEvaluation:
        # Produces varied, deterministic terminal values for sampled moves.
        value = sum(
            (s + 1) * p.piece_type * (1 if p.color else -1)
            for s, p in board.piece_map().items()
        )
        return EngineEvaluation(value if perspective else -value, None, 1, 1)

    def metrics(self) -> dict[str, int]:
        return {"workers": 1, "queue_size": 0}


class MemoryServer:
    def __init__(self, config: ServiceConfig | None = None) -> None:
        self.coordinator = RolloutCoordinator(
            opponent_pool=FakeEngine(),
            evaluator_pool=FakeEngine(),
            config=config or ServiceConfig(),
        )
        self.max_active = 0
        self.colors: dict[str, set[str]] = {}

    async def health(self):
        return self.coordinator.health()

    async def create_games(self, games):
        result = await self.coordinator.create_games(
            [GameSpec.from_dict(g) for g in games]
        )
        active = sum(
            g["status"] not in ("completed", "failed", "cancelled")
            for g in self.coordinator.get_games()
        )
        self.max_active = max(self.max_active, active)
        for game in games:
            self.colors.setdefault(game["metadata"]["group_id"], set()).add(
                game["policy_color"]
            )
        return result

    async def poll_policy_turns(self, max_items=64, timeout_ms=1000):
        return await self.coordinator.poll_policy_turns(max_items, timeout_ms / 1000)

    async def submit_actions(self, actions):
        return await self.coordinator.submit_actions(actions)

    async def poll_results(self, max_items=64, timeout_ms=1000):
        return await self.coordinator.poll_results(max_items, timeout_ms / 1000)

    async def cancel_games(self, game_ids):
        return await self.coordinator.cancel_games(game_ids)

    async def release_games(self, game_ids):
        return self.coordinator.release_games(game_ids)
