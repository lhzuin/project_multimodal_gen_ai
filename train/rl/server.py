"""An optional owned Stockfish environment using exactly the v1 API contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from rl_stockfish_service.config import EnginePoolConfig, ServiceConfig
from rl_stockfish_service.coordinator import RolloutCoordinator
from rl_stockfish_service.engine_pool import StockfishPool
from rl_stockfish_service.protocol import GameSpec
from rl_stockfish_service.position_play import play_positions


class LocalStockfishServer:
    """In-process API adapter, with Stockfish still running as CPU subprocesses.

    Use ``async with`` to own the lifecycle. Pass this object to an RL trainer just
    like AsyncRolloutClient. Transport-free calls preserve the established
    contract and remove local serialization overhead.
    """

    def __init__(
        self,
        opponent_config: EnginePoolConfig,
        evaluator_config: EnginePoolConfig,
        service_config: ServiceConfig | None = None,
    ) -> None:
        self.opponent = StockfishPool(opponent_config, name="rl-opponent")
        self.evaluator = StockfishPool(evaluator_config, name="rl-evaluator")
        self.coordinator = RolloutCoordinator(
            opponent_pool=self.opponent,
            evaluator_pool=self.evaluator,
            config=service_config or ServiceConfig(),
        )

    async def __aenter__(self) -> LocalStockfishServer:
        try:
            await self.opponent.start()
            await self.evaluator.start()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self.coordinator.shutdown()
        await self.opponent.close()
        await self.evaluator.close()

    async def health(self) -> dict[str, Any]:
        return self.coordinator.health()

    async def play_positions(
        self, positions: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return await play_positions(
            self.opponent,
            positions,
            max_batch_size=self.coordinator.config.max_batch_size,
        )

    async def create_games(
        self, games: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return await self.coordinator.create_games(
            [GameSpec.from_dict(g) for g in games]
        )

    async def poll_policy_turns(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]:
        return await self.coordinator.poll_policy_turns(max_items, timeout_ms / 1000)

    async def submit_actions(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return await self.coordinator.submit_actions([dict(a) for a in actions])

    async def poll_results(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]:
        return await self.coordinator.poll_results(max_items, timeout_ms / 1000)

    async def cancel_games(self, game_ids: Sequence[str]) -> list[dict[str, Any]]:
        return await self.coordinator.cancel_games(game_ids)

    async def release_games(self, game_ids: Sequence[str]) -> list[dict[str, Any]]:
        return self.coordinator.release_games(game_ids)
