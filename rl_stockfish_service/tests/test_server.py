from __future__ import annotations

import unittest

from rl_stockfish_service.client import AsyncRolloutClient, RolloutAPIError
from rl_stockfish_service.config import ServiceConfig
from rl_stockfish_service.coordinator import RolloutCoordinator
from rl_stockfish_service.server import RolloutTCPServer
from rl_stockfish_service.tests.test_coordinator import FakePool


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        coordinator = RolloutCoordinator(
            opponent_pool=FakePool(),
            evaluator_pool=FakePool(),
            config=ServiceConfig(),
        )
        self.server = RolloutTCPServer(coordinator, host="127.0.0.1", port=0)
        try:
            await self.server.start()
        except PermissionError:
            self.skipTest("runtime sandbox forbids loopback sockets")
        port = self.server.sockets[0].getsockname()[1]
        self.client = AsyncRolloutClient("127.0.0.1", port)

    async def asyncTearDown(self) -> None:
        await self.server.close()

    async def test_health_and_rollout_over_wire(self) -> None:
        health = await self.client.health()
        self.assertEqual(health["api_version"], "1")
        await self.client.create_games(
            [{"game_id": "wire", "policy_color": "white", "horizon_plies": 1}]
        )
        turn = (await self.client.poll_policy_turns())[0]
        await self.client.submit_actions(
            [{"request_id": turn["request_id"], "move_uci": "e2e4"}]
        )
        result = (await self.client.poll_results())[0]
        self.assertEqual(result["game_id"], "wire")
        self.assertEqual(result["termination"], "horizon")

    async def test_invalid_game_returns_api_error(self) -> None:
        with self.assertRaises(RolloutAPIError):
            await self.client.create_games([{"policy_color": "purple"}])

    async def test_stateless_play_positions_over_wire(self) -> None:
        result = await self.client.play_positions(
            [
                {"id": "first", "moves_uci": ["e2e4"]},
                {"id": "second", "moves_uci": ["d2d4"]},
            ]
        )
        self.assertEqual([r["id"] for r in result], ["first", "second"])
        self.assertTrue(all(r["error"] is None for r in result))
        self.assertEqual(await self.client.get_games(), [])


if __name__ == "__main__":
    unittest.main()
