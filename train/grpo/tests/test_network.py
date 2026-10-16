"""Verify the trainer uses the same contract over real JSON-lines TCP."""

import unittest

from rl_stockfish_service.client import AsyncRolloutClient
from rl_stockfish_service.server import RolloutTCPServer
from train.grpo.tests.helpers import MemoryServer
from train.grpo.tests.test_trainer import trainer


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = MemoryServer()
        self.server = RolloutTCPServer(self.environment.coordinator, port=0)
        try:
            await self.server.start()
        except PermissionError:
            self.skipTest("runtime sandbox forbids loopback sockets")
        port = self.server.sockets[0].getsockname()[1]
        self.client = AsyncRolloutClient(port=port)

    async def asyncTearDown(self):
        await self.server.close()
        await self.environment.coordinator.shutdown()

    async def test_training_against_network_server(self):
        instance = trainer(server=self.client)
        metrics = await instance.train_batch([0, 1])
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(metrics["valid_groups"], 2)
        self.assertEqual(self.environment.coordinator.get_games(), [])

    async def test_large_observation_response_exceeds_default_stream_limit(self):
        await self.client.create_games(
            [
                {
                    "game_id": "large-observation",
                    "policy_color": "white",
                    "metadata": {"padding": "x" * 100_000},
                }
            ]
        )
        observations = await self.client.poll_policy_turns()
        self.assertEqual(len(observations[0]["metadata"]["padding"]), 100_000)
        await self.client.cancel_games(["large-observation"])
        await self.client.release_games(["large-observation"])
