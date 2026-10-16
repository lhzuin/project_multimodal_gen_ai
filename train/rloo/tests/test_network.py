import unittest

from rl_stockfish_service import AsyncRolloutClient, RolloutTCPServer
from train.grpo.tests.helpers import MemoryServer
from train.rloo.tests.test_trainer import trainer


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_through_stockfish_v1_wire_contract(self):
        environment = MemoryServer()
        server = RolloutTCPServer(environment.coordinator, port=0)
        try:
            await server.start()
        except PermissionError:
            self.skipTest("runtime sandbox forbids loopback sockets")
        try:
            port = server.sockets[0].getsockname()[1]
            instance = trainer(server=AsyncRolloutClient(port=port))
            metrics = await instance.train_batch([0, 1])
            self.assertEqual(metrics["optimizer_steps"], 1)
            self.assertEqual(metrics["valid_groups"], 2)
        finally:
            await server.close()
            await environment.coordinator.shutdown()
