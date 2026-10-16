import asyncio
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

import torch

from rl_stockfish_service.config import ServiceConfig
from train.grpo import GRPOConfig, GRPOTrainer, PositionPrompt
from train.grpo.contracts import PolicyDecision
from train.grpo.objective import WeightedDecision
from train.grpo.rollouts import RolloutCollector
from train.grpo.tests.helpers import MemoryServer, TinyPolicy


def config(**kwargs):
    return GRPOConfig(
        training_batch_size=2,
        rollouts_per_entry=4,
        max_concurrent_games=2,
        sampling_batch_size=2,
        micro_batch_size=2,
        ply_depth=4,
        device="cpu",
        poll_timeout_ms=1,
        **kwargs,
    )


def trainer(server=None, cfg=None, database=None):
    return GRPOTrainer(
        database=database or [PositionPrompt(str(i)) for i in range(4)],
        server=server or MemoryServer(),
        checkpoint_path=None,
        model_type="encoder",
        config=cfg or config(),
        policy=TinyPolicy(),
    )


class TrainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_collection_limits_color_context_and_policy_mask(self):
        server = MemoryServer(
            ServiceConfig(policy_queue_capacity=1, result_queue_capacity=1)
        )
        policy = TinyPolicy()
        collector = RolloutCollector(
            server, policy, None, config(kl_beta=0), random.Random(2)
        )
        trajectories = await collector.collect([PositionPrompt("opening")])
        self.assertEqual(len(trajectories), 4)
        self.assertLessEqual(server.max_active, 2)
        self.assertTrue(all(len(colors) == 1 for colors in server.colors.values()))
        for rollout in trajectories:
            self.assertEqual(rollout.result["rollout_plies"], 4)
            self.assertEqual(len(rollout.decisions), 2)
            policy_color = rollout.result["policy_color"]
            for decision in rollout.decisions:
                turn = decision.observation["fen"].split()[1]
                self.assertEqual(turn, "w" if policy_color == "white" else "b")
                self.assertTrue(decision.observation["moves_uci"] or turn == "w")
        self.assertEqual(server.coordinator.get_games(), [])

    async def test_training_changes_policy_and_reference_stays_frozen(self):
        instance = trainer()
        before = instance.policy.model.weight.detach().clone()
        reference = instance.reference.model.weight.detach().clone()
        metrics = await instance.train_batch([0, 1])
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertGreater(metrics["policy_decisions"], 0)
        self.assertFalse(torch.equal(before, instance.policy.model.weight))
        torch.testing.assert_close(reference, instance.reference.model.weight)
        self.assertIsNone(instance.reference.model.weight.grad)
        self.assertFalse(instance.policy.model.training)

    async def test_micro_batch_accumulation_matches_full_batch(self):
        small = trainer(cfg=replace(config(kl_beta=0), micro_batch_size=1))
        large = trainer(cfg=replace(config(kl_beta=0), micro_batch_size=8))
        observation = {"legal_moves_uci": ["a2a3", "a2a4", "b2b3"]}
        with torch.no_grad():
            log = small.policy.log_distributions([observation], 1)[0]
        items = [
            WeightedDecision(
                PolicyDecision(observation, i % 3, float(log[i % 3]), None),
                1 if i % 2 else -1,
                1 / 7,
            )
            for i in range(7)
        ]
        small.update(items)
        large.update(items)
        torch.testing.assert_close(small.policy.model.weight, large.policy.model.weight)

    async def test_checkpoint_resume_preserves_reference_rng_and_data_cursor(self):
        uninterrupted = trainer()
        interrupted = trainer()
        await uninterrupted.fit(epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            await interrupted.fit(epochs=1, max_updates=1, checkpoint_path=path)
            resumed = trainer()
            resumed.resume(path)
            self.assertEqual(resumed._cursor, 2)
            await resumed.fit(epochs=1)
            self.assertEqual(resumed.optimizer_steps, uninterrupted.optimizer_steps)
            torch.testing.assert_close(
                resumed.policy.model.weight,
                uninterrupted.policy.model.weight,
            )
            torch.testing.assert_close(
                resumed.reference.model.weight,
                uninterrupted.reference.model.weight,
            )
            self.assertEqual(resumed.rng.getstate(), uninterrupted.rng.getstate())
            torch.testing.assert_close(
                resumed.collector.sampling_rng.get_state(),
                uninterrupted.collector.sampling_rng.get_state(),
            )

    async def test_abort_cleans_up_games_on_policy_error(self):
        server = MemoryServer()
        instance = trainer(server=server)

        def fail(*args):
            raise RuntimeError("test sampler failure")

        instance.policy.log_distributions = fail
        with self.assertRaisesRegex(RuntimeError, "test sampler failure"):
            await instance.train_batch([0])
        self.assertEqual(server.coordinator.get_games(), [])

    async def test_cancel_unblocks_full_policy_queue(self):
        server = MemoryServer(ServiceConfig(policy_queue_capacity=1))
        await server.create_games(
            [
                {
                    "game_id": str(i),
                    "policy_color": "white",
                    "metadata": {"group_id": "g"},
                }
                for i in range(3)
            ]
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(server.cancel_games(["0", "1", "2"]), 1)
        self.assertTrue(
            all(g["status"] == "cancelled" for g in server.coordinator.get_games())
        )

    async def test_decoder_data_contract_rejected_before_game_creation(self):
        instance = trainer(
            database=[PositionPrompt("bad", start_fen="8/8/8/8/8/8/4K3/6k1 w - - 0 1")]
        )
        instance.model_type = "decoder"
        with self.assertRaises(ValueError):
            await instance.train_batch([0])
        self.assertEqual(instance.server.coordinator.get_games(), [])
