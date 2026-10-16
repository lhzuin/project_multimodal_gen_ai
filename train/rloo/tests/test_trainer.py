from pathlib import Path
import tempfile
import unittest

import torch

from train.grpo import GRPOConfig, GRPOTrainer
from train.grpo.tests.helpers import MemoryServer, TinyPolicy
from train.rl.contracts import PolicyDecision, PositionPrompt
from train.rloo import RLOOConfig, RLOOTrainer
from train.rloo.objective import RLOODecision


def config(**kwargs):
    values = dict(
        training_batch_size=2,
        rollouts_per_entry=4,
        max_concurrent_games=2,
        sampling_batch_size=2,
        micro_batch_size=2,
        ply_depth=4,
        device="cpu",
        poll_timeout_ms=1,
    )
    values.update(kwargs)
    return RLOOConfig(**values)


def trainer(cfg=None, server=None):
    return RLOOTrainer(
        database=[PositionPrompt(str(i)) for i in range(4)],
        server=server or MemoryServer(),
        checkpoint_path=None,
        model_type="encoder",
        config=cfg or config(),
        policy=TinyPolicy(),
    )


class TrainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_rollout_updates_once_and_reference_stays_fixed(self):
        instance = trainer()
        original = instance.policy.model.weight.detach().clone()
        reference = instance.reference.model.weight.detach().clone()
        metrics = await instance.train_batch([0, 1])
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(metrics["valid_groups"], 2)
        self.assertEqual(metrics["policy_decisions"], 16)
        self.assertFalse(torch.equal(original, instance.policy.model.weight))
        torch.testing.assert_close(reference, instance.reference.model.weight)
        self.assertIsNone(instance.reference.model.weight.grad)
        self.assertFalse(instance.policy.model.training)
        self.assertLessEqual(instance.server.max_active, 2)
        self.assertEqual(instance.server.coordinator.get_games(), [])

    async def test_no_kl_omits_reference_and_still_trains(self):
        instance = trainer(config(kl_beta=0))
        self.assertIsNone(instance.reference)
        result = await instance.train_batch([0])
        self.assertEqual(result["optimizer_steps"], 1)

    async def test_micro_batch_accumulation_matches_full_batch(self):
        small = trainer(config(micro_batch_size=1, kl_beta=0))
        large = trainer(config(micro_batch_size=8, kl_beta=0))
        observation = {"legal_moves_uci": ["a2a3", "a2a4", "b2b3"]}
        with torch.no_grad():
            logs = small.policy.log_distributions(
                [observation], small.config.temperature
            )[0]
        items = [
            RLOODecision(
                PolicyDecision(observation, i % 3, float(logs[i % 3]), None),
                1 if i % 2 else -1,
                1 / 4,
            )
            for i in range(7)
        ]
        small.update(items)
        large.update(items)
        torch.testing.assert_close(small.policy.model.weight, large.policy.model.weight)

    async def test_resume_matches_uninterrupted_data_rng_policy_and_reference(self):
        uninterrupted, partial = trainer(), trainer()
        await uninterrupted.fit(epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            await partial.fit(epochs=1, max_updates=1, checkpoint_path=path)
            restored = trainer()
            restored.resume(path)
            self.assertEqual(restored._cursor, 2)
            await restored.fit(epochs=1)
            self.assertEqual(restored.optimizer_steps, uninterrupted.optimizer_steps)
            torch.testing.assert_close(
                restored.policy.model.weight, uninterrupted.policy.model.weight
            )
            torch.testing.assert_close(
                restored.reference.model.weight, uninterrupted.reference.model.weight
            )
            self.assertEqual(restored.rng.getstate(), uninterrupted.rng.getstate())
            torch.testing.assert_close(
                restored.collector.sampling_rng.get_state(),
                uninterrupted.collector.sampling_rng.get_state(),
            )

    async def test_rejects_grpo_resume_checkpoint(self):
        original = GRPOTrainer(
            database=[PositionPrompt(str(i)) for i in range(4)],
            server=MemoryServer(),
            checkpoint_path=None,
            model_type="encoder",
            config=GRPOConfig(device="cpu"),
            policy=TinyPolicy(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grpo.pt"
            original.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "not a RLOO resume"):
                trainer().resume(path)

    async def test_detects_policy_change_since_collection(self):
        instance = trainer(config(kl_beta=0))
        observation = {"legal_moves_uci": ["a2a3", "a2a4"]}
        with torch.no_grad():
            old = instance.policy.log_distributions([observation], 1)[0][0].item()
            instance.policy.model.weight.add_(1)
        item = RLOODecision(PolicyDecision(observation, 0, old, None), 1, 1)
        with self.assertRaisesRegex(ValueError, "fresh on-policy samples"):
            instance.update([item])
        self.assertEqual(instance.optimizer_steps, 0)
        self.assertIsNone(instance.policy.model.weight.grad)
