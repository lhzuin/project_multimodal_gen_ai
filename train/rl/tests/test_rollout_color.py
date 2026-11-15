from pathlib import Path
import random
import tempfile
import unittest

import chess
import torch

from train.grpo import GRPOConfig
from train.grpo.tests.helpers import MemoryServer, TinyPolicy
from train.rl.contracts import PositionPrompt
from train.rl.rollouts import RolloutCollector
from train.rloo import RLOOConfig, RLOOTrainer


def config(config_type=RLOOConfig, **kwargs):
    return config_type(
        rollouts_per_entry=2,
        max_concurrent_games=2,
        sampling_batch_size=2,
        micro_batch_size=1,
        ply_depth=1,
        device="cpu",
        poll_timeout_ms=1,
        kl_beta=0,
        **kwargs,
    )


class RolloutColorTests(unittest.IsolatedAsyncioTestCase):
    async def test_side_to_move_uses_history_and_base_fen_for_both_algorithms(self):
        prompts = [
            PositionPrompt("white"),
            PositionPrompt("black-after-opening", moves_uci=("e2e4",)),
            PositionPrompt("white-after-opening", moves_uci=("e2e4", "e7e5")),
            PositionPrompt(
                "black-base", start_fen=chess.STARTING_FEN.replace(" w ", " b ")
            ),
            PositionPrompt(
                "white-after-black-base",
                start_fen=chess.STARTING_FEN.replace(" w ", " b "),
                moves_uci=("e7e5",),
            ),
        ]
        for config_type in (GRPOConfig, RLOOConfig):
            with self.subTest(algorithm=config_type.__name__):
                server = MemoryServer()
                rng = random.Random(42)
                original_rng = rng.getstate()
                collector = RolloutCollector(
                    server,
                    TinyPolicy(),
                    None,
                    config(config_type, randomize_policy_color=False),
                    rng,
                )
                trajectories = await collector.collect(prompts)
                boards = {p.entry_id: p.board() for p in prompts}
                self.assertEqual(len(trajectories), 2 * len(prompts))
                for rollout in trajectories:
                    board = boards[rollout.result["metadata"]["entry_id"]]
                    self.assertEqual(
                        rollout.result["policy_color"],
                        "white" if board.turn else "black",
                    )
                    self.assertEqual(len(rollout.decisions), 1)
                    self.assertEqual(rollout.decisions[0].observation["rollout_ply"], 0)
                    self.assertEqual(
                        rollout.decisions[0].observation["fen"], board.fen()
                    )
                self.assertTrue(
                    all(len(colors) == 1 for colors in server.colors.values())
                )
                self.assertEqual(rng.getstate(), original_rng)

    async def test_default_randomization_keeps_seeded_group_behavior(self):
        prompts = [PositionPrompt(str(i)) for i in range(6)]
        rng = random.Random(42)
        expected_rng = random.Random(42)
        expected = {
            p.entry_id: expected_rng.choice(("white", "black")) for p in prompts
        }
        server = MemoryServer()
        collector = RolloutCollector(server, TinyPolicy(), None, config(), rng)
        trajectories = await collector.collect(prompts)
        for rollout in trajectories:
            entry = rollout.result["metadata"]["entry_id"]
            self.assertEqual(rollout.result["policy_color"], expected[entry])
        self.assertEqual(rng.getstate(), expected_rng.getstate())
        self.assertTrue(all(len(colors) == 1 for colors in server.colors.values()))

    async def test_config_requires_actual_boolean(self):
        for config_type in (GRPOConfig, RLOOConfig):
            for value in ("false", 0, None):
                with self.subTest(algorithm=config_type.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "must be a boolean"):
                        config(config_type, randomize_policy_color=value)

    async def test_legacy_checkpoint_resumes_as_random_but_rejects_mode_change(self):
        def make_trainer(randomize):
            return RLOOTrainer(
                database=[PositionPrompt("position")],
                server=MemoryServer(),
                checkpoint_path=None,
                model_type="encoder",
                policy=TinyPolicy(),
                config=config(randomize_policy_color=randomize),
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            make_trainer(True).save_checkpoint(path)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["config"]["randomize_policy_color"]
            torch.save(checkpoint, path)
            make_trainer(True).resume(path)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                make_trainer(False).resume(path)
