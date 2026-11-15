"""Opt-in real encoder/decoder and Stockfish tests, using temporary outputs."""

import os
from pathlib import Path
import tempfile
import unittest

import torch

from train.rloo.launch import run
from train.rl.config_io import load_config


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, model_type):
        stockfish = os.environ.get("RL_STOCKFISH_BIN")
        if not stockfish or not os.environ.get("RLOO_REAL_MODELS"):
            self.skipTest("set RL_STOCKFISH_BIN and RLOO_REAL_MODELS=1")
        settings = load_config(Path(f"configs/rl/examples/rloo_{model_type}.yaml"))
        settings["rloo"].update(
            {
                "device": "cpu",
                "training_batch_size": 1,
                "rollouts_per_entry": 2,
                "max_concurrent_games": 2,
                "micro_batch_size": 1,
            }
        )
        settings["max_updates"] = 1
        pool = {
            "executable": stockfish,
            "workers": 1,
            "hash_mb_per_engine": 16,
            "default_limit": {"depth": 1},
        }
        settings["server"] = {"mode": "local", "opponent": pool, "evaluator": pool}
        with tempfile.TemporaryDirectory() as directory:
            settings["output_checkpoint"] = str(Path(directory) / "latest.pt")
            settings["export_weights"] = str(Path(directory) / "weights.pt")
            await run(settings)
            saved = torch.load(settings["output_checkpoint"], weights_only=True)
            self.assertEqual(saved["format"], "chess-rloo-v1")
            self.assertEqual(saved["optimizer_steps"], 1)
            self.assertIsNotNone(saved["reference"])
            self.assertTrue(Path(settings["export_weights"]).is_file())
            # Continue on the second entry after resume: this exercises a
            # non-initial policy, nonzero KL costs, and fresh sampling.
            settings["resume_path"] = settings["output_checkpoint"]
            settings["max_updates"] = 2
            await run(settings)
            resumed = torch.load(settings["output_checkpoint"], weights_only=True)
            self.assertEqual(resumed["optimizer_steps"], 2)
            for name, value in saved["reference"].items():
                torch.testing.assert_close(value, resumed["reference"][name])

    async def test_original_encoder_with_launcher_and_resume(self):
        await self._run("encoder")

    async def test_original_decoder_with_launcher_and_resume(self):
        await self._run("decoder")
