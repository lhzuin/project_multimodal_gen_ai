"""Opt-in: original encoder/decoder checkpoints, real engine, actual gradients."""

import os
from pathlib import Path
import tempfile
import unittest

import torch

from rl_stockfish_service import EnginePoolConfig, SearchLimit
from train.grpo import GRPOConfig, GRPOTrainer, LocalStockfishServer, PositionPrompt
from train.grpo.launch import run
from train.rl.config_io import load_config


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, model_type, checkpoint, kwargs):
        stockfish = os.environ.get("RL_STOCKFISH_BIN")
        if not stockfish or not os.environ.get("GRPO_REAL_MODELS"):
            self.skipTest(
                "set RL_STOCKFISH_BIN and GRPO_REAL_MODELS=1 for original-model tests"
            )
        if not Path(checkpoint).is_file():
            self.skipTest("original checkpoint not present")
        pool = EnginePoolConfig(
            executable=stockfish,
            workers=1,
            hash_mb_per_engine=16,
            default_limit=SearchLimit(depth=1),
        )
        async with LocalStockfishServer(pool, pool) as server:
            trainer = GRPOTrainer(
                database=[PositionPrompt("opening", moves_uci=("e2e4", "e7e5"))],
                server=server,
                checkpoint_path=checkpoint,
                model_type=model_type,
                model_kwargs=kwargs,
                config=GRPOConfig(
                    training_batch_size=1,
                    rollouts_per_entry=2,
                    max_concurrent_games=2,
                    sampling_batch_size=2,
                    micro_batch_size=1,
                    ply_depth=10,
                    device="cpu",
                    poll_timeout_ms=1,
                ),
            )
            metrics = await trainer.train_batch([0])
            self.assertEqual(metrics["optimizer_steps"], 1)
            self.assertEqual(metrics["valid_groups"], 1)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "latest.pt"
                trainer.save_checkpoint(checkpoint_path)
                trainer.resume(checkpoint_path)
                trainer.export_weights(Path(directory) / "weights.pt")
            self.assertTrue(
                all(torch.isfinite(p).all() for p in trainer.policy.model.parameters())
            )

    async def test_original_encoder(self):
        await self._run(
            "encoder",
            "checkpoints/chess_encoder_seq_balanced_v1.best.pt",
            {
                "version": 2,
                "n_layers": 8,
            },
        )

    async def test_original_decoder(self):
        await self._run(
            "decoder",
            "checkpoints/chess_llm_decoder_balanced_v1.best.pt",
            {
                "n_layers": 12,
                "tokenizer_path": "tokenizers/chess_uci_vocab.json",
            },
        )

    async def test_yaml_launcher_with_owned_server(self):
        stockfish = os.environ.get("RL_STOCKFISH_BIN")
        if not stockfish or not os.environ.get("GRPO_REAL_MODELS"):
            self.skipTest(
                "set RL_STOCKFISH_BIN and GRPO_REAL_MODELS=1 for launcher test"
            )
        configuration = load_config(Path("configs/rl/examples/grpo_encoder.yaml"))
        configuration["grpo"].update(
            {
                "device": "cpu",
                "training_batch_size": 1,
                "rollouts_per_entry": 2,
                "max_concurrent_games": 2,
                "ply_depth": 2,
            }
        )
        configuration["max_updates"] = 1
        pool = {
            "executable": stockfish,
            "workers": 1,
            "hash_mb_per_engine": 16,
            "default_limit": {"depth": 1},
        }
        configuration["server"] = {"mode": "local", "opponent": pool, "evaluator": pool}
        with tempfile.TemporaryDirectory() as directory:
            configuration["output_checkpoint"] = str(Path(directory) / "latest.pt")
            configuration["export_weights"] = str(Path(directory) / "weights.pt")
            await run(configuration)
            state = torch.load(configuration["output_checkpoint"], weights_only=True)
            self.assertEqual(state["optimizer_steps"], 1)
            self.assertTrue(Path(configuration["export_weights"]).is_file())
