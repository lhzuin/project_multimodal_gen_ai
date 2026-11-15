"""Opt-in real UCI model + Stockfish smoke test, using temporary result files."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import chess
import chess.pgn


class RealTournamentTests(unittest.TestCase):
    def _run(self, model_type):
        stockfish = os.environ.get("RL_STOCKFISH_BIN")
        if not stockfish or not os.environ.get("ASYNC_TOURNAMENT_REAL_MODELS"):
            self.skipTest("set RL_STOCKFISH_BIN and ASYNC_TOURNAMENT_REAL_MODELS=1")
        with tempfile.TemporaryDirectory(prefix="async-tournament-smoke-") as directory:
            root = Path(directory)
            suite = root / "suite.json"
            board = chess.Board()
            prefix = ["e2e4", "e7e5", "g1f3", "b8c6"]
            for uci in prefix:
                board.push_uci(uci)
            suite.write_text(
                json.dumps([{"id": "smoke", "fen": board.fen(), "uci_prefix": prefix}])
            )
            checkpoint = {
                "encoder": "checkpoints/chess_encoder_seq_balanced_v1.best.pt",
                "decoder": "checkpoints/chess_llm_decoder_balanced_v1.best.pt",
            }[model_type]
            command = [
                sys.executable,
                "utils/tournament_runner_async.py",
                "--openings_json",
                str(suite),
                "--stockfish",
                stockfish,
                "--model_type",
                model_type,
                "--version",
                "2",
                "--ckpt",
                checkpoint,
                "--greedy",
                "--movetime_ms",
                "0",
                "--depth",
                "1",
                "--max_plies",
                "2",
                "--hash_mb",
                "16",
                "--max_concurrent_games",
                "2",
                "--stockfish_workers",
                "2",
                "--out_dir",
                str(root / "out"),
                "--write_pgn",
            ]
            if model_type == "decoder":
                command += [
                    "--tokenizer_path",
                    "tokenizers/chess_uci_vocab.json",
                    "--n_layers",
                    "12",
                ]
            subprocess.run(
                command,
                check=True,
                timeout=180,
                capture_output=True,
                text=True,
            )
            output = root / "out"
            summary = json.loads((output / "summary.json").read_text())
            games = [
                json.loads(line)
                for line in (output / "games.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["n_games"], 2)
            self.assertEqual([g["model_color"] for g in games], ["white", "black"])
            self.assertTrue(all(g["termination"] == "max_plies" for g in games))
            with (output / "games.pgn").open() as handle:
                for _ in range(2):
                    game = chess.pgn.read_game(handle)
                    self.assertIsNotNone(game)
                    self.assertFalse(game.errors)
                    self.assertEqual(len(list(game.mainline_moves())), 2)
            execution = json.loads((output / "execution.json").read_text())
            self.assertEqual(
                execution["server_health_at_end"]["evaluator_pool"]["submitted"], 0
            )
            self.assertTrue(
                execution["server_health_at_end"]["opponent_pool"]["engine_ids"]
            )

    def test_original_encoder_cli_and_pgn(self):
        self._run("encoder")

    def test_original_decoder_cli_and_pgn(self):
        self._run("decoder")
