"""Architecture arguments must reach the UCI wrapper for either model type."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.tournament_runner import build_parser, launch_model_uci_engine


class LauncherTests(unittest.TestCase):
    def test_layer_count_is_forwarded_once_for_encoder_and_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "engine.py"
            wrapper.touch()
            for model_type in ("encoder", "decoder"):
                with self.subTest(model_type=model_type):
                    args = build_parser().parse_args([
                        "--openings_json", "unused", "--stockfish", "unused",
                        "--model_type", model_type, "--ckpt", "unused",
                        "--model_uci_py", str(wrapper), "--n_layers", "6",
                        "--tokenizer_path", "unused-tokenizer",
                    ])
                    with patch("utils.tournament_runner.chess.engine.SimpleEngine.popen_uci") as launch:
                        launch_model_uci_engine(args)
                    command = launch.call_args.args[0]
                    self.assertEqual(command.count("--n_layers"), 1)
                    self.assertEqual(command[command.index("--n_layers") + 1], "6")
