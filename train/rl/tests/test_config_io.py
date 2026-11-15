"""YAML entry points and immutable settings equivalence for migrated runs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from train.rl.config_io import dump_config, load_config
from train.rl.launch import launch_main
from train.rloo import RLOOConfig


class ConfigIOTests(unittest.TestCase):
    def test_v1_matches_canonical_original_json_settings(self):
        # Historical v1 settings, with machine-specific executable paths normalized to stockfish.
        expected = {
            "v1": "6872b7cc77dc5eb68d7bcf3b323cc69a3919d9c46251acd0ad1f98dcf3047525",
        }
        for version, digest in expected.items():
            with self.subTest(version=version):
                config = load_config(
                    Path(f"configs/rl/rloo_encoder_lichess_mixed_{version}.yaml")
                )
                canonical = json.dumps(
                    config, sort_keys=True, separators=(",", ":")
                ).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
                RLOOConfig(**config["rloo"])
                self.assertIsInstance(config["rloo"]["learning_rate"], float)
                self.assertIs(config["rloo"]["randomize_policy_color"], False)

    def test_current_rl_configs_validate_and_have_separate_outputs(self):
        outputs = []
        for version in ("v1", "v2", "v3", "v4", "v5"):
            config = load_config(Path(f"configs/rl/rloo_encoder_lichess_mixed_{version}.yaml"))
            RLOOConfig(**config["rloo"])
            outputs.extend((config["output_checkpoint"], config["export_weights"]))
            self.assertEqual(config["checkpoint_path"], "checkpoints/chess_encoder_seq_balanced_v1.best.pt")
            self.assertNotIn("resume_path", config)
        self.assertEqual(len(set(outputs)), len(outputs))

    def test_v4_continuation_preserves_training_settings_and_pilot_paths(self):
        pilot = load_config(Path("configs/rl/rloo_encoder_lichess_mixed_v4.yaml"))
        continued = load_config(Path("configs/rl/rloo_encoder_lichess_mixed_v4_continue.yaml"))
        self.assertEqual(continued["rloo"], pilot["rloo"])
        self.assertEqual(continued["server"], pilot["server"])
        self.assertEqual(continued["resume_path"], pilot["output_checkpoint"])
        self.assertNotEqual(continued["output_checkpoint"], pilot["output_checkpoint"])
        self.assertNotEqual(continued["export_weights"], pilot["export_weights"])

    def test_roundtrip_comments_yml_and_cli(self):
        expected = {
            "rloo": {"learning_rate": 5e-6, "randomize_policy_color": False},
            "resume_path": None,
            "epochs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yml"
            path.write_text("# Experiment configuration\n" + dump_config(expected))
            self.assertEqual(load_config(path), expected)
            received = []

            async def run(config):
                received.append(config)

            with patch("sys.argv", ["rloo", "--config", str(path)]):
                launch_main(run, "rloo")
            self.assertEqual(received, [expected])

    def test_rejects_unsupported_extensions_duplicates_and_unsafe_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run.json"
            path.write_text('{"epochs": 1}')
            with self.assertRaisesRegex(ValueError, "yaml"):
                load_config(path)
            for text in (
                "epochs: 1\nepochs: 2\n",
                "- not-a-mapping\n",
                "",
                "1: value\n",
            ):
                path = root / "run.yaml"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)
            path.write_text("!!python/object/apply:os.system ['echo unsafe']")
            with self.assertRaises(yaml.YAMLError):
                load_config(path)
