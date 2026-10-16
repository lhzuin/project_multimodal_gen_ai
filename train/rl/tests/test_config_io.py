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
    def test_v1_v2_match_canonical_original_json_settings(self):
        expected = {
            "v1": "a63151d1e127368969d90edee765159e55206c01ae17c261f5d562ecb4657644",
            "v2": "be3271a30ee0618349593a09671d4b6a4213e92e9779bbb588ac0a809fff6c31",
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
