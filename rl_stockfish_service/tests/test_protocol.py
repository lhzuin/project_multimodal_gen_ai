from __future__ import annotations

import unittest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from rl_stockfish_service.config import EnginePoolConfig, SearchLimit
from rl_stockfish_service.protocol import GameSpec, PolicyColor


class ProtocolTests(unittest.TestCase):
    def test_engine_path_accepts_path_lookup_and_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "stockfish"
            executable.touch(mode=0o755)
            config = EnginePoolConfig(executable="stockfish")
            with patch.dict(os.environ, {"PATH": directory}, clear=True):
                self.assertEqual(config.validate_executable(), executable.resolve())
            with patch.dict(os.environ, {"STOCKFISH_EXECUTABLE": str(executable)}):
                self.assertEqual(config.validate_executable(), executable.resolve())

    def test_cache_reset_setting_is_explicit_boolean(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            EnginePoolConfig(executable="stockfish", clear_hash_before_search="true")
        self.assertFalse(EnginePoolConfig(executable="stockfish").clear_hash_before_search)
    def test_game_spec_validates_opening(self) -> None:
        spec = GameSpec.from_dict(
            {
                "opening_moves_uci": ["e2e4", "e7e5"],
                "policy_color": "black",
                "horizon_plies": 8,
                "metadata": {"group_id": 3},
            }
        )
        self.assertEqual(spec.policy_color, PolicyColor.BLACK)
        self.assertEqual(spec.opening_moves_uci, ("e2e4", "e7e5"))

    def test_game_spec_rejects_illegal_opening(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal opening move"):
            GameSpec(opening_moves_uci=("e2e5",))

    def test_search_limit_requires_budget(self) -> None:
        with self.assertRaises(ValueError):
            SearchLimit()

    def test_elo_requires_limited_strength(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit_strength"):
            EnginePoolConfig(executable="stockfish", uci_elo=1700)


if __name__ == "__main__":
    unittest.main()
