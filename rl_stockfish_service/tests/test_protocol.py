from __future__ import annotations

import unittest

from rl_stockfish_service.config import EnginePoolConfig, SearchLimit
from rl_stockfish_service.protocol import GameSpec, PolicyColor


class ProtocolTests(unittest.TestCase):
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
