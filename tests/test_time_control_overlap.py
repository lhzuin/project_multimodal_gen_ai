"""Canonical-game identity and category-matched overlap sensitivity."""

import io
from pathlib import Path
import sqlite3
import unittest

import chess.pgn

from analysis.time_control_comparison.overlap import identity, local_path
from analysis.time_control_comparison.sensitivity import matched_table


class OverlapTests(unittest.TestCase):
    def test_identity_ignores_headers_but_retains_the_initial_position(self):
        game = chess.pgn.read_game(io.StringIO("1. e4 e5 *"))
        original = identity(game)
        game.headers["Event"] = "Another publication"
        self.assertEqual(identity(game), original)
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
        self.assertNotEqual(identity(game), original)

    def test_saved_paths_relocate_from_both_operating_systems(self):
        root = Path("/project")
        for value in ("/old/project/processed_games/plan", r"C:\old\processed_games\plan"):
            self.assertEqual(local_path(value, root), root / "processed_games/plan")
        with self.assertRaises(ValueError):
            local_path("unrelated/path", root)

    def test_matched_cohorts_use_rank_and_preserve_position_weighting(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE games(game_id TEXT, category TEXT)")
            connection.execute("CREATE TABLE predictions(game_id TEXT, model TEXT, top1_correct INT, target_logp REAL)")
            selected = []
            for category in ("classic", "rapid", "blitz"):
                for rank in (2, 1):
                    identifier = f"{category}-{rank}"
                    selected.append(dict(game_id=identifier, category=category, rank=rank))
                    connection.execute("INSERT INTO games VALUES (?,?)", (identifier, category))
                    connection.executemany("INSERT INTO predictions VALUES (?,?,?,?)",
                                           [(identifier, "classic", rank == 1, -float(rank))] * rank)
            result = matched_table(connection, selected, quota=1)
            for category in result:
                self.assertEqual(result[category]["games"], 1)
                self.assertEqual(result[category]["models"]["classic"]["positions"], 1)
                self.assertEqual(result[category]["models"]["classic"]["agreement"], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 9)


if __name__ == "__main__":
    unittest.main()
