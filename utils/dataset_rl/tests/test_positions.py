import unittest

import chess

from utils.dataset_rl.config import BuildConfig
from utils.dataset_rl.positions import (
    game_id,
    position_key,
    puzzle_candidate,
    quiet_candidate,
)
from utils.dataset_rl.sources import eligible_buckets


def row_fixture():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    return {
        "PuzzleId": "fixture",
        "GameId": "abcdefgh/black#4",
        "FEN": board.fen(),
        "Moves": "b8c6 f1b5 a7a6 b5a4 g8f6",
        "Themes": ["middlegame", "short"],
        "Rating": 1800,
        "RatingDeviation": 75,
        "Popularity": 95,
        "NbPlays": 500,
        "WhiteElo": 2200,
        "BlackElo": 2200,
        "variant": "standard",
        # Original played response Bc4 intentionally differs from solution Bb5.
        "moves": "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6",
        "analysis": [],
    }


class PositionTests(unittest.TestCase):
    def test_trigger_is_applied_and_solution_is_not_appended(self):
        row = row_fixture()
        candidate = puzzle_candidate(row, "tactics")
        self.assertEqual(candidate.prompt.moves_uci, ("e2e4", "e7e5", "g1f3", "b8c6"))
        self.assertTrue(candidate.board.turn)
        self.assertNotIn("f1b5", candidate.prompt.moves_uci)
        self.assertNotIn("f1c4", candidate.prompt.moves_uci)
        self.assertEqual(game_id(row), "abcdefgh")
        self.assertNotIn("Moves", candidate.prompt.metadata)
        candidate.prompt.validate("decoder")

    def test_wrong_fen_or_trigger_is_rejected(self):
        for field, value in (("FEN", chess.STARTING_FEN), ("Moves", "b8a6 f1b5")):
            row = row_fixture()
            row[field] = value
            with self.assertRaises(ValueError):
                puzzle_candidate(row, "tactics")

    def test_uppercase_puzzle_moves_cannot_substitute_for_full_history(self):
        row = row_fixture()
        row["moves"] = row["Moves"]
        with self.assertRaises(ValueError):
            puzzle_candidate(row, "tactics")

    def test_thresholds_and_exclusive_puzzle_categories(self):
        config = BuildConfig()
        row = row_fixture()
        self.assertEqual(eligible_buckets(row, config), ["quiet", "tactics"])
        row["Themes"] = ["endgame"]
        self.assertEqual(eligible_buckets(row, config), ["quiet", "endgame"])
        row["RatingDeviation"] = 200
        self.assertEqual(eligible_buckets(row, config), ["quiet"])
        row["WhiteElo"] = 1000
        self.assertEqual(eligible_buckets(row, config), [])

    def test_no_annotations_does_not_pretend_position_is_quiet(self):
        with self.assertRaisesRegex(ValueError, "no eligible quiet"):
            quiet_candidate(row_fixture(), BuildConfig())

    def test_dedup_key_preserves_turn_but_ignores_counters(self):
        first = chess.Board()
        second = chess.Board(chess.STARTING_FEN.replace("0 1", "4 3"))
        self.assertEqual(position_key(first), position_key(second))
        second.turn = chess.BLACK
        self.assertNotEqual(position_key(first), position_key(second))
