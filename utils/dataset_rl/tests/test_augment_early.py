import copy
import random
import unittest

import chess

from utils.dataset_rl.augment_early import augment
from train.rl.contracts import PositionPrompt
from utils.dataset_rl.positions import position_key


def fixtures():
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        rows = []
        for game in range(6):
            rng = random.Random(100 * index + game)
            board = chess.Board()
            for san in ("e4", "e5", "Nf3", "Nc6"):
                board.push_san(san)
            while len(board.move_stack) < 14:
                board.push(rng.choice(list(board.legal_moves)))
            rows.append({"entry_id": f"{split}-{game}", "start_fen": chess.STARTING_FEN,
                         "moves_uci": [m.uci() for m in board.move_stack],
                         "metadata": {"game_id": f"game-{split}-{game}", "bucket": "quiet"}})
        splits[split] = rows
    return splits


class EarlyDatasetTests(unittest.TestCase):
    def test_zero_quota_preserves_original_splits(self):
        source = fixtures()
        output, report = augment(source, dict.fromkeys(source, 0), set())
        self.assertEqual(output, source)
        self.assertTrue(all(row["added_early"] == 0 for row in report["splits"].values()))

    def test_reproducible_balanced_legal_and_strictly_truncated(self):
        source = fixtures()
        before = copy.deepcopy(source)
        counts = dict.fromkeys(source, 2)
        first, report = augment(source, counts, set(), min_ply=4, max_ply=10)
        second, _ = augment(source, counts, set(), min_ply=4, max_ply=10)
        self.assertEqual(first, second)
        self.assertEqual(source, before)
        keys = set()
        for split, rows in first.items():
            self.assertEqual(rows[:6], source[split])
            self.assertEqual(report["splits"][split]["added_colors"], {"white": 1, "black": 1})
            parents = {r["entry_id"]: r for r in source[split]}
            for row in rows:
                prompt = PositionPrompt.from_mapping(row)
                prompt.validate("encoder")
                key = position_key(prompt.board())
                self.assertNotIn(key, keys)
                keys.add(key)
            for row in rows[6:]:
                parent = parents[row["metadata"]["parent_entry_id"]]
                n = len(row["moves_uci"])
                self.assertLess(n, len(parent["moves_uci"]))
                self.assertEqual(row["moves_uci"], parent["moves_uci"][:n])
                # All splits share ply 4, which must never become a new anchor.
                self.assertGreater(n, 4)

    def test_complete_exclusion_prevents_filling_quota(self):
        source = fixtures()
        excluded = set()
        for row in source["train"]:
            board = chess.Board()
            for uci in row["moves_uci"][:10]:
                board.push_uci(uci)
                excluded.add(position_key(board))
        with self.assertRaisesRegex(ValueError, "eligible early states"):
            augment(source, dict.fromkeys(source, 2), excluded, min_ply=4, max_ply=10)

    def test_game_partition_overlap_is_rejected(self):
        source = fixtures()
        source["test"][0]["metadata"]["game_id"] = source["train"][0]["metadata"]["game_id"]
        with self.assertRaisesRegex(ValueError, "source game overlaps"):
            augment(source, dict.fromkeys(source, 2), set(), min_ply=4, max_ply=10)

    def test_duplicate_original_state_is_rejected(self):
        source = fixtures()
        source["test"][0]["moves_uci"] = source["train"][0]["moves_uci"][:]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            augment(source, dict.fromkeys(source, 2), set(), min_ply=4, max_ply=10)

    def test_impossible_or_unbalanced_quota_is_rejected(self):
        source = fixtures()
        for counts in (dict.fromkeys(source, 100), dict.fromkeys(source, 3)):
            with self.assertRaises(ValueError):
                augment(source, counts, set(), min_ply=4, max_ply=10)


if __name__ == "__main__":
    unittest.main()
