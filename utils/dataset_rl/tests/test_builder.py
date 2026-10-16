from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from utils.dataset_rl.builder import DatasetBuilder, file_hash, split_for_game
from utils.dataset_rl.config import BuildConfig
from utils.dataset_rl.sources import iter_json_array
from utils.dataset_rl.verify_dataset import verify


class FakeConfidence:
    def __init__(self, probability=0.5):
        self.probability = probability

    def score(self, candidates):
        return [
            {"max_probability": self.probability, "entropy_nats": 1.0}
            for _ in candidates
        ]


def make_rows(count=500):
    rows = []
    attempt = 0
    while len(rows) < count:
        rng = random.Random(attempt)
        board = chess.Board()
        sans, ucis, fens = [], [], []
        for _ in range(90):
            if board.is_game_over(claim_draw=True):
                break
            move = rng.choice(list(board.legal_moves))
            fens.append(board.fen())
            sans.append(board.san(move))
            ucis.append(move.uci())
            board.push(move)
        attempt += 1
        if len(ucis) < 70:
            continue
        index = len(rows)
        trigger = 50 + index % 2
        rows.append(
            {
                "PuzzleId": f"puzzle-{index}",
                "GameId": f"game-{index}/black#50",
                "FEN": fens[trigger],
                "Moves": " ".join(ucis[trigger : trigger + 5]),
                "Themes": ["endgame"] if index % 3 == 0 else ["middlegame"],
                "Rating": 1800,
                "RatingDeviation": 75,
                "Popularity": 95,
                "NbPlays": 500,
                "WhiteElo": 2200,
                "BlackElo": 2200,
                "variant": "standard",
                "moves": " ".join(sans),
                "analysis": [json.dumps({"eval": 0})] * len(ucis),
            }
        )
    return rows


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source.parquet"
        pq.write_table(pa.Table.from_pylist(make_rows()), cls.source)
        cls.checkpoint = cls.root / "dummy.pt"
        cls.checkpoint.write_bytes(
            b"fixture checkpoint identity; fake confidence adapter"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, name):
        return BuildConfig(
            output_dir=str(self.root / name),
            local_parquet=(str(self.source),),
            train_size=20,
            validation_size=20,
            test_size=20,
            candidate_multiplier=10,
            confidence_checkpoint=str(self.checkpoint),
            exclusion_openings=(),
        )

    def test_complete_build_is_deterministic_disjoint_and_contract_ready(self):
        first = self.config("first")
        second = self.config("second")
        for config in (first, second):
            report = DatasetBuilder(config, FakeConfidence()).build()
            self.assertEqual(report["actual"], report["quotas"])
            result = verify(Path(config.output_dir))
            self.assertEqual(result["validated_positions"], 60)
            self.assertEqual(result["source_rows_independently_replayed"], 60)
        for split in ("train", "validation", "test"):
            self.assertEqual(
                (Path(first.output_dir) / f"{split}.json").read_bytes(),
                (Path(second.output_dir) / f"{split}.json").read_bytes(),
            )
        with self.assertRaises(FileExistsError):
            DatasetBuilder(first, FakeConfidence()).build()

        # Integrity checking catches modifications before loading examples.
        directory = Path(second.output_dir)
        training_path = directory / "train.json"
        rows = json.loads(training_path.read_text())
        rows[0]["metadata"]["solution"] = "must never be passed to the trainer"
        training_path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            verify(directory)
        # Even with a recomputed checksum, the runtime schema rejects answers.
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["train.json"] = file_hash(training_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unexpected metadata"):
            verify(directory)

    def test_overconfident_positions_are_not_published(self):
        config = self.config("overconfident")
        builder = DatasetBuilder(config, FakeConfidence(0.99))
        with self.assertRaisesRegex(ValueError, "candidate supply exhausted"):
            builder.build()
        self.assertGreater(builder.counts["overconfident_rejected"], 0)
        self.assertFalse(Path(config.output_dir).exists())

    def test_insufficient_supply_does_not_publish_partial_dataset(self):
        config = replace(self.config("insufficient"), min_player_elo=9999)
        with self.assertRaisesRegex(ValueError, "candidate supply exhausted"):
            DatasetBuilder(config, FakeConfidence()).build()
        self.assertFalse(Path(config.output_dir).exists())

    def test_streaming_opening_reader_handles_chunk_boundaries_and_truncation(self):
        path = self.root / "openings.json"
        values = [
            {"fen": chess.STARTING_FEN, "padding": "x" * 100000},
            {"fen": chess.STARTING_FEN},
        ]
        path.write_text(json.dumps(values))
        self.assertEqual(list(iter_json_array(path)), values)
        path.write_text('[{"fen": "broken"}')
        with self.assertRaisesRegex(ValueError, "truncated"):
            list(iter_json_array(path))

    def test_game_split_is_deterministic(self):
        config = self.config("unused")
        self.assertEqual(
            split_for_game("same-game", config), split_for_game("same-game", config)
        )
