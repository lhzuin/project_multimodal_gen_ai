"""Alignment, paired statistics, provenance and interruption regressions."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch

from analysis.decoder_alignment.__main__ import baseline_row, main
from analysis.decoder_alignment.inference import check_baseline_predictions, predict_game, validate_positions
from analysis.decoder_alignment.report import summarize_arrays
from analysis.decoder_alignment.storage import BASE_FIELDS, Store, run_lock
from analysis.ply_comparison.aggregate import bucket_for_ply, relative_bucket_for_ply
from analysis.ply_comparison.evaluate import PER_MOVE_FIELDS, ROOT, decoder_result, game_examples, sha256
from dataset.offsets import build_pgn_offsets
from models.chess_decoder_player import ChessDecoderPlayer


TOKENIZER = ROOT / "tokenizers/chess_uci_vocab.json"


def make_game(sans=("e4", "e5", "Nf3", "Nc6")):
    game = chess.pgn.Game()
    game.headers.update(Event="Open", WhiteElo="2200", BlackElo="2100", TWICIssue="1667", Date="2026.10.20")
    board, node = game.board(), game
    for san in sans:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    return game


def small_model():
    torch.manual_seed(123)
    return ChessDecoderPlayer(tokenizer_path=str(TOKENIZER), model_dim=32, n_heads=4,
                               n_layers=1, dropout=0.0, max_seq_len=256).eval()


def csv_rows(game, game_id, model):
    tokenizer = model.decoder.tokenizer
    examples = game_examples(game, game_id, tokenizer, include_board_grid=False)
    predictions = predict_game(examples, model, torch.device("cpu"))
    for e, p in zip(examples, predictions):
        yield {"game_id": game_id, "ply": e.ply, "game_plies": e.game_plies,
               "bucket": bucket_for_ply(e.ply), "relative_bucket": relative_bucket_for_ply(e.ply, e.game_plies),
               "model_id": "decoder_baseline", "twic_issue": e.issue, "date": e.date,
               "white_elo": e.white_elo, "black_elo": e.black_elo, "fen_before": e.fen,
               "target_uci": e.target.uci(), **p, "p_target_legal": float(np.exp(-p["nll_legal"])),
               "n_legal": len(e.legal_moves), "context_len": len(e.move_ids),
               "context_truncated": int(e.context_truncated), "pair_correct": ""}


def fixture(root):
    corpus = root / "processed_games/eval"
    corpus.mkdir(parents=True)
    (root / "tokenizers").mkdir()
    (root / "tokenizers/chess_uci_vocab.json").write_bytes(TOKENIZER.read_bytes())
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    base = checkpoints / "chess_llm_decoder_v12.pt"
    candidate = checkpoints / "rloo_decoder_v12_lichess_mixed_v5_horizon8.pt"
    base.write_bytes(b"baseline fixture")
    candidate.write_bytes(b"candidate fixture")
    games = [make_game(), make_game(("d4", "d5", "c4", "e6"))]
    pgn = corpus / "twic_1662_1667_classic.pgn"
    pgn.write_text("\n\n".join(str(g) for g in games) + "\n\n")
    index = corpus / "twic_1662_1667_classic_index.json"
    build_pgn_offsets(str(pgn), str(index))
    saved = root / "outputs/twic_1662_1667"
    saved.mkdir(parents=True)
    manifest = {"format": "twic_disjoint_comparison_v1", "complete": True, "full_corpus": True,
                "excluded_games": {}, "accepted_games": 2, "corpus_games": 2,
                "files": {k: {"sha256": sha256(v)} for k, v in (
                    ("pgn", pgn), ("index", index), ("tokenizer", root / "tokenizers/chess_uci_vocab.json"),
                    ("decoder_baseline", base))}}
    (saved / "manifest.json").write_text(json.dumps(manifest))
    model = small_model()
    with (saved / "per_move.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_MOVE_FIELDS)
        writer.writeheader()
        for gid, game in enumerate(games):
            writer.writerows(csv_rows(game, gid, model))
    return model


class DecoderInferenceTests(unittest.TestCase):
    def test_causal_cache_and_truncated_windows_match_individual_prefixes(self):
        game = make_game(())
        board, node = game.board(), game
        for index in range(260):
            move = chess.Move.from_uci(("g1f3", "g8f6", "f3g1", "f6g8")[index % 4])
            node = node.add_variation(move)
            board.push(move)
        model = small_model()
        examples = game_examples(game, 0, model.decoder.tokenizer, include_board_grid=False)
        rows = predict_game(examples, model, torch.device("cpu"), batch_size=2)
        for ply in (0, 1, 254, 255, 256, 257, 259):
            e = examples[ply]
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([e.move_ids]), piece_input_ids=torch.tensor([e.piece_ids]),
                               start_turn=torch.tensor([e.start_turn]))["logits"][0, -1]
            predicted, top5, nll = decoder_result(e, logits, model.decoder.tokenizer)
            self.assertEqual(rows[ply]["predicted_uci"], predicted)
            self.assertEqual(rows[ply]["top5_correct"], int(e.target.uci() in top5))
            self.assertAlmostEqual(rows[ply]["nll_legal"], nll, places=5)
        self.assertIsNone(examples[0].labels64)

    def test_promotion_scoring_and_changed_position_rejection(self):
        game = chess.pgn.Game.from_board(chess.Board("k7/4P3/8/8/8/8/8/K7 w - - 0 1"))
        game.headers.update(TWICIssue="1667", WhiteElo="2200", BlackElo="2200")
        game.add_variation(chess.Move.from_uci("e7e8n"))
        model = small_model()
        rows = list(csv_rows(game, 7, model))
        converted = [{"game_id": 7, **baseline_row({k: str(v) for k, v in row.items()})} for row in rows]
        examples = game_examples(game, 7, model.decoder.tokenizer, include_board_grid=False)
        validate_positions(examples, converted)
        self.assertEqual(examples[0].target.uci(), "e7e8n")
        converted[0]["fen_before"] = chess.STARTING_FEN
        with self.assertRaisesRegex(ValueError, "position mismatch"):
            validate_positions(examples, converted)

    def test_baseline_check_rejects_metric_drift(self):
        row = {"ply": 0, "predicted_uci": "e2e4", "top1_correct": 1, "top5_correct": 1, "nll_legal": .5}
        check_baseline_predictions([row], [dict(row)])
        with self.assertRaisesRegex(ValueError, "likelihoods"):
            check_baseline_predictions([row], [{**row, "nll_legal": .6}])


class StatisticsTests(unittest.TestCase):
    def test_position_weighting_pairing_and_switch_accounting(self):
        array = np.array([[2, 0, 1, 6, 2, 2, 4, 2, 2, 0], [8, 8, 8, 8, 4, 6, 10, 4, 0, 4]], dtype=float)
        summaries, deltas = summarize_arrays({"overall": array}, replicates=1000, seed=42)
        self.assertAlmostEqual(summaries[0]["accuracy_top1"], .8)
        self.assertAlmostEqual(summaries[1]["accuracy_top1"], .6)
        self.assertAlmostEqual(summaries[1]["accuracy_top1_equal_game"], .75)
        self.assertAlmostEqual(deltas[0]["delta_accuracy_top1"], -.2)
        self.assertAlmostEqual(deltas[0]["ci_low_accuracy_top1"], -.5)
        self.assertAlmostEqual(deltas[0]["ci_high_accuracy_top1"], 1.)
        self.assertEqual(deltas, summarize_arrays({"overall": array}, replicates=1000, seed=42)[1])


class StorageAndPipelineTests(unittest.TestCase):
    def test_transaction_rolls_back_partial_candidate_game(self):
        with tempfile.TemporaryDirectory() as raw:
            store = Store(Path(raw) / "results.sqlite", {"test": True}, resume=False)
            try:
                rows = []
                for ply in range(2):
                    row = {key: "" for key in BASE_FIELDS}
                    row.update(ply=ply, game_plies=2, twic_issue=1667, white_elo=2000, black_elo=2000,
                               n_legal=20, context_len=ply+1, context_truncated=0, top1_correct=1,
                               top5_correct=1, nll_legal=.5)
                    rows.append(row)
                store.save_baseline(0, rows)
                predictions = [{"ply": p, "predicted_uci": "e2e4", "top1_correct": 1, "top5_correct": 1, "nll_legal": .5} for p in range(2)]
                store.db.execute("CREATE TRIGGER interrupt_candidate BEFORE INSERT ON candidate WHEN NEW.ply=1 BEGIN SELECT RAISE(ABORT, 'interrupted'); END")
                with self.assertRaises(sqlite3.IntegrityError):
                    store.save_candidate(0, predictions)
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM candidate").fetchone()[0], 0)
                self.assertEqual(store.completed(), set())
                store.db.execute("DROP TRIGGER interrupt_candidate")
                store.save_candidate(0, predictions)
                store.validate_complete({0})
            finally:
                store.close()

    def test_lock_rejects_concurrent_run(self):
        with tempfile.TemporaryDirectory() as raw, run_lock(Path(raw)):
            with self.assertRaises(RuntimeError):
                with run_lock(Path(raw)):
                    pass

    def test_end_to_end_reuses_baseline_resumes_and_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = fixture(root)
            output = root / "comparison"
            args = ["--project-root", str(root), "--output-dir", str(output),
                    "--max-games", "2", "--device", "cpu", "--bootstrap", "20"]
            with patch("analysis.decoder_alignment.__main__.load_model", return_value=model), redirect_stdout(io.StringIO()):
                main(args)
            result = json.loads((output / "summary.json").read_text())
            self.assertFalse(result["full_corpus"])
            self.assertEqual(result["paired_deltas"][0]["delta_accuracy_top1"], 0)
            self.assertEqual(result["summaries"][0]["n_positions"], 8)
            self.assertEqual(json.loads((output / "progress.json").read_text())["stage"], "complete")
            with patch("analysis.decoder_alignment.__main__.load_model", side_effect=AssertionError("must not reload")), redirect_stdout(io.StringIO()):
                main(args + ["--resume"])
                main(args + ["--resume", "--stage", "export"])
            with (output / "per_move.csv").open() as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 8)
            self.assertIn("candidate_nll_legal", exported[0])
            candidate = root / "checkpoints/rloo_decoder_v12_lichess_mixed_v5_horizon8.pt"
            candidate.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "identity changed"), redirect_stdout(io.StringIO()):
                main(args + ["--resume"])

    def test_interrupted_evaluation_resumes_only_the_unfinished_game(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = fixture(root)
            output = root / "comparison"
            args = ["--project-root", str(root), "--output-dir", str(output),
                    "--device", "cpu", "--bootstrap", "20"]
            calls = []

            def interrupt(examples, *a):
                calls.append(examples[0].game_id)
                if len(calls) == 3:  # baseline check, first candidate, interrupted second
                    raise KeyboardInterrupt
                return predict_game(examples, *a)

            with patch("analysis.decoder_alignment.__main__.load_model", return_value=model), patch(
                    "analysis.decoder_alignment.__main__.predict_game", side_effect=interrupt), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
            self.assertEqual(raised.exception.code, 130)
            state = json.loads((output / "progress.json").read_text())
            self.assertEqual(state["stage"], "interrupted")
            self.assertEqual(state["completed_games"], 1)
            with patch("analysis.decoder_alignment.__main__.load_model", return_value=model), patch(
                    "analysis.decoder_alignment.__main__.predict_game", wraps=predict_game) as resumed, redirect_stdout(io.StringIO()):
                main(args + ["--resume"])
            self.assertEqual(resumed.call_count, 1)
            self.assertEqual(json.loads((output / "progress.json").read_text())["completed_games"], 2)

    def test_output_cannot_overwrite_the_baseline(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            baseline = root / "baseline"
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(["--project-root", str(root), "--baseline-dir", str(baseline), "--output-dir", str(baseline)])


if __name__ == "__main__":
    unittest.main()
