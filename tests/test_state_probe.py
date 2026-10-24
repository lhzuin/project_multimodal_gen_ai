import io
import csv
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import chess
import chess.pgn
import numpy as np
import torch

from analysis.state_probe.core import (
    BloomIndex, assign_split, board_target, build_splits, extract_condition_batch,
    generated_bucket, history_tokens, position_digest, truncate_history,
)
from analysis.state_probe.pipeline import (
    _cross_replay_bootstrap, _drift_bootstrap, _encode_uci_history,
    _greedy_move, _paired_bootstrap, _trajectory_gap_bootstrap,
)
from analysis.state_probe.pipeline import METRIC_FIELDS
from analysis.state_probe.plot_results import main as plot_main
from models.chess_decoder_player import ChessDecoderPlayer
from tests.test_twic_category_builder import workspace_temp_dir
from dataset.offsets import ensure_offsets


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizers/chess_uci_vocab.json"


def game(text="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *", fen=None):
    headers = ['[Event "probe test"]', '[TWICIssue "1662"]']
    if fen:
        headers += ['[SetUp "1"]', f'[FEN "{fen}"]']
    return chess.pgn.read_game(io.StringIO("\n".join(headers) + "\n\n" + text))


class HiddenCaptureTests(unittest.TestCase):
    def test_capture_does_not_change_logits_and_final_is_policy_input(self):
        torch.manual_seed(1)
        model = ChessDecoderPlayer(
            tokenizer_path=str(TOKENIZER), model_dim=32, n_heads=4,
            n_layers=12, dropout=0.0, max_seq_len=32, tie_weights=False,
        ).eval()
        ids = torch.tensor([[model.bos_id, model.decoder.tokenizer.move2id["e2e4"]]])
        pieces = torch.tensor([[model.decoder.tokenizer.p_bos_id,
                                model.decoder.tokenizer.piece2id["P"]]])
        turn = torch.tensor([0])
        ordinary = model(ids, piece_input_ids=pieces, start_turn=turn)
        captured = model(ids, piece_input_ids=pieces, start_turn=turn,
                         capture_layers=(3, 6, 9, 12))
        self.assertTrue(torch.equal(ordinary["logits"], captured["logits"]))
        self.assertEqual(set(captured["hidden_states"]), {3, 6, 9, 12})
        expected = model.lm_head(captured["hidden_states"][12])
        self.assertTrue(torch.equal(expected, captured["logits"]))

    def test_batched_contexts_have_all_conditions(self):
        torch.manual_seed(2)
        model = ChessDecoderPlayer(
            tokenizer_path=str(TOKENIZER), model_dim=32, n_heads=4,
            n_layers=12, dropout=0.0, max_seq_len=256, tie_weights=False,
        ).eval()
        states = list(history_tokens(game(), model.decoder.tokenizer))[:-1]
        histories = [(ids, pieces) for _, _, ids, pieces in states]
        values = extract_condition_batch(model, histories, 0, torch.device("cpu"), batch_size=3)
        self.assertEqual(set(values), {
            "final_w32", "final_w64", "final_w128", "final_w256",
            "layer3_w256", "layer6_w256", "layer9_w256",
        })
        self.assertEqual(values["final_w256"].shape, (len(histories), 32))


class StateTargetTests(unittest.TestCase):
    def test_board_metadata_handles_castling_and_en_passant(self):
        board = chess.Board()
        target = board_target(board)
        self.assertEqual(target["board"].shape, (64,))
        self.assertEqual(target["castling"].tolist(), [1, 1, 1, 1])
        board.push_uci("e2e4")
        # python-chess records e3 even though no legal ep capture exists; labels
        # preserve board.ep_square while novelty canonicalization uses legal ep.
        self.assertEqual(board_target(board)["ep"], chess.E3 + 1)

    def test_promotion_target_has_promoted_piece(self):
        board = chess.Board("k7/4P3/8/8/8/8/8/K7 w - - 0 1")
        board.push_uci("e7e8q")
        labels = board_target(board)["board"]
        self.assertEqual(int((labels == 5).sum()), 1)

    def test_context_turn_alignment_at_sliding_boundary(self):
        ids = list(range(300))
        pieces = list(range(300))
        kept_ids, kept_pieces, start_turn = truncate_history(ids, pieces, 0, 256)
        self.assertEqual(kept_ids, ids[-256:])
        self.assertEqual(kept_pieces, pieces[-256:])
        self.assertEqual(start_turn, (44 - 1) % 2)


class TrajectoryAndStatisticsTests(unittest.TestCase):
    def test_replay_is_legal_and_reaches_expected_fen(self):
        model = ChessDecoderPlayer(
            tokenizer_path=str(TOKENIZER), model_dim=16, n_heads=4,
            n_layers=1, dropout=0.0, max_seq_len=32,
        )
        moves = [chess.Move.from_uci(move) for move in ("e2e4", "e7e5", "g1f3")]
        ids, pieces, initial_turn, board = _encode_uci_history(chess.Board(), moves,
                                                              model.decoder.tokenizer)
        expected = chess.Board()
        for move in moves:
            expected.push(move)
        self.assertEqual(board.fen(), expected.fen())
        self.assertEqual(initial_turn, 0)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(pieces), 4)

    def test_greedy_selfplay_choice_is_deterministic_and_legal(self):
        torch.manual_seed(3)
        model = ChessDecoderPlayer(
            tokenizer_path=str(TOKENIZER), model_dim=16, n_heads=4,
            n_layers=1, dropout=0.0, max_seq_len=32,
        ).eval()
        board = chess.Board()
        ids = [model.bos_id]
        pieces = [model.decoder.tokenizer.p_bos_id]
        first = _greedy_move(model, board, ids, pieces, 0, torch.device("cpu"))
        second = _greedy_move(model, board, ids, pieces, 0, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertIn(first, board.legal_moves)

    def test_bloom_has_no_false_negatives_for_insertions(self):
        bloom = BloomIndex(100, 1e-4)
        boards = [chess.Board()]
        for move in ("e2e4", "e7e5", "g1f3"):
            board = boards[-1].copy(stack=False)
            board.push_uci(move)
            boards.append(board)
        for board in boards:
            bloom.add_digest(position_digest(board))
        self.assertTrue(all(bloom.contains_digest(position_digest(board)) for board in boards))

    def test_generated_buckets_cover_rollout(self):
        self.assertEqual(generated_bucket(0), "0_10")
        self.assertEqual(generated_bucket(119), "80_120")
        with self.assertRaises(ValueError):
            generated_bucket(120)

    def test_bootstrap_uses_complete_game_units(self):
        rows = []
        for game_id in (1, 2, 3):
            for ply in (0, 1):
                for model, value in (("decoder_baseline", .4), ("decoder_balanced", .5)):
                    rows.append({
                        "trajectory_source": "human", "game_id": game_id,
                        "ply": ply, "condition": "final_w256", "encoded_by": model,
                        "bucket": "0_20", "board_nll": value,
                        "occupied_nll": value + .1, "board_hamming": value / 2,
                    })
        result = _paired_bootstrap(rows, {"bootstrap": {"seed": 42, "replicates": 50}})
        self.assertTrue(result)
        self.assertTrue(all(row["n_games"] == 3 for row in result))
        self.assertTrue(all(np.isclose(row["balanced_minus_baseline"], .1) or
                            np.isclose(row["balanced_minus_baseline"], .05)
                            for row in result))

    def test_on_policy_bootstraps_keep_complete_openings(self):
        rows = []
        for opening in (1, 2, 3):
            for model_index, model in enumerate(("decoder_baseline", "decoder_balanced")):
                for generated_ply, band, absolute in ((0, "0_10", "0_20"),
                                                       (90, "80_120", "100_plus")):
                    base = .2 + .05 * model_index + .001 * generated_ply
                    rows.append({
                        "unit_id": opening, "game_id": opening,
                        "trajectory_source": "decoder_baseline", "encoded_by": model,
                        "condition": "final_w256", "generated_bucket": band,
                        "bucket": absolute, "board_nll": base,
                        "occupied_nll": base + .1, "board_hamming": base / 2,
                    })
                    rows.append({
                        "unit_id": opening, "game_id": opening,
                        "trajectory_source": "human", "encoded_by": model,
                        "condition": "final_w256", "generated_bucket": "",
                        "bucket": absolute, "board_nll": base - .03,
                        "occupied_nll": base + .07, "board_hamming": base / 2 - .03,
                    })
        config = {"bootstrap": {"seed": 42, "replicates": 50}}
        cross = _cross_replay_bootstrap(rows, config)
        drift = _drift_bootstrap(rows, config)
        gaps = _trajectory_gap_bootstrap(rows, config)
        self.assertTrue(cross and drift and gaps)
        self.assertTrue(all(row["n_openings"] == 3 for row in cross))
        self.assertTrue(any(row["encoded_by"] == "balanced_minus_baseline" for row in drift))
        self.assertTrue(all(row["ci_low"] <= row["ci_high"] for row in (*cross, *drift, *gaps)))

    def test_split_assignment_is_stable(self):
        raw = b"[Event \"same game\"]\n\n1. e4 *\n"
        self.assertEqual(assign_split(raw), assign_split(raw))

    def test_split_manifest_is_issue_stratified_and_disjoint(self):
        with workspace_temp_dir() as raw_directory:
            directory = Path(raw_directory)
            pgn = directory / "games.pgn"
            records = []
            for issue in range(1662, 1668):
                for number in range(60):
                    records.append(
                        f'[Event "issue {issue} game {number}"]\n'
                        f'[TWICIssue "{issue}"]\n\n1. e4 e5 *\n\n'
                    )
            pgn.write_text("".join(records), encoding="utf-8")
            index = directory / "index.json"
            ensure_offsets(str(pgn), str(index))
            rows = build_splits({
                "paths": {"pgn": str(pgn), "index": str(index)},
                "sample_games_per_issue": {"train": 1, "val": 1, "test": 1},
            })
            self.assertEqual(len(rows), 360)
            self.assertEqual(len({row["game_id"] for row in rows}), 360)
            selected = [row for row in rows if row["selected"]]
            counts = {}
            for row in selected:
                counts[(row["twic_issue"], row["split"])] = counts.get((row["twic_issue"], row["split"]), 0) + 1
            self.assertEqual(set(counts.values()), {1})
            self.assertEqual(len(counts), 18)


class PlotArtifactTests(unittest.TestCase):
    def test_plot_fixture_generates_six_svgs_and_report(self):
        with workspace_temp_dir() as raw_directory:
            directory = Path(raw_directory)
            summary_fields = (
                "trajectory_source", "encoded_by", "condition", "bucket",
                "n_units", "n_positions", "novel_fraction", *METRIC_FIELDS,
            )
            rows = []
            models = ("decoder_baseline", "decoder_balanced")
            human_buckets = ("0_20", "20_40", "40_60", "60_80", "80_100", "100_plus")
            conditions = (
                "final_w32", "final_w64", "final_w128", "final_w256",
                "final_w256_sample", "layer3_w256", "layer6_w256", "layer9_w256",
            )
            for model_index, model in enumerate(models):
                for condition_index, condition in enumerate(conditions):
                    for bucket_index, bucket in enumerate(human_buckets):
                        row = {
                            "trajectory_source": "human", "encoded_by": model,
                            "condition": condition, "bucket": bucket,
                            "n_units": 4, "n_positions": 20, "novel_fraction": .1,
                        }
                        row.update({metric: .1 + model_index * .01 + condition_index * .001 + bucket_index * .002
                                    for metric in METRIC_FIELDS})
                        rows.append(row)
            generated_buckets = ("0_10", "10_20", "20_40", "40_80", "80_120")
            for source_index, source in enumerate(models):
                for model_index, model in enumerate(models):
                    for bucket_index, bucket in enumerate(generated_buckets):
                        row = {
                            "trajectory_source": source, "encoded_by": model,
                            "condition": "final_w256", "bucket": bucket,
                            "n_units": 4, "n_positions": 20,
                            "novel_fraction": .2 + bucket_index * .05,
                        }
                        row.update({metric: .2 + source_index * .01 + model_index * .01 + bucket_index * .01
                                    for metric in METRIC_FIELDS})
                        rows.append(row)
            with (directory / "bucket_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(rows)
            for filename, fields in (
                ("paired_deltas.csv", ("condition",)),
                ("cross_replay_interactions.csv", ("trajectory_source",)),
            ):
                with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writeheader()
            association_fields = ("encoded_by", "bucket", "probe_error_quartile",
                                  "n_positions", "policy_top1", "policy_nll")
            with (directory / "policy_error_association.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=association_fields)
                writer.writeheader()
                for model in models:
                    for quartile in range(1, 5):
                        writer.writerow({
                            "encoded_by": model, "bucket": "0_20",
                            "probe_error_quartile": quartile, "n_positions": 10,
                            "policy_top1": .5, "policy_nll": 1 + quartile * .1,
                        })
            plot_main(["--output-dir", str(directory)])
            for filename in (
                "layerwise_decodability.svg", "context_length.svg",
                "human_vs_selfplay_drift.svg", "cross_replay.svg",
                "common_reference_novelty.svg", "policy_error_association.svg",
            ):
                self.assertEqual(ET.parse(directory / "figures" / filename).getroot().tag,
                                 "{http://www.w3.org/2000/svg}svg")
            self.assertIn("state drift", (directory / "README.md").read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
