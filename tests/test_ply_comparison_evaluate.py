import io
import math
from pathlib import Path
import unittest

import chess
import chess.pgn
import torch

from analysis.ply_comparison.evaluate import (
    ROOT, cached_decoder_logits, encoder_result, game_examples, select_issue_sample,
)
from dataset.offsets import ensure_offsets, load_pgn_offsets
from models.chess_decoder_player import ChessDecoderPlayer
from models.chess_tokenizer import ChessTokenizer
from tests.test_twic_category_builder import workspace_temp_dir


TOKENIZER_PATH = ROOT / "tokenizers/chess_uci_vocab.json"


def game_from_text(moves: str, *, initial_fen: str | None = None):
    headers = ['[Event "Test"]', '[WhiteElo "2200"]', '[BlackElo "2200"]',
               '[TWICIssue "1667"]']
    if initial_fen:
        headers.extend(['[SetUp "1"]', f'[FEN "{initial_fen}"]'])
    return chess.pgn.read_game(io.StringIO("\n".join(headers) + "\n\n" + moves + " *"))


class EvaluationAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ChessTokenizer(str(TOKENIZER_PATH))

    def test_decoder_context_boundary_and_turn_alignment(self):
        game = chess.pgn.Game()
        game.headers["TWICIssue"] = "1667"
        game.headers["WhiteElo"] = "2200"
        game.headers["BlackElo"] = "2200"
        board = game.board()
        node = game
        cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
        for i in range(260):
            move = chess.Move.from_uci(cycle[i % 4])
            self.assertIn(move, board.legal_moves)
            node = node.add_variation(move)
            board.push(move)
        examples = game_examples(game, 3, self.tokenizer)
        self.assertEqual(len(examples[255].move_ids), 256)
        self.assertFalse(examples[255].context_truncated)
        self.assertEqual(len(examples[256].move_ids), 256)
        self.assertTrue(examples[256].context_truncated)
        self.assertEqual(examples[256].start_turn, 0)
        self.assertEqual(examples[256].move_ids[0], self.tokenizer.move2id["g1f3"])

    def test_initial_fen_and_promotion_probability(self):
        fen = "k7/4P3/8/8/8/8/8/K7 w - - 0 1"
        game = game_from_text("1. e8=Q", initial_fen=fen)
        examples = game_examples(game, 0, self.tokenizer)
        example = examples[0]
        self.assertEqual(example.target.uci(), "e7e8q")
        self.assertEqual(example.fen, fen)
        policy = torch.zeros((64, 64))
        policy[chess.E7, chess.E8] = 9
        promo = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0])
        greedy, top5, nll, pair_correct = encoder_result(example, policy, promo)
        self.assertTrue(pair_correct)
        self.assertIn(greedy, [move.uci() for move in example.legal_moves])
        self.assertEqual(top5[0], greedy)
        self.assertTrue(math.isfinite(nll))
        self.assertGreaterEqual(nll, 0)

    def test_full_causal_pass_matches_individual_prefixes(self):
        game = game_from_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")
        examples = game_examples(game, 0, self.tokenizer)
        model = ChessDecoderPlayer(
            tokenizer_path=str(TOKENIZER_PATH), model_dim=32,
            n_heads=4, n_layers=1, dropout=0.0, max_seq_len=256,
        ).eval()
        models = {"decoder_baseline": model, "decoder_balanced": model}
        cached = cached_decoder_logits(examples, models, torch.device("cpu"))
        with torch.inference_mode():
            for example in examples:
                output = model(
                    input_ids=torch.tensor([example.move_ids]),
                    piece_input_ids=torch.tensor([example.piece_ids]),
                    start_turn=torch.tensor([example.start_turn]),
                )["logits"][0, -1]
                self.assertTrue(torch.allclose(
                    cached["decoder_baseline"][example.ply], output,
                    atol=1e-5, rtol=1e-5,
                ))

    def test_hash_ranked_sample_covers_each_issue(self):
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            pgn = root / "games.pgn"
            records = []
            for issue in range(1662, 1668):
                for game_number in range(3):
                    records.append(
                        f'[Event "Sample {game_number}"]\n'
                        f'[TWICIssue "{issue}"]\n'
                        '[WhiteElo "2200"]\n[BlackElo "2200"]\n\n'
                        '1. e4 e5 *\n\n'
                    )
            pgn.write_text("".join(records), encoding="utf-8")
            index_path = root / "index.json"
            ensure_offsets(str(pgn), str(index_path))
            index = load_pgn_offsets(str(index_path))
            first = select_issue_sample(pgn, index, 1)
            self.assertEqual(first, select_issue_sample(pgn, index, 1))
            self.assertEqual(len(first), 6)
            for issue in range(1662, 1668):
                self.assertEqual(len(first.intersection(set(range((issue - 1662) * 3,
                                                            (issue - 1662) * 3 + 3)))), 1)


if __name__ == "__main__":
    unittest.main()
