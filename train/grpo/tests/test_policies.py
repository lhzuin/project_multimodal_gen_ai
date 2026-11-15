from pathlib import Path
import tempfile
import unittest

import chess
import torch
from torch import nn

from train.grpo.contracts import PositionPrompt
from train.grpo.policies import (
    DecoderPolicyAdapter,
    EncoderPolicyAdapter,
    encoder_move_distribution,
)


def observation(moves=()):
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return {
        "fen": board.fen(),
        "start_fen": chess.STARTING_FEN,
        "moves_uci": list(moves),
        "legal_moves_uci": [m.uci() for m in board.legal_moves],
    }


class CapturingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Parameter(torch.zeros(64, 64))
        self.promo = nn.Parameter(torch.zeros(5))
        self.inputs = None

    def forward(self, **kwargs):
        self.inputs = kwargs
        batch = kwargs["turn"].numel()
        return {
            "policy_logits": self.policy.expand(batch, -1, -1),
            "promo_logits": self.promo.expand(batch, -1),
        }


class PolicyTests(unittest.TestCase):
    def test_encoder_orientation_and_legal_normalization(self):
        model = CapturingEncoder()
        logs = EncoderPolicyAdapter(model).log_distributions([observation()], 0.8)[0]
        self.assertAlmostEqual(logs.exp().sum().item(), 1, places=6)
        self.assertEqual(model.inputs["labels64"][0, 0].item(), 10)  # black rook a8
        self.assertEqual(model.inputs["labels64"][0, 56].item(), 4)  # white rook a1
        self.assertEqual(model.inputs["turn"][0].item(), 1)

    def test_promotion_keeps_pair_mass_and_has_gradient(self):
        moves = [
            chess.Move.from_uci(u)
            for u in (
                "a7a8q",
                "a7a8r",
                "a7a8b",
                "a7a8n",
                "e1d1",
            )
        ]
        pairs = torch.zeros(64, 64, requires_grad=True)
        promos = torch.zeros(5, requires_grad=True)
        logs = encoder_move_distribution(pairs, promos, moves, 1)
        torch.testing.assert_close(logs.exp(), torch.tensor([0.125] * 4 + [0.5]))
        (-logs[0]).backward()
        self.assertNotEqual(promos.grad[4].item(), 0)
        self.assertNotEqual(pairs.grad[chess.A7, chess.A8].item(), 0)

    def test_decoder_rejects_fen_only_prompts(self):
        prompt = PositionPrompt("fen", start_fen=chess.Board().mirror().fen())
        with self.assertRaises(ValueError):
            prompt.validate("decoder")

    def test_decoder_padding_crop_and_gradient(self):
        from models.chess_decoder_player import ChessDecoderPlayer

        with tempfile.TemporaryDirectory() as directory:
            model = ChessDecoderPlayer(
                tokenizer_path=str(Path(directory) / "tok.json"),
                model_dim=16,
                n_heads=2,
                n_layers=1,
                dropout=0,
                max_seq_len=4,
            ).eval()
            adapter = DecoderPolicyAdapter(model, max_seq_len=4)
            observations = [observation(), observation(("e2e4", "e7e5", "g1f3"))]
            with torch.no_grad():
                batched = adapter.log_distributions(observations, 1)
                separate = [adapter.log_distributions([o], 1)[0] for o in observations]
            for actual, expected in zip(batched, separate):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            longer = observation(("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"))
            move_ids, _, turn = adapter._encode_history(longer)
            self.assertEqual(len(move_ids), 4)
            self.assertEqual(turn, 1)
            loss = -adapter.log_distributions([longer], 1)[0][0]
            loss.backward()
            self.assertIsNotNone(model.decoder.tok_emb.weight.grad)
