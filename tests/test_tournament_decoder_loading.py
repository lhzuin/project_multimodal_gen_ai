"""Keep old decoder inference intact while loading the auxiliary variant strictly."""

from pathlib import Path
import unittest

import torch

from models.chess_decoder_board_player import ChessDecoderBoardPlayer
from models.chess_decoder_player import ChessDecoderPlayer
from models.checkpoints import load_decoder_player


TOKENIZER = str(Path(__file__).resolve().parents[1] / "tokenizers/chess_uci_vocab.json")


class DecoderLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_architectures_load_strictly_and_preserve_policy_logits(self):
        for model_class in (ChessDecoderPlayer, ChessDecoderBoardPlayer):
            with self.subTest(model=model_class.__name__):
                original = model_class(tokenizer_path=TOKENIZER, max_seq_len=16, n_layers=1).eval()
                loaded = load_decoder_player(original.state_dict(), tokenizer_path=TOKENIZER,
                                             max_seq_len=16, n_layers=1).eval()
                self.assertIs(type(loaded), model_class)
                inputs = {"input_ids": torch.tensor([[1, 2]]),
                          "piece_input_ids": torch.tensor([[1, 2]]),
                          "attention_mask": torch.ones(1, 2, dtype=torch.long),
                          "start_turn": torch.tensor([1])}
                with torch.no_grad():
                    expected, actual = original(**inputs), loaded(**inputs)
                torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
                torch.testing.assert_close(actual["piece_logits"], expected["piece_logits"], rtol=0, atol=0)
                self.assertNotIn("board_logits", actual)

    def test_auxiliary_head_is_not_silently_dropped_or_partially_loaded(self):
        original = ChessDecoderBoardPlayer(tokenizer_path=TOKENIZER, max_seq_len=16, n_layers=1)
        weights = dict(original.state_dict())
        del weights["board_head.bias"]
        with self.assertRaises(RuntimeError):
            load_decoder_player(weights, tokenizer_path=TOKENIZER, max_seq_len=16, n_layers=1)


if __name__ == "__main__":
    unittest.main()
