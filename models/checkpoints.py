"""Checkpoint construction shared by the GUI and tournament UCI entry points."""

from models.chess_decoder_board_player import ChessDecoderBoardPlayer
from models.chess_decoder_player import ChessDecoderPlayer


def load_decoder_player(state_dict, *, tokenizer_path, max_seq_len, n_layers):
    """Load a plain or auxiliary-board decoder strictly from its weight keys."""
    decoder_class = (
        ChessDecoderBoardPlayer
        if any(key.startswith("board_head.") for key in state_dict)
        else ChessDecoderPlayer
    )
    player = decoder_class(
        tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, n_layers=n_layers,
    )
    player.load_state_dict(state_dict, strict=True)
    return player
