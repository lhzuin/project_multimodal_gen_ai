"""Board targets for the existing decoder windows, with full-prefix PGN replay."""

import torch

from dataset.pgn_move_dataset import PGNMoveDataset
from dataset.renderer import board_to_grid_ids


class PGNMoveBoardDataset(PGNMoveDataset):
    """Extend both balanced and legacy items with unshifted ``board_labels``.

    At absolute token k, replay k moves (zero at BOS). Crops retain the true
    position from the original game, including custom FEN starting positions.
    EOS has no reconstruction target. No board state is passed to the decoder.
    """

    def _additional_item_fields(self, game, first_token_index: int, token_count: int):
        moves = list(game.mainline_moves())
        board = game.board()
        labels = torch.full((token_count, 64), -100, dtype=torch.long)
        stop = first_token_index + token_count
        # One incremental replay covers the prefix and selected window. Captures,
        # en passant, castling and promotion use python-chess's normal push path.
        for token_index in range(min(stop, len(moves) + 1)):
            if token_index > 0:
                board.push(moves[token_index - 1])
            if token_index >= first_token_index:
                labels[token_index - first_token_index] = torch.from_numpy(
                    board_to_grid_ids(board).reshape(64)
                )
        return {"board_labels": labels}
