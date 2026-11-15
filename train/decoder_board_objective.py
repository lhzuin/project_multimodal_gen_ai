"""Fixed-scale composition of the existing decoder loss and board reconstruction.

See docs/decoder_board_auxiliary.md for the derivation and normalization tradeoffs.
"""

import math

from torch import nn
from torch.nn import functional as F


class DecoderBoardObjective(nn.Module):
    """Mix two mean losses, optionally normalized by uniform-prediction CE.

    ``current = (CE_move + lambda * CE_piece) / (log(V) + lambda * log(P))``
    ``board = CE_board / log(13)``
    ``total = (1 - board_loss_weight) * current + board_loss_weight * board``

    ``normalization='none'`` disables the fixed denominators for ablations.
    All-ignored targets return differentiable zero rather than NaN. Raw losses
    are returned alongside scaled losses so training remains interpretable.
    """

    def __init__(
        self, board_loss_weight: float = 0.3, piece_loss_weight: float = 0.3,
        normalization: str = "uniform",
    ):
        super().__init__()
        if not math.isfinite(board_loss_weight) or not 0 <= board_loss_weight <= 1:
            raise ValueError("board_loss_weight must be finite and in [0, 1]")
        if not math.isfinite(piece_loss_weight) or piece_loss_weight < 0:
            raise ValueError("piece_loss_weight must be finite and nonnegative")
        if normalization not in {"uniform", "none"}:
            raise ValueError("normalization must be 'uniform' or 'none'")
        self.board_loss_weight = board_loss_weight
        self.piece_loss_weight = piece_loss_weight
        self.normalization = normalization

    @staticmethod
    def _mean_ce(logits, labels):
        # Sum/count explicitly handles an all-ignored batch. Float32 also keeps
        # this reduction stable under CUDA mixed precision.
        values = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="none",
        )
        return values.sum() / (labels != -100).sum().clamp_min(1)

    def forward(self, out, labels, piece_labels, board_labels):
        move_ce = self._mean_ce(out["logits"], labels)
        piece_ce = self._mean_ce(out["piece_logits"], piece_labels)
        current_raw = move_ce + self.piece_loss_weight * piece_ce
        board_ce = current_raw.new_zeros(())
        if self.board_loss_weight > 0:
            if board_labels.shape != (*labels.shape, 64):
                raise ValueError("board_labels must have shape [B, T, 64]")
            if out["board_logits"].shape != (*labels.shape, 64, 13):
                raise ValueError("board_logits must have shape [B, T, 64, 13]")
            # Defense in depth: callers cannot accidentally supervise tokens
            # excluded by the balanced move mask, even without our collator.
            board_labels = board_labels.masked_fill((labels == -100).unsqueeze(-1), -100)
            board_ce = self._mean_ce(out["board_logits"], board_labels)
        current_scale = board_scale = 1.0
        if self.normalization == "uniform":
            current_scale = math.log(out["logits"].size(-1)) + self.piece_loss_weight * math.log(out["piece_logits"].size(-1))
            board_scale = math.log(13)
        current = current_raw / current_scale
        board = board_ce / board_scale
        weighted_current = (1 - self.board_loss_weight) * current
        weighted_board = self.board_loss_weight * board
        return {
            "loss": weighted_current + weighted_board,
            "move_ce": move_ce, "piece_ce": piece_ce, "current_raw": current_raw,
            "board_ce": board_ce, "current_normalized": current,
            "board_normalized": board, "weighted_current": weighted_current,
            "weighted_board": weighted_board,
        }
