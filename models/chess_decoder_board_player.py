"""Decoder variant with a training-only piece-placement reconstruction head."""

from torch import nn

from models.chess_decoder_player import ChessDecoderPlayer


class ChessDecoderBoardPlayer(ChessDecoderPlayer):
    """Keep the base policy interface and optionally predict ``[B,T,64,13]``.

    Each square has 13 classes: empty, six white pieces, six black pieces.
    Squares follow the existing renderer's order: a8..h8, ..., a1..h1.
    Targets describe the position *after* the input move (BOS: initial position),
    which is the position from which the next move is predicted. Targets are
    never inputs. In eval mode the head is skipped unless explicitly requested
    for validation. Inherited sampling and legal-mask methods work unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_head = nn.Linear(self.lm_head.in_features, 64 * 13)

    def forward(
        self, input_ids, attention_mask=None, start_turn=None,
        piece_input_ids=None, capture_layers=None, *, return_board_logits: bool | None = None,
    ):
        features = self.forward_features(
            input_ids=input_ids, attention_mask=attention_mask,
            start_turn=start_turn, piece_input_ids=piece_input_ids,
            capture_layers=capture_layers,
        )
        h, captured = (features, None) if capture_layers is None else features
        out = {"logits": self.lm_head(h), "piece_logits": self.piece_head(h)}
        if captured is not None:
            out["hidden_states"] = captured
        compute_board = self.training if return_board_logits is None else return_board_logits
        if compute_board:
            out["board_logits"] = self.board_head(h).reshape(*h.shape[:2], 64, 13)
        return out

    def get_param_groups(self):
        """Add the auxiliary parameters without duplicating tied policy weights."""
        groups = dict(super().get_param_groups())
        groups["board_head"] = list(self.board_head.parameters())
        return groups

    def inference_state_dict(self):
        """Return weights strictly loadable by the original ChessDecoderPlayer."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("board_head.")}
