"""Differentiable adapters for this repository's encoder and decoder policies.

All normalization happens over actual legal actions. Sampling, old-policy
probabilities, reference KL and gradient replay therefore share one distribution.
No no_grad-only sample_moves method is reused for gradient computation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import chess
import torch
from torch import nn
from torch.nn import functional as F


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(requested)


def _checked_moves(observation: Mapping[str, Any]) -> list[chess.Move]:
    moves = [chess.Move.from_uci(uci) for uci in observation["legal_moves_uci"]]
    if not moves:
        raise ValueError("a policy observation must have at least one legal action")
    return moves


def encoder_move_distribution(
    policy_logits: torch.Tensor,
    promo_logits: torch.Tensor,
    moves: Sequence[chess.Move],
    temperature: float,
) -> torch.Tensor:
    """P(move) = P(from,to) * P(promotion | from,to).

    Four promotion variants share one pair's probability mass. A flat softmax
    over duplicated pair logits would incorrectly favor promotion squares.
    Unlike the tournament wrapper, promotions are sampled, so both heads learn.
    """
    pair_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, move in enumerate(moves):
        pair_to_indices[(move.from_square, move.to_square)].append(index)
    pairs = list(pair_to_indices)
    pair_values = torch.stack([policy_logits[f, t] for f, t in pairs]).float()
    pair_log_probs = F.log_softmax(pair_values / temperature, dim=0)
    promotion_ids = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    values: list[torch.Tensor | None] = [None] * len(moves)
    for pair_index, pair in enumerate(pairs):
        indices = pair_to_indices[pair]
        if len(indices) == 1:
            values[indices[0]] = pair_log_probs[pair_index]
            continue
        promo_values = torch.stack(
            [promo_logits[promotion_ids[moves[index].promotion]] for index in indices]
        ).float()
        promo_log_probs = F.log_softmax(promo_values / temperature, dim=0)
        for offset, index in enumerate(indices):
            values[index] = pair_log_probs[pair_index] + promo_log_probs[offset]
    return torch.stack([value for value in values if value is not None])


class EncoderPolicyAdapter:
    def __init__(self, model: nn.Module, *, version: int = 2) -> None:
        if version not in (1, 2):
            raise ValueError("encoder version must be 1 or 2")
        self.model = model
        self.version = version

    def log_distributions(
        self, observations: Sequence[Mapping[str, Any]], temperature: float
    ) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        boards = [chess.Board(observation["fen"]) for observation in observations]
        labels = []
        for board in boards:
            row = [0] * 64
            for square, piece in board.piece_map().items():
                # Dataset scan order is a8..h1; policy action coordinates remain
                # python-chess a1..h8, matching the existing supervised model.
                row[square ^ 56] = piece.piece_type + (0 if piece.color else 6)
            labels.append(row)
        labels64 = torch.tensor(labels, dtype=torch.long, device=device)
        kwargs = {
            "turn": torch.tensor([int(b.turn) for b in boards], device=device),
            "castling": torch.tensor(
                [
                    [
                        int(b.has_kingside_castling_rights(chess.WHITE)),
                        int(b.has_queenside_castling_rights(chess.WHITE)),
                        int(b.has_kingside_castling_rights(chess.BLACK)),
                        int(b.has_queenside_castling_rights(chess.BLACK)),
                    ]
                    for b in boards
                ],
                device=device,
            ),
            "ep_square": torch.tensor(
                [-1 if b.ep_square is None else b.ep_square for b in boards],
                device=device,
            ),
        }
        if self.version == 2:
            kwargs["labels64"] = labels64
        else:
            kwargs["piece_probs"] = F.one_hot(labels64, num_classes=13).float()
        output = self.model(**kwargs)
        return [
            encoder_move_distribution(
                output["policy_logits"][index],
                output["promo_logits"][index],
                _checked_moves(observation),
                temperature,
            )
            for index, observation in enumerate(observations)
        ]


class DecoderPolicyAdapter:
    def __init__(self, model: nn.Module, *, max_seq_len: int = 256) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.model = model
        self.tokenizer = model.decoder.tokenizer
        self.max_seq_len = max_seq_len

    def _encode_history(
        self, observation: Mapping[str, Any]
    ) -> tuple[list[int], list[int], int]:
        tok = self.tokenizer
        base_fen = observation["start_fen"]
        if base_fen != chess.STARTING_FEN:
            raise ValueError(
                "decoder requires history from the standard starting position"
            )
        board = chess.Board(base_fen)
        move_ids = [tok.bos_id]
        piece_ids = [tok.p_bos_id]
        for uci in observation["moves_uci"]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"invalid decoder history move {uci}")
            piece = board.piece_at(move.from_square)
            if uci not in tok.move2id or piece is None:
                raise ValueError(f"decoder history is not representable: {uci}")
            move_ids.append(tok.move2id[uci])
            piece_ids.append(tok.piece2id[piece.symbol().upper()])
            board.push(move)
        if board.fen() != chess.Board(observation["fen"]).fen():
            raise ValueError("decoder history and observation FEN disagree")
        # Keep the supervised/tournament token-turn convention. No trailing EOS
        # is added: it would be an end-of-game marker in an ongoing rollout.
        start = max(0, len(move_ids) - self.max_seq_len)
        start_turn = max(0, start - 1) % 2
        return move_ids[start:], piece_ids[start:], start_turn

    def log_distributions(
        self, observations: Sequence[Mapping[str, Any]], temperature: float
    ) -> list[torch.Tensor]:
        device = next(self.model.parameters()).device
        histories = [self._encode_history(observation) for observation in observations]
        lengths = [len(history[0]) for history in histories]
        width = max(lengths)
        tok = self.tokenizer
        input_ids = torch.full(
            (len(histories), width), tok.pad_id, device=device, dtype=torch.long
        )
        piece_ids = torch.full_like(input_ids, tok.p_pad_id)
        attention = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, (moves, pieces, _) in enumerate(histories):
            input_ids[index, : len(moves)] = torch.tensor(moves, device=device)
            piece_ids[index, : len(pieces)] = torch.tensor(pieces, device=device)
            attention[index, : len(moves)] = True
        output = self.model(
            input_ids=input_ids,
            piece_input_ids=piece_ids,
            attention_mask=attention,
            start_turn=torch.tensor([h[2] for h in histories], device=device),
        )
        result = []
        for index, observation in enumerate(observations):
            try:
                legal_ids = [tok.move2id[uci] for uci in observation["legal_moves_uci"]]
            except KeyError as exc:
                raise ValueError(
                    f"legal move missing from decoder vocabulary: {exc}"
                ) from exc
            # Right-padded batches must gather the last real token per row.
            logits = output["logits"][index, lengths[index] - 1, legal_ids].float()
            result.append(F.log_softmax(logits / temperature, dim=0))
        return result


def _load_weights(path: str | Path) -> Mapping[str, torch.Tensor]:
    # Original latest checkpoints also contain NumPy RNG state. The caller
    # supplies its own trusted training checkpoint, as in existing trainers.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            checkpoint = checkpoint[key]
            break
    # Strip prefixes only when all keys share them, without corrupting real names.
    for prefix in ("module.", "model."):
        if checkpoint and all(key.startswith(prefix) for key in checkpoint):
            checkpoint = {
                key[len(prefix) :]: value for key, value in checkpoint.items()
            }
    return checkpoint


def load_policy(
    checkpoint_path: str | Path,
    model_type: str,
    model_kwargs: Mapping[str, Any] | None = None,
) -> EncoderPolicyAdapter | DecoderPolicyAdapter:
    """Strictly load original weights; constructor args are explicit, Hydra-free.

    Attention head count is not recoverable from tensor shapes. Supply the same
    architecture arguments as supervised training; defaults are class defaults.
    """
    kwargs = dict(model_kwargs or {})
    if model_type == "encoder":
        from models.chess_encoder_player import ChessEncoderPlayer, ChessEncoderPlayerV2

        version = int(kwargs.pop("version", 2))
        if version not in (1, 2):
            raise ValueError("encoder version must be 1 or 2")
        kwargs.setdefault("img_size", 256)
        kwargs.setdefault("freeze_vit", True)
        model = (ChessEncoderPlayerV2 if version == 2 else ChessEncoderPlayer)(**kwargs)
        model.load_state_dict(_load_weights(checkpoint_path), strict=True)
        # State input never uses the visual path. Keep its original weights for
        # compatible exports but do not allocate optimizer state for it.
        model.vit.requires_grad_(False)
        return EncoderPolicyAdapter(model, version=version)
    if model_type == "decoder":
        from models.chess_decoder_player import ChessDecoderPlayer

        tokenizer_path = kwargs.get("tokenizer_path")
        if not tokenizer_path or not Path(tokenizer_path).is_file():
            raise ValueError(
                "decoder model_kwargs must specify an existing tokenizer_path"
            )
        model = ChessDecoderPlayer(**kwargs)
        model.load_state_dict(_load_weights(checkpoint_path), strict=True)
        model.piece_head.requires_grad_(
            False
        )  # auxiliary supervised head has no RL target
        return DecoderPolicyAdapter(model, max_seq_len=kwargs.get("max_seq_len", 256))
    raise ValueError("model_type must be encoder or decoder")
