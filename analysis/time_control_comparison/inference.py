"""Batched perfect-board encoder inference and complete legal-move scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import chess
import numpy as np
import torch

from dataset.renderer import board_to_grid_ids
from models.chess_encoder_player import ChessEncoderPlayerV2
from .common import CATEGORIES, bucket

PROMOTION_CLASS = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def log_softmax(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite legal-move logits")
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


@dataclass(frozen=True)
class MoveScores:
    predicted: str
    target_logp: float
    predicted_logp: float
    top5_correct: bool
    pair_correct: bool
    entropy: float


def score_moves(legal: list[chess.Move], target: chess.Move, policy, promotion) -> MoveScores:
    """Preserve deployed pair-first argmax, including class-zero promotion fallback.

    Pair ties use ascending flattened (from*64+to) index, as torch.argmax does.
    Legal promotion classes share their pair's probability mass; class zero is
    excluded only from the conditional distribution used for complete-move NLL.
    NLL is untempered/untruncated; top-k sampling is irrelevant to greedy play.
    """
    if not legal or target not in legal:
        raise ValueError("target must belong to a nonempty legal-move set")
    groups: dict[int, list[chess.Move]] = {}
    for move in legal:
        groups.setdefault(move.from_square * 64 + move.to_square, []).append(move)
    keys = sorted(groups)
    pair_logp = log_softmax([float(policy[key // 64, key % 64]) for key in keys])
    move_logp = {}
    for key, base in zip(keys, pair_logp):
        candidates = groups[key]
        if len(candidates) == 1:
            move_logp[candidates[0].uci()] = float(base)
        else:
            conditional = log_softmax([float(promotion[PROMOTION_CLASS[move.promotion]]) for move in candidates])
            for move, value in zip(candidates, conditional):
                move_logp[move.uci()] = float(base + value)
    best_pair = keys[int(np.argmax(pair_logp))]
    candidates = groups[best_pair]
    chosen = candidates[0]
    if len(candidates) > 1:
        # Include class zero for the actual deployment decision, matching sample_moves.
        values = np.asarray(promotion, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite promotion logits")
        preferred = int(np.argmax(values))
        chosen = next((move for move in candidates if PROMOTION_CLASS.get(move.promotion) == preferred), chosen)
    target_uci, predicted = target.uci(), chosen.uci()
    ranked = sorted(move_logp, key=lambda move: (-move_logp[move], move))
    entropy = -sum(math.exp(value) * value for value in move_logp.values())
    return MoveScores(predicted, move_logp[target_uci], move_logp[predicted], target_uci in ranked[:5],
                      best_pair == target.from_square * 64 + target.to_square, entropy)


def load_models(checkpoints: dict[str, Path], device: torch.device):
    models = {}
    for category in CATEGORIES:
        path = checkpoints[category]
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(weights, dict):
            for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
                if isinstance(weights.get(key), dict):
                    weights = weights[key]
                    break
        model = ChessEncoderPlayerV2(img_size=256, piece_embed_dim=32, ep_embed_dim=16,
                                     encoder_dim=256, n_heads=4, n_layers=8,
                                     encoder_dropout=0.1, vit_path=None, freeze_vit=True)
        model.load_state_dict(weights, strict=True)
        model.requires_grad_(False)
        models[category] = model.eval().to(device)
        print(f"Loaded {category}: {path}", flush=True)
    return models


def game_positions(game):
    """Replay recorded moves; inputs contain only the board before each target."""
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        legal = list(board.legal_moves)
        if move == chess.Move.null() or move not in legal:
            raise ValueError(f"illegal target at ply {ply}")
        yield {
            "ply": ply, "bucket": bucket(ply), "fen_before": board.fen(),
            "target_uci": move.uci(), "n_legal": len(legal),
            "turn_white": int(board.turn), "is_promotion": int(move.promotion is not None),
            "labels": board_to_grid_ids(board).reshape(-1), "legal": legal, "target": move,
            "castling": (int(board.has_kingside_castling_rights(chess.WHITE)),
                         int(board.has_queenside_castling_rights(chess.WHITE)),
                         int(board.has_kingside_castling_rights(chess.BLACK)),
                         int(board.has_queenside_castling_rights(chess.BLACK))),
            "ep_square": board.ep_square if board.ep_square is not None else -1,
        }
        board.push(move)


@torch.inference_mode()
def evaluate_game(game, models, device: torch.device, batch_size: int):
    """Return one complete game for an atomic database transaction."""
    positions = list(game_positions(game))
    predictions = []
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        inputs = {
            "labels64": torch.as_tensor(np.stack([p["labels"] for p in batch]), dtype=torch.long, device=device),
            "turn": torch.tensor([p["turn_white"] for p in batch], dtype=torch.long, device=device),
            "castling": torch.tensor([p["castling"] for p in batch], dtype=torch.long, device=device),
            "ep_square": torch.tensor([p["ep_square"] for p in batch], dtype=torch.long, device=device),
        }
        for category in CATEGORIES:
            output = models[category](**inputs)
            policies = output["policy_logits"].float().cpu().numpy()
            promotions = output["promo_logits"].float().cpu().numpy()
            for p, policy, promotion in zip(batch, policies, promotions):
                score = score_moves(p["legal"], p["target"], policy, promotion)
                predictions.append({
                    "ply": p["ply"], "model": category, "predicted_uci": score.predicted,
                    "target_logp": score.target_logp, "predicted_logp": score.predicted_logp,
                    "top1_correct": int(score.predicted == p["target_uci"]),
                    "top5_correct": int(score.top5_correct), "pair_correct": int(score.pair_correct),
                    "entropy": score.entropy,
                })
    stored_positions = [{k: p[k] for k in (
        "ply", "bucket", "fen_before", "target_uci", "n_legal", "turn_white", "is_promotion")}
        for p in positions]
    return stored_positions, predictions
