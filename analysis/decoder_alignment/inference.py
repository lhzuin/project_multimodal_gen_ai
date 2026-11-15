"""Reuse the original evaluator's inputs, causal cache and legal UCI metrics."""
from __future__ import annotations

import math
import torch

from analysis.ply_comparison.evaluate import cached_decoder_logits, decoder_result, load_weights
from models.chess_decoder_player import ChessDecoderPlayer


def load_model(checkpoint, tokenizer_path, device):
    model = ChessDecoderPlayer(tokenizer_path=str(tokenizer_path), model_dim=256,
                               n_heads=4, n_layers=12, mlp_ratio=4.0, dropout=0.1,
                               max_seq_len=256, tie_weights=True)
    model.load_state_dict(load_weights(checkpoint), strict=True)
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def predict_game(examples, model, device, batch_size=32):
    """Score one game exactly; restart positional indices for truncated windows.

    Up to ply 255, a single causal pass yields all prefix predictions. Later
    decisions require separate 256-token windows, batched without padding.
    """
    cache = cached_decoder_logits(examples, {"decoder_candidate": model}, device)["decoder_candidate"]
    rows = []

    def append(example, logits):
        predicted, top5, nll = decoder_result(example, logits, model.decoder.tokenizer)
        if not math.isfinite(nll) or nll < 0:
            raise ValueError(f"nonfinite/negative NLL: game {example.game_id}, ply {example.ply}")
        target = example.target.uci()
        rows.append({"ply": example.ply, "predicted_uci": predicted,
                     "top1_correct": int(predicted == target),
                     "top5_correct": int(target in top5), "nll_legal": nll})

    for example in examples[:256]:
        append(example, cache[example.ply])
    for start in range(256, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        ids = torch.tensor([e.move_ids for e in batch], dtype=torch.long, device=device)
        pieces = torch.tensor([e.piece_ids for e in batch], dtype=torch.long, device=device)
        turns = torch.tensor([e.start_turn for e in batch], dtype=torch.long, device=device)
        logits = model(input_ids=ids, piece_input_ids=pieces,
                       attention_mask=torch.ones_like(ids, dtype=torch.bool),
                       start_turn=turns)["logits"][:, -1].detach().float().cpu()
        for example, values in zip(batch, logits):
            append(example, values)
    return rows


def validate_positions(examples, rows):
    """Reject any CSV/PGN/input mismatch before crediting a candidate game."""
    if len(examples) != len(rows):
        raise ValueError("baseline/corpus game length mismatch")
    for example, row in zip(examples, rows):
        expected = {"game_id": example.game_id, "ply": example.ply,
                    "game_plies": example.game_plies, "fen_before": example.fen,
                    "target_uci": example.target.uci(), "twic_issue": example.issue,
                    "white_elo": example.white_elo, "black_elo": example.black_elo,
                    "date": example.date, "n_legal": len(example.legal_moves),
                    "context_len": len(example.move_ids), "context_truncated": int(example.context_truncated)}
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError(f"baseline/corpus position mismatch: game {example.game_id}, ply {example.ply}")
        if row["predicted_uci"] not in {m.uci() for m in example.legal_moves}:
            raise ValueError(f"illegal saved baseline prediction: game {example.game_id}, ply {example.ply}")


def check_baseline_predictions(saved, actual, *, tolerance=3e-4):
    """Small compatibility check, not a rerun of the four-million-position test."""
    if len(saved) != len(actual):
        raise ValueError("saved baseline prediction count differs from the current evaluator")
    for baseline, current in zip(saved, actual):
        if any(baseline[k] != current[k] for k in ("ply", "predicted_uci", "top1_correct", "top5_correct")):
            raise ValueError("saved baseline predictions differ from the current decoder evaluator")
        if not math.isclose(baseline["nll_legal"], current["nll_legal"], abs_tol=tolerance, rel_tol=1e-5):
            raise ValueError("saved baseline likelihoods differ from the current decoder evaluator")
