#!/usr/bin/env python3
"""
Plot teacher vs encoder vs decoder policy distributions for one specific position,
keeping only the first 5 moves from the original ordered union.

This version is specialized for:
    idx = 623762

Output:
    pos_004_idx_623762_top5.png
"""

from omegaconf import OmegaConf
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import chess

# ---- repo-local imports ----
from dataset.distillation_db import (
    DistillationEncoderDataset,
    DistillationShardReader,
    _MultiShardIndex,
    _scores_to_probs,
)
from models.chess_encoder_player import ChessEncoderPlayerV2
from models.chess_decoder_player import ChessDecoderPlayer


PROMO_TO_CLASS = {
    None: 0,  # "none"
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def register_minimal_hydra_resolver():
    if not OmegaConf.has_resolver("hydra"):
        def _hydra_resolver(key: str):
            if key == "runtime.cwd":
                return os.getcwd()
            raise ValueError(f"Unsupported hydra interpolation: {key}")
        OmegaConf.register_new_resolver("hydra", _hydra_resolver)


def load_model_from_cfg(cfg, ckpt_path: str, device: str):
    mcfg = cfg.model.instance

    model = ChessEncoderPlayerV2(
        img_size=int(mcfg.img_size),
        encoder_dim=int(mcfg.encoder_dim),
        n_heads=int(mcfg.n_heads),
        n_layers=int(mcfg.n_layers),
        encoder_dropout=float(mcfg.encoder_dropout),
        ep_embed_dim=int(mcfg.ep_embed_dim),
        vit_path=mcfg.get("vit_path", None),
        freeze_vit=bool(mcfg.get("freeze_vit", True)),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_model] missing keys:", len(missing))
        print("[load_model] unexpected keys:", len(unexpected))

    model.eval()
    return model


def load_decoder_model(
    *,
    tokenizer_path: str,
    ckpt_path: str,
    device: str,
    model_dim: int = 256,
    mlp_ratio: float = 4.0,
    n_heads: int = 4,
    n_layers: int = 6,
    dropout: float = 0.1,
    max_seq_len: int = 256,
    tie_weights: bool = True,
    use_turn_embed: bool = True,
):
    model = ChessDecoderPlayer(
        tokenizer_path=tokenizer_path,
        model_dim=model_dim,
        mlp_ratio=mlp_ratio,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
        tie_weights=tie_weights,
        use_turn_embed=use_turn_embed,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_decoder] missing keys:", len(missing))
        print("[load_decoder] unexpected keys:", len(unexpected))

    model.eval()
    return model


def decoder_inputs_from_uci_prefix(
    decoder: ChessDecoderPlayer,
    uci_prefix: list[str],
    *,
    base_fen: str,
    device: str,
    max_seq_len: int,
):
    tok = decoder.decoder.tokenizer

    move_full = [tok.bos_id]
    piece_full = [tok.p_bos_id]

    tmp = chess.Board(fen=base_fen)

    for u in uci_prefix:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break

        if mv not in tmp.legal_moves:
            break

        moved_piece = tmp.piece_at(mv.from_square)
        p = "P" if moved_piece is None else moved_piece.symbol().upper()

        move_full.append(tok.move2id.get(u, tok.unk_id))
        piece_full.append(tok.piece2id.get(p, tok.p_unk_id))

        tmp.push(mv)

    move_full.append(tok.eos_id)
    piece_full.append(tok.p_eos_id)

    full_len = len(move_full)
    global_start = max(0, full_len - max_seq_len)

    move_win = move_full[global_start:]
    piece_win = piece_full[global_start:]

    if len(move_win) > 0 and move_win[-1] == tok.eos_id:
        move_win = move_win[:-1]
        piece_win = piece_win[:-1]

    input_ids = torch.tensor([move_win], dtype=torch.long, device=device)
    piece_input_ids = torch.tensor([piece_win], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    return input_ids, piece_input_ids, attention_mask, global_start


@torch.no_grad()
def decoder_distribution_over_legal_moves(
    decoder: ChessDecoderPlayer,
    *,
    fen: str,
    uci_prefix: list[str],
    device: str,
    temperature: float = 1.0,
):
    tok = decoder.decoder.tokenizer

    max_seq_len = (
        decoder.decoder.pos_enc.pe.size(1)
        if hasattr(decoder.decoder, "pos_enc") else 256
    )

    base_fen = chess.STARTING_FEN

    input_ids, piece_input_ids, attention_mask, global_start = decoder_inputs_from_uci_prefix(
        decoder,
        uci_prefix,
        base_fen=base_fen,
        device=device,
        max_seq_len=max_seq_len,
    )

    ply_before = max(0, global_start - 1)

    base_turn = chess.Board(base_fen).turn
    base_turn_idx = 0 if base_turn == chess.WHITE else 1
    start_turn = torch.tensor(
        [(base_turn_idx + ply_before) % 2],
        dtype=torch.long,
        device=device,
    )

    out = decoder(
        input_ids=input_ids,
        piece_input_ids=piece_input_ids,
        attention_mask=attention_mask,
        start_turn=start_turn,
    )

    next_logits = out["logits"][:, -1, :]
    next_logits = next_logits / max(1e-6, float(temperature))

    legal_mask = decoder.legal_token_mask_from_fen([fen], device=next_logits.device)
    next_logits = decoder.apply_token_mask(next_logits, legal_mask)

    probs = torch.softmax(next_logits[0], dim=-1).detach().cpu()

    board = chess.Board(fen)
    out_dict = {}
    for mv in board.legal_moves:
        uci = mv.uci()
        tid = tok.move2id.get(uci, None)
        if tid is None:
            continue
        out_dict[uci] = float(probs[tid].item())

    return out_dict


@torch.no_grad()
def model_distribution_over_legal_moves(model, *, item, device: str, temperature: float = 1.0):
    """
    Returns dict {uci: prob} over ALL legal moves in the position.

    Important:
    - For ChessEncoderPlayerV2, prefer labels64 if available, because that is the
      native embedding path.
    - Output policy_logits are indexed in raw python-chess coordinates, exactly
      like sample_moves() does.
    """
    fen = item["fen"]
    board = chess.Board(fen)

    turn = item["turn"].unsqueeze(0).to(device)                # [1]
    castling = item["castling"].unsqueeze(0).to(device)        # [1,4]
    ep_square = item["ep_square"].unsqueeze(0).to(device)      # [1]

    # Use the same input style the model expects best
    if "labels64" in item:
        labels64 = item["labels64"].unsqueeze(0).to(device)    # [1,64]
        out = model(
            labels64=labels64,
            turn=turn,
            castling=castling,
            ep_square=ep_square,
        )
    else:
        piece_probs = item["piece_probs"].unsqueeze(0).to(device)  # [1,64,13]
        out = model(
            piece_probs=piece_probs,
            turn=turn,
            castling=castling,
            ep_square=ep_square,
        )

    policy_logits = out["policy_logits"][0]  # [64,64]
    promo_logits = out["promo_logits"][0]    # [5]

    moves = list(board.legal_moves)
    if len(moves) == 0:
        return {}

    logits = []
    uci_list = []
    for mv in moves:
        # IMPORTANT: raw python-chess coordinates, same as sample_moves()
        base = policy_logits[mv.from_square, mv.to_square]
        promo_class = PROMO_TO_CLASS.get(mv.promotion, 0)
        full = base + promo_logits[promo_class]
        logits.append(full)
        uci_list.append(mv.uci())

    logits = torch.stack(logits, dim=0).float()
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

    return {u: float(p) for u, p in zip(uci_list, probs)}


def teacher_distribution_from_row(row, *, tau: float, mate_cp: int):
    uci_topk = json.loads(row["uci_topk_json"])
    cp_topk = json.loads(row["cp_topk_json"])
    mate_topk = json.loads(row["mate_topk_json"])

    probs = _scores_to_probs(
        cp_topk,
        mate_topk,
        tau=float(tau),
        mate_cp=int(mate_cp),
    ).cpu().numpy()

    return {u: float(p) for u, p in zip(uci_topk, probs)}


def choose_union_moves_3(
    teacher_probs: dict,
    enc_probs: dict,
    dec_probs: dict,
    *,
    enc_topk: int = 10,
    dec_topk: int = 10,
):
    enc_sorted = sorted(enc_probs.items(), key=lambda kv: kv[1], reverse=True)
    dec_sorted = sorted(dec_probs.items(), key=lambda kv: kv[1], reverse=True)

    top_enc = [u for u, _ in enc_sorted[:enc_topk]]
    top_dec = [u for u, _ in dec_sorted[:dec_topk]]

    union = set(teacher_probs.keys()) | set(top_enc) | set(top_dec)

    def score(u):
        return max(
            teacher_probs.get(u, 0.0),
            enc_probs.get(u, 0.0),
            dec_probs.get(u, 0.0),
        )

    return sorted(list(union), key=score, reverse=True)


# def plot_one_position(
#     *,
#     idx: int,
#     fen: str,
#     moves: list[str],
#     teacher_probs: dict,
#     enc_probs: dict,
#     dec_probs: dict,
#     outpath: Path,
# ):
#     t = np.array([teacher_probs.get(u, 0.0) for u in moves], dtype=np.float32)
#     e = np.array([enc_probs.get(u, 0.0) for u in moves], dtype=np.float32)
#     d = np.array([dec_probs.get(u, 0.0) for u in moves], dtype=np.float32)

#     x = np.arange(len(moves))
#     w = 0.25

#     plt.figure(figsize=(8, 5))
#     plt.bar(x - w, t, width=w, label="Teacher")
#     plt.bar(x,     e, width=w, label="Encoder")
#     plt.bar(x + w, d, width=w, label="Decoder")

#     plt.xticks(x, moves, rotation=35, ha="right", fontsize=10)
#     plt.ylabel("Probability")
#     plt.title("Probability Distribution Comparison")
#     ymax = float(max(t.max(initial=0.0), e.max(initial=0.0), d.max(initial=0.0)))
#     plt.ylim(0.0, max(0.05, ymax * 1.15))
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=180)
#     plt.close()

def plot_one_position(
    *,
    idx: int,
    fen: str,
    moves: list[str],
    teacher_probs: dict,
    enc_probs: dict,
    dec_probs: dict,
    outpath: Path,
):
    t = np.array([teacher_probs.get(u, 0.0) for u in moves], dtype=np.float32)
    e = np.array([enc_probs.get(u, 0.0) for u in moves], dtype=np.float32)
    d = np.array([dec_probs.get(u, 0.0) for u in moves], dtype=np.float32)

    x = np.arange(len(moves))
    w = 0.25

    plt.figure(figsize=(9, 5.8))
    plt.bar(x - w, t, width=w, label="Teacher")
    plt.bar(x,     e, width=w, label="Encoder")
    plt.bar(x + w, d, width=w, label="Decoder")

    plt.xticks(x, moves, rotation=35, ha="right", fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel("Probability", fontsize=15)
    plt.title(
        "Probability Distribution Comparison",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

    ymax = float(max(t.max(initial=0.0), e.max(initial=0.0), d.max(initial=0.0)))
    plt.ylim(0.0, max(0.05, ymax * 1.15))

    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/distillation_encoder_v9.yaml")
    ap.add_argument("--cfg_encoder", type=str, default="configs/chess_encoder_seq_v1.yaml")
    ap.add_argument("--ckpt", type=str, default="checkpoints/chess_encoder_player_v3.pt")
    ap.add_argument("--decoder_ckpt", type=str, default="checkpoints/chess_llm_decoder_v9.pt")
    ap.add_argument("--decoder_tokenizer_path", type=str, default="tokenizers/chess_uci_vocab.json")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--outdir", type=str, default="distill_plots")
    ap.add_argument("--model_topk", type=int, default=10)
    ap.add_argument("--decoder_topk", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--decoder_temperature", type=float, default=1.0)
    args = ap.parse_args()

    register_minimal_hydra_resolver()
    cfg = OmegaConf.load(args.cfg)
    cfg_encoder = OmegaConf.load(args.cfg_encoder)

    meta_path = str(cfg.distill.meta_path)
    tau = float(cfg.distill.tau)
    mate_cp = int(cfg.distill.mate_cp)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = DistillationEncoderDataset(
        meta_path=meta_path,
        tau=tau,
        mate_cp=mate_cp,
        return_text=False,
        return_state_tensors=True,
    )

    index = _MultiShardIndex(meta_path)
    readers = [DistillationShardReader(p) for (p, _) in index.shards]

    model = load_model_from_cfg(cfg_encoder, args.ckpt, args.device)

    decoder = load_decoder_model(
        tokenizer_path=args.decoder_tokenizer_path,
        ckpt_path=args.decoder_ckpt,
        device=args.device,
        model_dim=256,
        mlp_ratio=4.0,
        n_heads=4,
        n_layers=12,
        dropout=0.1,
        max_seq_len=256,
        tie_weights=True,
        use_turn_embed=True,
    )

    # Specific case only
    idx = 623762
    item = ds[idx]
    if not item.get("valid", False):
        raise RuntimeError(f"Item idx={idx} is not valid.")

    shard_id, local = index.locate(int(idx))
    row = readers[shard_id].get_by_index(int(local))
    uci_prefix = json.loads(row["uci_prefix_json"])

    teacher_probs = teacher_distribution_from_row(row, tau=tau, mate_cp=mate_cp)
    enc_probs = model_distribution_over_legal_moves(
        model,
        item=item,
        device=args.device,
        temperature=args.temperature,
    )
    dec_probs = decoder_distribution_over_legal_moves(
        decoder,
        fen=item["fen"],
        uci_prefix=uci_prefix,
        device=args.device,
        temperature=args.decoder_temperature,
    )

    # Build the same ordered union logic as before
    moves_union = choose_union_moves_3(
        teacher_probs,
        enc_probs,
        dec_probs,
        enc_topk=int(args.model_topk),
        dec_topk=int(args.decoder_topk),
    )

    # Keep only the first 5 moves that appear in the original ordering
    moves_top5 = moves_union[:5]

    print("[info] idx =", idx)
    print("[info] FEN =", item["fen"])
    print("[info] top-5 moves =", moves_top5)

    outpath = outdir / "pos_004_idx_623762_top5.png"
    plot_one_position(
        idx=idx,
        fen=item["fen"],
        moves=moves_top5,
        teacher_probs=teacher_probs,
        enc_probs=enc_probs,
        dec_probs=dec_probs,
        outpath=outpath,
    )

    print(f"[done] saved to {outpath.resolve()}")


if __name__ == "__main__":
    main()