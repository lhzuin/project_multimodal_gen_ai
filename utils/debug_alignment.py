import argparse
import random
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import chess

# Adjust these imports to your repo layout:
# - ChessGameSampleDataset from your dataset.py
# - CollatePositionsFromGames if you want batching (optional)
from dataset.dataset import ChessGameSampleDataset
from models.chess_encoder_player import ChessEncoderPlayer


# -------------------------
# Inference-style encoding (the thing your engine does)
# -------------------------
PIECE2IDX = {
    None: 0,
    chess.Piece(chess.PAWN, chess.WHITE): 1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK, chess.WHITE): 4,
    chess.Piece(chess.QUEEN, chess.WHITE): 5,
    chess.Piece(chess.KING, chess.WHITE): 6,
    chess.Piece(chess.PAWN, chess.BLACK): 7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.BLACK): 10,
    chess.Piece(chess.QUEEN, chess.BLACK): 11,
    chess.Piece(chess.KING, chess.BLACK): 12,
}

IDX2PIECE = {v: k for k, v in PIECE2IDX.items()}

def board_to_piece_probs_pythonchess(board: chess.Board, device) -> torch.Tensor:
    # [1,64,13], indexed by python-chess square id (0=a1 ... 63=h8)
    x = torch.zeros((1, 64, 13), dtype=torch.float32, device=device)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        idx = PIECE2IDX.get(p, 0)
        x[0, sq, idx] = 1.0
    return x

def board_metadata_pythonchess(board: chess.Board, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # IMPORTANT: must match dataset encoding
    # dataset: turn = 1 if white to move else 0
    turn = torch.tensor([1 if board.turn == chess.WHITE else 0], device=device, dtype=torch.long)

    castling = torch.tensor([[
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]], device=device, dtype=torch.long)

    ep = -1 if board.ep_square is None else int(board.ep_square)
    ep_square = torch.tensor([ep], device=device, dtype=torch.long)
    return turn, castling, ep_square


# -------------------------
# Helpers to detect square-order transforms
# -------------------------
def idx_to_rc(idx: int) -> Tuple[int, int]:
    # python-chess squares: 0=a1, file increases, then rank increases
    file = idx % 8
    rank = idx // 8
    return rank, file  # r in [0..7] (rank1..8), c in [0..7] (a..h)

def rc_to_idx(r: int, c: int) -> int:
    return r * 8 + c

def apply_transform_idx(idx: int, tname: str) -> int:
    r, c = idx_to_rc(idx)
    if tname == "identity":
        rr, cc = r, c
    elif tname == "flip_rank":          # vertical flip
        rr, cc = 7 - r, c
    elif tname == "flip_file":          # horizontal flip
        rr, cc = r, 7 - c
    elif tname == "rot180":
        rr, cc = 7 - r, 7 - c
    elif tname == "transpose":          # swap r,c
        rr, cc = c, r
    elif tname == "rot90":
        # (r,c) -> (c, 7-r)
        rr, cc = c, 7 - r
    elif tname == "rot270":
        # (r,c) -> (7-c, r)
        rr, cc = 7 - c, r
    elif tname == "anti_transpose":
        # (r,c)->(7-c,7-r) then swap? common anti-diagonal reflection
        rr, cc = 7 - c, 7 - r
    else:
        raise ValueError(tname)
    return rc_to_idx(rr, cc)

TRANSFORMS = [
    "identity",
    "flip_rank",
    "flip_file",
    "rot180",
    "transpose",
    "rot90",
    "rot270",
    "anti_transpose",
]

def best_square_transform(dataset_piece_ids: np.ndarray, board_piece_ids: np.ndarray) -> Tuple[str, int]:
    """
    dataset_piece_ids: [64] predicted piece-id at each dataset position index i
    board_piece_ids:   [64] piece-id at python-chess square index sq
    We test whether dataset index i corresponds to some transformed chess square.
    """
    best = ("identity", 10**9)
    for t in TRANSFORMS:
        mism = 0
        for i in range(64):
            sq = apply_transform_idx(i, t)
            if dataset_piece_ids[i] != board_piece_ids[sq]:
                mism += 1
        if mism < best[1]:
            best = (t, mism)
    return best


# -------------------------
# Core tests
# -------------------------
@torch.no_grad()
def run_one_sample_checks(
    model: ChessEncoderPlayer,
    sample: Dict,
    device: torch.device,
    topk_moves: int = 12,
) -> None:
    assert sample.get("valid", True), "Sample marked invalid"

    fen = sample["fen"]
    board = chess.Board(fen)

    # --- (A) Compare metadata encodings ---
    ds_turn = sample["turn"].view(1).to(device)
    ds_castling = sample["castling"].view(1, 4).to(device)
    ds_ep = sample["ep_square"].view(1).to(device)

    inf_turn, inf_castling, inf_ep = board_metadata_pythonchess(board, device)

    print("\n=== METADATA CHECK ===")
    print(f"FEN: {fen}")
    print(f"dataset turn={int(ds_turn.item())}, inference turn={int(inf_turn.item())}  (1=white-to-move expected)")
    print(f"dataset castling={ds_castling.cpu().tolist()}, inference castling={inf_castling.cpu().tolist()}")
    print(f"dataset ep={int(ds_ep.item())}, inference ep={int(inf_ep.item())}")

    if int(ds_turn.item()) != int(inf_turn.item()):
        print("!! MISMATCH: turn encoding differs between dataset and inference.")
    if not torch.equal(ds_castling, inf_castling):
        print("!! MISMATCH: castling encoding differs between dataset and inference.")
    if int(ds_ep.item()) != int(inf_ep.item()):
        print("!! MISMATCH: ep_square differs between dataset and inference.")

    # --- (B) Compare piece_probs mapping & square ordering ---
    print("\n=== PIECE PROBS / SQUARE ORDER CHECK ===")
    assert "piece_probs" in sample, "Need dataset(return_piece_probs=True) to run this check."
    ds_piece_probs = sample["piece_probs"].view(1, 64, 13).to(device)

    inf_piece_probs = board_to_piece_probs_pythonchess(board, device)  # [1,64,13]

    ds_ids = ds_piece_probs.argmax(dim=-1).view(-1).cpu().numpy()      # [64]
    inf_ids = inf_piece_probs.argmax(dim=-1).view(-1).cpu().numpy()    # [64], by python-chess square id

    # Direct index-by-index mismatch count:
    direct_mism = int(np.sum(ds_ids != inf_ids))
    print(f"Direct (same index) mismatches: {direct_mism} / 64")

    # Transform detection:
    tname, mism = best_square_transform(ds_ids, inf_ids)
    print(f"Best transform mapping dataset_index -> chess_square is: {tname} with mismatches={mism}/64")
    if mism > 0:
        print("First few mismatches under best transform:")
        shown = 0
        for i in range(64):
            sq = apply_transform_idx(i, tname)
            if ds_ids[i] != inf_ids[sq]:
                ds_piece = IDX2PIECE.get(int(ds_ids[i]), None)
                bd_piece = IDX2PIECE.get(int(inf_ids[sq]), None)
                print(f"  dataset idx {i:2d} vs chess sq {chess.square_name(sq)}: dataset={ds_piece} board={bd_piece}")
                shown += 1
                if shown >= 10:
                    break

    if tname != "identity" and mism == 0:
        print("!! Strong signal: your dataset square indexing is a transformed view of python-chess squares.")
        print("   That means your inference board_to_piece_probs likely uses the wrong square order.")
        print("   Fix by reusing the SAME board_to_grid_ids / ordering everywhere.")

    # --- (C) Target move indexing sanity ---
    print("\n=== TARGET MOVE / LEGALITY CHECK ===")
    mf = int(sample["move_from"].item())
    mt = int(sample["move_to"].item())
    mp = int(sample["move_promo"].item())
    uci_basic = chess.Move(mf, mt)
    print(f"Target (from,to) = ({mf},{mt}) = {chess.square_name(mf)}->{chess.square_name(mt)}  promo_id={mp}")
    print(f"Is (from,to) legal ignoring promotion detail? {any(m.from_square==mf and m.to_square==mt for m in board.legal_moves)}")
    print(f"Is exact UCI move object legal (no promo)? {uci_basic in board.legal_moves}")

    # --- (D) Forward pass + illegal mass check ---
    print("\n=== MODEL FORWARD + MASKING CHECK ===")
    model.eval()

    out = model(
        piece_probs=ds_piece_probs,     # IMPORTANT: we test using EXACT dataset feature tensor
        turn=ds_turn,
        castling=ds_castling,
        ep_square=ds_ep,
    )
    policy_logits = out["policy_logits"]  # [1,64,64]
    promo_logits = out["promo_logits"]    # [1,5]

    # legal mask from fen
    legal_mask = model.legal_mask_from_fen([fen], device=device)  # [1,64,64]
    masked = model.apply_legal_mask(policy_logits, legal_mask)
    flat = masked.view(1, -1)  # [1,4096]
    probs = F.softmax(flat, dim=-1).view(64, 64)

    illegal_mass = float(probs.masked_fill(legal_mask[0], 0.0).sum().item())
    legal_mass = float(probs.masked_fill(~legal_mask[0], 0.0).sum().item())
    print(f"Total prob mass on legal moves:  {legal_mass:.6f}")
    print(f"Total prob mass on illegal moves:{illegal_mass:.6f}  (should be ~0)")

    # show top moves
    flat_probs = probs.view(-1)
    topv, topi = torch.topk(flat_probs, k=topk_moves)
    print(f"\nTop-{topk_moves} moves by prob (after masking):")
    for rank, (v, idx) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        fs = idx // 64
        ts = idx % 64
        cands = [m for m in board.legal_moves if m.from_square == fs and m.to_square == ts]
        uci = chess.Move(fs, ts).uci()
        if len(cands) == 1:
            uci = cands[0].uci()
        elif len(cands) > 1:
            # promotion ambiguity
            uci = cands[0].uci() + f" (+{len(cands)-1} promos)"
        print(f"  {rank:2d}. p={v:.6f}  {chess.square_name(fs)}->{chess.square_name(ts)}  {uci}")

    # target prob (from,to)
    tgt_idx = mf * 64 + mt
    tgt_p = float(flat_probs[tgt_idx].item())
    print(f"\nProb assigned to target (from,to) after masking: p={tgt_p:.6f}")
    print(f"Promo logits (softmax): {F.softmax(promo_logits[0], dim=-1).detach().cpu().numpy().round(4).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn_path", default="games_large.pgn")
    ap.add_argument("--index_path", default="games_large_index.json")
    ap.add_argument("--sprites_dir", default="dataset/sprites")
    ap.add_argument("--ckpt", default="checkpoints/chess_encoder_player_v3.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_trials", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--sample_ratio", type=float, default=0.10)
    ap.add_argument("--max_positions_per_game", type=int, default=32)
    ap.add_argument("--apply_augmentations", action="store_true")
    args = ap.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # IMPORTANT: return_piece_probs=True + include fen
    ds = ChessGameSampleDataset(
        pgn_path=args.pgn_path,
        index_path=args.index_path,
        sprites_dir=args.sprites_dir,
        resolution=args.resolution,
        sample_ratio=args.sample_ratio,
        max_positions_per_game=args.max_positions_per_game,
        seed=args.seed,
        apply_augmentations=args.apply_augmentations,
        return_images=False,
        return_piece_probs=True,
        return_fen=True,
    )

    # Load model
    model = ChessEncoderPlayer(
        img_size=args.resolution,
        encoder_dim=256,
        n_heads=4,
        n_layers=6,
        encoder_dropout=0.1,
        vit_path=None,
        freeze_vit=True,
    ).to(device)

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

    # Strip common prefixes if needed
    def strip_prefix(state, prefix: str):
        if any(kk.startswith(prefix) for kk in state.keys()):
            return {kk[len(prefix):]: vv for kk, vv in state.items()}
        return state
    sd = strip_prefix(sd, "model.")
    sd = strip_prefix(sd, "module.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint.")
    if missing:
        print("Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    if unexpected:
        print("Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    # Run a few random samples
    print(f"\nDataset size: {len(ds)}")
    for t in range(args.num_trials):
        idx = random.randrange(len(ds))
        sample = ds[idx]
        if not sample.get("valid", False):
            print(f"\n[trial {t}] sample idx {idx} invalid; skipping")
            continue
        print(f"\n\n==============================")
        print(f"TRIAL {t}  (dataset idx={idx})")
        print(f"==============================")
        run_one_sample_checks(model, sample, device=device)

if __name__ == "__main__":
    main()