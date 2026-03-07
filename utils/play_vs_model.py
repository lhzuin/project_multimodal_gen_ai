import argparse
import sys
import torch
import chess

from models.chess_decoder_player import ChessDecoderPlayer
from models.chess_encoder_player import ChessEncoderPlayer


# ----------------------------
# Robust checkpoint loading
# ----------------------------
def load_state_dict_any(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        return obj  # already a state_dict
    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")


def history_to_inputs(tok, moves_uci, device, max_len=256):
    ids = [tok.bos_id]
    piece_ids = [tok.bos_id]

    board = chess.Board()
    for u in moves_uci:
        mv = chess.Move.from_uci(u)
        moved_piece = board.piece_at(mv.from_square)
        # fallback
        p = "P" if moved_piece is None else moved_piece.symbol().upper()  # 'P','N',...
        piece_ids.append(tok.piece2id.get(p, tok.unk_id))

        tid = tok.move2id.get(u, tok.unk_id)
        ids.append(tid)
        board.push(mv)

    ids = ids[-max_len:]
    piece_ids = piece_ids[-max_len:]

    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    piece_input_ids = torch.tensor([piece_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, piece_input_ids, attn


def strip_prefix_if_present(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


# ----------------------------
# Encoder helpers (perfect perception)
# ----------------------------
# 13 channels is consistent with your vit output [B,64,13]. :contentReference[oaicite:2]{index=2}
# We'll use: 0 empty, then 1..12 = P,N,B,R,Q,K,p,n,b,r,q,k
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


# def board_to_piece_probs(board: chess.Board, device):
#     # [1,64,13] one-hot "probabilities"
#     x = torch.zeros((1, 64, 13), dtype=torch.float32, device=device)
#     for sq in chess.SQUARES:
#         p = board.piece_at(sq)
#         idx = PIECE2IDX.get(p, 0)
#         x[0, sq, idx] = 1.0
#     return x


def flip_rank_square(sq: int) -> int:
    # python-chess: sq = file + 8*rank, rank in [0..7]
    file = sq % 8
    rank = sq // 8
    flipped_rank = 7 - rank
    return flipped_rank * 8 + file

def board_to_piece_probs(board: chess.Board, device) -> torch.Tensor:
    x = torch.zeros((1, 64, 13), dtype=torch.float32, device=device)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        idx = PIECE2IDX.get(p, 0)
        ds_idx = flip_rank_square(sq)   # <-- KEY LINE
        x[0, ds_idx, idx] = 1.0
    return x


def board_metadata(board: chess.Board, device):
    # turn: [1] 0=white,1=black (your code expects float later) :contentReference[oaicite:3]{index=3}
    turn = torch.tensor([1 if board.turn == chess.WHITE else 0], device=device)

    # castling: [1,4] (WK,WQ,BK,BQ) :contentReference[oaicite:4]{index=4}
    castling = torch.tensor([[
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]], device=device)

    # ep_square: [1] with -1 if none :contentReference[oaicite:5]{index=5}
    ep = -1 if board.ep_square is None else int(board.ep_square)
    ep_square = torch.tensor([ep], device=device)

    return turn, castling, ep_square


# ----------------------------
# Decoder helpers
# ----------------------------
def history_to_inputs(tok, moves_uci, base_fen: str, device, max_len=256):
    move_full = [tok.bos_id]
    piece_full = [tok.p_bos_id]

    tmp = chess.Board(fen=base_fen)

    for u in moves_uci:
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

    # training-style EOS
    move_full.append(tok.eos_id)
    piece_full.append(tok.p_eos_id)

    full_len = len(move_full)
    global_start = max(0, full_len - max_len)

    move_win = move_full[global_start:]
    piece_win = piece_full[global_start:]

    # inference: remove trailing EOS so we predict next token
    if len(move_win) > 0 and move_win[-1] == tok.eos_id:
        move_win = move_win[:-1]
        piece_win = piece_win[:-1]

    input_ids = torch.tensor([move_win], device=device, dtype=torch.long)
    piece_input_ids = torch.tensor([piece_win], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, piece_input_ids, attn, global_start


# ----------------------------
# CLI loop
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["decoder", "encoder"], required=True)
    ap.add_argument("--ckpt", required=True, help="Path to .pt/.pth checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # decoder-specific
    ap.add_argument("--tokenizer_path", default=None, help="Needed for decoder (ChessLLM tokenizer)")
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=12)

    # encoder-specific
    ap.add_argument("--img_size", type=int, default=256, help="Only used to construct ChessEncoderPlayer")
    ap.add_argument("--vit_path", default=None, help="Optional; not needed if using perfect piece_probs")

    # play options
    ap.add_argument("--you", choices=["white", "black"], default="white")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device)

    # --- build + load model
    sd = load_state_dict_any(args.ckpt)

    if args.model_type == "decoder":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer_path is required for decoder model.")
        model = ChessDecoderPlayer(
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            n_layers=args.n_layers,
        )
        # common patterns: "module." or "model."
        sd2 = strip_prefix_if_present(sd, "module.")
        sd2 = strip_prefix_if_present(sd2, "model.")
        model.load_state_dict(sd2, strict=True)
        model.eval().to(device)
        tok = model.decoder.tokenizer

    else:
        model = ChessEncoderPlayer(
            img_size=args.img_size,
            vit_path=args.vit_path,
            freeze_vit=True,
        )
        sd2 = strip_prefix_if_present(sd, "module.")
        sd2 = strip_prefix_if_present(sd2, "model.")
        model.load_state_dict(sd2, strict=True)
        model.eval().to(device)

    # --- game loop
    board = chess.Board()
    human_is_white = (args.you == "white")

    # decoder history in UCI
    moves_uci = []
    base_fen = chess.STARTING_FEN

    def model_move():
        fen = board.fen()

        if args.model_type == "decoder":
            input_ids, piece_input_ids, attn, global_start = history_to_inputs(
                tok, moves_uci, base_fen, device, max_len=args.max_seq_len
            )

            ply_before = max(0, global_start - 1)
            base_turn = chess.Board(base_fen).turn
            base_turn_idx = 0 if base_turn == chess.WHITE else 1
            start_turn = torch.tensor([(base_turn_idx + ply_before) % 2], device=device, dtype=torch.long)

            mv = model.sample_moves(
                input_ids=input_ids,
                piece_input_ids=piece_input_ids,
                attention_mask=attn,
                fen_list=[fen],
                start_turn=start_turn,
                temperature=args.temperature,
                topk=(args.topk if args.topk > 0 else None),
                greedy=args.greedy,
            )[0]
            return mv

        else:
            piece_probs = board_to_piece_probs(board, device)
            turn, castling, ep_square = board_metadata(board, device)
            mv = model.sample_moves(
                piece_probs=piece_probs,
                turn=turn,
                castling=castling,
                ep_square=ep_square,
                fen=[fen],
                temperature=args.temperature,
                topk=(args.topk if args.topk > 0 else None),
                greedy=args.greedy,
            )[0]
            return mv

    while not board.is_game_over():
        print(board)
        print("FEN:", board.fen())
        side_to_move_is_white = (board.turn == chess.WHITE)
        human_turn = (side_to_move_is_white and human_is_white) or ((not side_to_move_is_white) and (not human_is_white))

        if human_turn:
            s = input("Your move (UCI like e2e4, or 'quit'): ").strip()
            if s.lower() in {"q", "quit", "exit"}:
                break
            try:
                mv = chess.Move.from_uci(s)
            except Exception:
                print("Invalid UCI format.")
                continue
            if mv not in board.legal_moves:
                print("Illegal move.")
                continue
            board.push(mv)
            moves_uci.append(mv.uci())
        else:
            mv = model_move()
            if mv not in board.legal_moves:
                # should not happen due to legal masking, but keep a safe fallback
                mv = next(iter(board.legal_moves))
            print("Model plays:", mv.uci())
            board.push(mv)
            moves_uci.append(mv.uci())

    print("Game over:", board.result(), board.outcome())


if __name__ == "__main__":
    main()