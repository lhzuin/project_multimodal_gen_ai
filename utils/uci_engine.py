import sys
import time
import argparse
import chess
import torch

from models.chess_decoder_player import ChessDecoderPlayer
from models.chess_encoder_player import ChessEncoderPlayer

# --- encoder helpers (perfect state -> piece_probs) ---
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


def board_metadata(board: chess.Board, device): # Encoder only. Obs: white color=1 (different from decoder's turn encoding)
    turn = torch.tensor([1 if board.turn == chess.WHITE else 0], device=device)
    castling = torch.tensor([[
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]], device=device)
    ep = -1 if board.ep_square is None else int(board.ep_square)
    ep_square = torch.tensor([ep], device=device)
    return turn, castling, ep_square

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

def load_state_dict_any(path: str):
    print(f"Loading checkpoint from {path}...")
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise ValueError("Unrecognized checkpoint format")

def strip_prefix(sd, prefix):
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["decoder", "encoder"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer_path", default=None)  # decoder
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--n_layers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)

    sd = load_state_dict_any(args.ckpt)
    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "model.")

    if args.model_type == "decoder":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer_path required for decoder")
        player = ChessDecoderPlayer(tokenizer_path=args.tokenizer_path, max_seq_len=args.max_seq_len, n_layers=args.n_layers)
        player.load_state_dict(sd, strict=True)
        player.eval().to(device)
        tok = player.decoder.tokenizer

        def history_to_inputs(moves_uci, base_fen: str):
            # We build the SAME logical stream as training:
            #   [BOS] + move_tokens + [EOS]
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

            # add EOS (training does this)
            move_full.append(tok.eos_id)
            piece_full.append(tok.p_eos_id)

            full_len = len(move_full)
            global_start = max(0, full_len - args.max_seq_len)

            # take last max_seq_len tokens (training-style crop from left)
            move_win = move_full[global_start:]
            piece_win = piece_full[global_start:]

            # for inference, remove trailing EOS so we predict the next move token
            if len(move_win) > 0 and move_win[-1] == tok.eos_id:
                move_win = move_win[:-1]
                piece_win = piece_win[:-1]

            input_ids = torch.tensor([move_win], device=device, dtype=torch.long)
            piece_input_ids = torch.tensor([piece_win], device=device, dtype=torch.long)
            attn = torch.ones_like(input_ids, dtype=torch.long)
            return input_ids, piece_input_ids, attn, global_start

    else:
        player = ChessEncoderPlayer(img_size=256, vit_path=None, freeze_vit=True)
        player.load_state_dict(sd, strict=False)
        player.eval().to(device)

    # UCI state
    board = chess.Board()
    moves_uci = []
    base_fen = chess.STARTING_FEN

    def set_position(tokens):
        nonlocal board, moves_uci, base_fen
        moves_uci = []  

        if tokens[1] == "startpos":
            board = chess.Board()
            idx = 2
            base_fen = chess.STARTING_FEN
        elif tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
            base_fen = fen
            board = chess.Board(fen=fen)
            idx = 8
        else:
            return

        if idx < len(tokens) and tokens[idx] == "moves":
            for u in tokens[idx + 1:]:
                mv = chess.Move.from_uci(u)
                if mv in board.legal_moves:
                    board.push(mv)
                    moves_uci.append(u)
                else:
                    break

    def pick_move():
        fen = board.fen()

        if args.model_type == "decoder":
            input_ids, piece_input_ids, attn, global_start = history_to_inputs(moves_uci, base_fen)
            # match PGNMoveDataset logic:
            ply_before = max(0, global_start - 1)

            base_turn = chess.Board(base_fen).turn
            base_turn_idx = 0 if base_turn == chess.WHITE else 1  # your convention
            start_turn = torch.tensor([(base_turn_idx + ply_before) % 2], device=device, dtype=torch.long)

            mv = player.sample_moves(
                input_ids=input_ids,
                piece_input_ids=piece_input_ids,
                attention_mask=attn,
                fen_list=[fen],
                start_turn=start_turn,
                temperature=args.temperature,
                topk=(args.topk if args.topk > 0 else None),
                greedy=args.greedy,
            )[0]
        else:
            piece_probs = board_to_piece_probs(board, device)
            turn, castling, ep_square = board_metadata(board, device)
            mv = player.sample_moves(
                piece_probs=piece_probs,
                turn=turn,
                castling=castling,
                ep_square=ep_square,
                fen=[fen],
                temperature=args.temperature,
                topk=(args.topk if args.topk > 0 else None),
                greedy=args.greedy,
            )[0]

        if mv not in board.legal_moves:
            mv = next(iter(board.legal_moves))  # safety fallback
        return mv

    # Main UCI loop
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        if tokens[0] == "uci":
            print("id name MyModelEngine")
            print("id author you")
            print("uciok", flush=True)

        elif tokens[0] == "isready":
            print("readyok", flush=True)

        elif tokens[0] == "ucinewgame":
            board = chess.Board()
            moves_uci = []
            base_fen = chess.STARTING_FEN

        elif tokens[0] == "position":
            set_position(tokens)

        elif tokens[0] == "go":
            # simplest: support movetime (ms); ignore other time controls for now
            movetime_ms = 200
            if "movetime" in tokens:
                i = tokens.index("movetime")
                if i + 1 < len(tokens):
                    movetime_ms = int(tokens[i + 1])

            t0 = time.time()
            mv = pick_move()
            # optional: sleep until movetime
            dt = (time.time() - t0) * 1000
            if dt < movetime_ms:
                time.sleep((movetime_ms - dt) / 1000.0)

            board.push(mv)
            moves_uci.append(mv.uci())
            print(f"bestmove {mv.uci()}", flush=True)

        elif tokens[0] == "quit":
            break

        else:
            # ignore unknown commands
            pass

if __name__ == "__main__":
    main()