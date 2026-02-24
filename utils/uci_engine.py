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

def board_to_piece_probs(board: chess.Board, device):
    x = torch.zeros((1, 64, 13), dtype=torch.float32, device=device)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        x[0, sq, PIECE2IDX.get(p, 0)] = 1.0
    return x

def board_metadata(board: chess.Board, device):
    turn = torch.tensor([0 if board.turn == chess.WHITE else 1], device=device)
    castling = torch.tensor([[
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]], device=device)
    ep = -1 if board.ep_square is None else int(board.ep_square)
    ep_square = torch.tensor([ep], device=device)
    return turn, castling, ep_square

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
    args = ap.parse_args()

    device = torch.device(args.device)

    sd = load_state_dict_any(args.ckpt)
    sd = strip_prefix(sd, "module.")
    sd = strip_prefix(sd, "model.")

    if args.model_type == "decoder":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer_path required for decoder")
        player = ChessDecoderPlayer(tokenizer_path=args.tokenizer_path, max_seq_len=args.max_seq_len)
        player.load_state_dict(sd, strict=False)
        player.eval().to(device)
        tok = player.decoder.tokenizer

        def history_to_input_ids(moves_uci):
            ids = [tok.bos_id]
            for u in moves_uci:
                ids.append(tok.move2id.get(u, tok.unk_id))
            ids = ids[-args.max_seq_len:]
            input_ids = torch.tensor([ids], device=device, dtype=torch.long)
            attn = torch.ones_like(input_ids, dtype=torch.long)
            return input_ids, attn

    else:
        player = ChessEncoderPlayer(img_size=256, vit_path=None, freeze_vit=True)
        player.load_state_dict(sd, strict=False)
        player.eval().to(device)

    # UCI state
    board = chess.Board()
    moves_uci = []

    def set_position(tokens):
        nonlocal board, moves_uci
        moves_uci = []

        if tokens[1] == "startpos":
            board = chess.Board()
            idx = 2
        elif tokens[1] == "fen":
            fen = " ".join(tokens[2:8])
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
            input_ids, attn = history_to_input_ids(moves_uci)
            mv = player.sample_moves(
                input_ids=input_ids,
                attention_mask=attn,
                fen_list=[fen],
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