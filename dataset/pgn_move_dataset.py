import torch
from torch.utils.data import Dataset
import io
import chess.pgn
import random
from typing import Optional
from models.chess_tokenizer import ChessTokenizer
from dataset.offsets import load_pgn_offsets


class PGNMoveDataset(Dataset):
    def __init__(
        self,
        pgn_path: str,
        index_path: str,
        tokenizer_path: str,                 
        block_size: Optional[int] = None,
        strict: bool = True,
        seed: int = 0,
    ):
        self.pgn_path = pgn_path
        self.index = load_pgn_offsets(index_path)
        self.games = self.index["games"]
        self.tok = ChessTokenizer(tokenizer_path)   
        self.block_size = block_size
        self.strict = strict
        self.rng = random.Random(seed)
        self._fh = None

    def __len__(self):
        return len(self.games)

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.pgn_path, "rb")
        return self._fh

    def _read_game_bytes(self, start: int, end: int) -> bytes:
        fh = self._get_fh()
        fh.seek(start)
        return fh.read(end - start)

    def __getitem__(self, idx: int):
        g = self.games[idx]
        raw = self._read_game_bytes(g["start"], g["end"])
        txt = raw.decode("utf-8", errors="ignore")

        game = chess.pgn.read_game(io.StringIO(txt))
        if game is None:
            return {"valid": False}

        try:
            move_ids, piece_ids = self.tok.encode_pgn_game(game, strict=self.strict)
        except KeyError:
            print(f"Encoding error in game idx={idx}, skipping")
            return {"valid": False}
        

        # optional: sample a fixed-length window (block_size)
        if self.block_size is not None:
            # we need at least 2 tokens to make (x,y)
            if len(move_ids) < 2:
                return {"valid": False}
            if len(move_ids) > self.block_size:
                start = self.rng.randrange(0, len(move_ids) - self.block_size + 1)
                move_ids = move_ids[start:start + self.block_size]
                piece_ids = piece_ids[start:start + self.block_size]
            else:
                start = 0
        
        bos_id = self.tok.bos_id
        # global index of first move token (exclude BOS if present)
        global_first_token_idx = start
        if global_first_token_idx == 0 and len(move_ids) > 0 and move_ids[0] == bos_id:
            # window begins with BOS, first move token is next one in the window
            ply_before = 0
        else:
            # if the full stream had BOS, then token index 1 corresponds to ply 0.
            # So ply_before = (global_first_token_idx - 1) if BOS was in original stream.
            had_bos = (len(move_ids) > 0 and move_ids[0] == bos_id) or True  # encode_pgn_game adds BOS by default
            ply_before = max(0, global_first_token_idx - 1) if had_bos else global_first_token_idx

        board = game.board()
        moves = list(game.mainline_moves())
        # replay exactly ply_before moves
        for k in range(min(ply_before, len(moves))):
            board.push(moves[k])

        start_turn = 0 if board.turn == chess.WHITE else 1
        return {
            "valid": True,
            "move_ids": torch.tensor(move_ids, dtype=torch.long),
            "piece_ids": torch.tensor(piece_ids, dtype=torch.long),
            "start_turn": torch.tensor(start_turn, dtype=torch.long),
        }