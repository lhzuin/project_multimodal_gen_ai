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
        tokenizer: ChessTokenizer,
        block_size: Optional[int] = None,  # if set, sample random window
        strict: bool = True,
        seed: int = 0,
    ):
        self.pgn_path = pgn_path
        self.index = load_pgn_offsets(index_path)
        self.games = self.index["games"]
        self.tok = tokenizer
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

        move_ids, piece_ids = self.tok.encode_pgn_game(game, strict=self.strict)

        # optional: sample a fixed-length window (block_size)
        if self.block_size is not None:
            # we need at least 2 tokens to make (x,y)
            if len(move_ids) < 2:
                return {"valid": False}
            if len(move_ids) > self.block_size:
                start = self.rng.randrange(0, len(move_ids) - self.block_size + 1)
                move_ids = move_ids[start:start + self.block_size]
                piece_ids = piece_ids[start:start + self.block_size]

        return {
            "valid": True,
            "move_ids": torch.tensor(move_ids, dtype=torch.long),
            "piece_ids": torch.tensor(piece_ids, dtype=torch.long),
        }