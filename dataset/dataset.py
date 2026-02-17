import io
import random
from typing import Callable, Optional, Any, Dict
import torch
from torch.utils.data import Dataset
import chess.pgn
import numpy as np
from renderer import encode_move_from_board
from offsets import load_pgn_offsets
from renderer import SpriteBoardRenderer, board_to_grid_ids
from PIL import Image

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    PIL RGB -> float tensor [3,H,W] in [0,1]
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))               # [3,H,W]
    return torch.from_numpy(arr)

class ChessGameSampleDataset(Dataset):
    """
    Map-style dataset:
      __getitem__(i) -> returns a dict containing M sampled positions from game i.

    Requirements:
      - pgn must be uncompressed so we can seek by byte offsets.
      - index must be built with build_pgn_offsets.
    """
    def __init__(
        self,
        pgn_path: str,
        index_path: str,
        sprites_dir: str,
        resolution: int = 256,
        sample_ratio: float = 0.10,
        max_positions_per_game: int = 32,
        seed: int = 0,
        augment: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        return_fen: bool = False,
    ):
        self.pgn_path = pgn_path
        self.index = load_pgn_offsets(index_path)
        self.games = self.index["games"]
        self.resolution = int(resolution)
        self.sample_ratio = float(sample_ratio)
        self.max_positions_per_game = int(max_positions_per_game)
        self.rng = random.Random(seed)
        self.augment = augment
        self.return_fen = return_fen

        # choose square_px so 8*square_px close-ish to resolution (then render resizes anyway)
        square_px = max(24, resolution // 8)
        self.renderer = SpriteBoardRenderer(sprites_dir=sprites_dir, square_px=square_px)

        # keep one file handle per worker (opened lazily)
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

    def _parse_game(self, game_bytes: bytes) -> Optional[chess.pgn.Game]:
        # decode with errors ignored; PGNs are mostly ascii/utf-8
        txt = game_bytes.decode("utf-8", errors="ignore")
        pgn_io = io.StringIO(txt)
        try:
            return chess.pgn.read_game(pgn_io)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = self.games[idx]
        game_bytes = self._read_game_bytes(g["start"], g["end"])
        game = self._parse_game(game_bytes)
        if game is None:
            # return empty sample (collate will skip)
            return {"images": torch.empty(0,3,self.resolution,self.resolution), "valid": False}

        # extract all ply positions (pre-move)
        board = game.board()
        moves = list(game.mainline_moves())
        n = len(moves)
        if n == 0:
            return {"images": torch.empty(0,3,self.resolution,self.resolution), "valid": False}

        k = max(1, int(round(self.sample_ratio * n)))
        k = min(k, n, self.max_positions_per_game)

        # sample ply indices
        chosen = sorted(self.rng.sample(range(n), k=k))

        # prepare outputs
        images = []
        labels64 = []
        meta_turn = []
        meta_castling = []
        meta_ep = []
        move_from = []
        move_to = []
        move_promo = []
        next_from = []
        next_to = []
        next_promo = []
        fens = [] if self.return_fen else None
        ply_indices = []

        # We need fast access to position at chosen plies:
        # simplest: replay once, and when at ply i, if chosen, record.
        chosen_set = set(chosen)
        chosen_ptr = 0

        for ply, mv in enumerate(moves):
            # position BEFORE pushing mv is what we want to render/label
            if ply in chosen_set:
                # label grid + metadata
                grid = board_to_grid_ids(board)  # (8,8)
                labels64.append(torch.from_numpy(grid.reshape(-1)).long())

                meta_turn.append(torch.tensor(1 if board.turn else 0, dtype=torch.long))  # 1=w,0=b
                # castling: 4 bits KQkq
                ck  = int(board.has_kingside_castling_rights(chess.WHITE))
                cq  = int(board.has_queenside_castling_rights(chess.WHITE))
                ck2 = int(board.has_kingside_castling_rights(chess.BLACK))
                cq2 = int(board.has_queenside_castling_rights(chess.BLACK))
                meta_castling.append(torch.tensor([ck, cq, ck2, cq2], dtype=torch.long))
                meta_ep.append(torch.tensor(board.ep_square if board.ep_square is not None else -1, dtype=torch.long))

                # target move encoding
                fs, ts, pr = encode_move_from_board(board, mv)
                move_from.append(torch.tensor(fs, dtype=torch.long))
                move_to.append(torch.tensor(ts, dtype=torch.long))
                move_promo.append(torch.tensor(pr, dtype=torch.long))

                # render image on-the-fly
                img = self.renderer.render(board, out_size=self.resolution)
                if self.augment is not None:
                    # augment expects numpy HWC uint8
                    np_img = np.asarray(img, dtype=np.uint8)
                    np_img = self.augment(np_img)
                    img = Image.fromarray(np_img)

                images.append(pil_to_tensor(img))

                if self.return_fen:
                    fens.append(board.fen())
                ply_indices.append(ply)

            board.push(mv)

        if len(images) == 0:
            return {"images": torch.empty(0,3,self.resolution,self.resolution), "valid": False}

        # next chosen move = move target at next sampled ply (not “next ply in game”)
        # last one gets -1
        for i in range(len(chosen)):
            if i + 1 < len(chosen):
                next_from.append(move_from[i + 1])
                next_to.append(move_to[i + 1])
                next_promo.append(move_promo[i + 1])
            else:
                next_from.append(torch.tensor(-1, dtype=torch.long))
                next_to.append(torch.tensor(-1, dtype=torch.long))
                next_promo.append(torch.tensor(-1, dtype=torch.long))

        out = {
            "valid": True,
            "game_idx": torch.tensor(idx, dtype=torch.long),
            "ply_idx": torch.tensor(ply_indices, dtype=torch.long),             # [M]
            "images": torch.stack(images, dim=0),                               # [M,3,H,W]
            "labels64": torch.stack(labels64, dim=0),                           # [M,64]
            "turn": torch.stack(meta_turn, dim=0),                              # [M]
            "castling": torch.stack(meta_castling, dim=0),                      # [M,4]
            "ep_square": torch.stack(meta_ep, dim=0),                           # [M]
            "move_from": torch.stack(move_from, dim=0),                         # [M]
            "move_to": torch.stack(move_to, dim=0),                             # [M]
            "move_promo": torch.stack(move_promo, dim=0),                       # [M]
            "next_move_from": torch.stack(next_from, dim=0),                    # [M]
            "next_move_to": torch.stack(next_to, dim=0),                        # [M]
            "next_move_promo": torch.stack(next_promo, dim=0),                  # [M]
        }
        if self.return_fen:
            out["fen"] = fens  # list[str], ragged; keep as python list
        return out