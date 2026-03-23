import io
import os
import json
import random
from dataclasses import dataclass
from tracemalloc import start
from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import chess
import chess.pgn
from PIL import Image

from .offsets import load_pgn_offsets, ensure_offsets
from .renderer import SpriteBoardRenderer, board_to_grid_ids, encode_move_from_board
from .augmentations import make_albu_augment


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3,H,W] in [0,1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))               # [3,H,W]
    return torch.from_numpy(arr)


def _parse_game_bytes(game_bytes: bytes) -> Optional[chess.pgn.Game]:
    txt = game_bytes.decode("utf-8", errors="ignore")
    pgn_io = io.StringIO(txt)
    try:
        return chess.pgn.read_game(pgn_io)
    except Exception:
        return None


def _default_samples_path(pgn_path: str, sample_ratio: float, max_positions_per_game: int, seed: int) -> str:
    # Make it stable & filesystem-friendly
    base = os.path.splitext(os.path.basename(pgn_path))[0]
    ratio_str = f"{sample_ratio:.4f}".rstrip("0").rstrip(".")  # e.g. 0.1, 0.025, 0.3333
    return f"{base}_samples_s{seed}_r{ratio_str}_m{max_positions_per_game}.json"


def build_samples_file(
    *,
    pgn_path: str,
    index_path: str,
    out_samples_path: str,
    sample_ratio: float,
    max_positions_per_game: int,
    seed: int,
) -> None:
    """
    One-time builder: parse each game once and sample ply indices.
    Writes JSON payload with "samples": [[game_list_idx, ply_idx], ...]
    """
    ensure_offsets(pgn_path, index_path)
    idx = load_pgn_offsets(index_path)
    games = idx["games"]

    rng = random.Random(seed)
    samples: List[Tuple[int, int]] = []

    with open(pgn_path, "rb") as fh:
        for game_list_idx, g in enumerate(games):
            fh.seek(g["start"])
            game_bytes = fh.read(g["end"] - g["start"])
            game = _parse_game_bytes(game_bytes)
            if game is None:
                continue

            moves = list(game.mainline_moves())
            n = len(moves)
            if n <= 0:
                continue

            k = max(1, int(round(sample_ratio * n)))
            k = min(k, n, max_positions_per_game)

            chosen = rng.sample(range(n), k=k)
            for ply in chosen:
                samples.append((game_list_idx, ply))

    rng.shuffle(samples)

    payload = {
        "pgn_path": os.path.abspath(pgn_path),
        "index_path": os.path.abspath(index_path),
        "sample_ratio": sample_ratio,
        "max_positions_per_game": max_positions_per_game,
        "seed": seed,
        "num_samples": len(samples),
        "samples": [[gi, pi] for (gi, pi) in samples],
    }

    with open(out_samples_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[build_samples] wrote {len(samples)} samples -> {out_samples_path}")


class ChessGameSampleDataset(Dataset):
    """
    Position-level dataset:
      __getitem__(i) returns exactly one position sample (image + labels + metadata + next move).

    It uses a precomputed samples file containing (game_list_idx, ply_idx) pairs.
    If samples file doesn't exist, it is built automatically (one-time cost).
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
        apply_augmentations: bool = False,
        return_fen: bool = False,
        samples_path: Optional[str] = None,
        cache_size: int = 64,   # LRU of move-lists per worker
        return_images: bool = True,
        return_piece_probs: bool = True,
    ):
        self.pgn_path = pgn_path
        self.index_path = index_path
        self.resolution = int(resolution)
        self.sample_ratio = float(sample_ratio)
        self.max_positions_per_game = int(max_positions_per_game)
        self.seed = int(seed)
        self.return_fen = bool(return_fen)
        self.return_images = bool(return_images)
        self.return_piece_probs = bool(return_piece_probs)

        # Augmentation is applied only to rendered image
        self.augment = make_albu_augment(self.resolution) if apply_augmentations else None

        # Renderer
        square_px = max(24, self.resolution // 8)
        self.renderer = SpriteBoardRenderer(sprites_dir=sprites_dir, square_px=square_px)

        # Load offsets index (small JSON; safe in __init__)
        self.index = load_pgn_offsets(self.index_path)
        self.games = self.index["games"]

        # Samples file path
        self.samples_path = samples_path or _default_samples_path(
            pgn_path=self.pgn_path,
            sample_ratio=self.sample_ratio,
            max_positions_per_game=self.max_positions_per_game,
            seed=self.seed,
        )

        # Ensure samples file exists (build once)
        if not os.path.exists(self.samples_path):
            build_samples_file(
                pgn_path=self.pgn_path,
                index_path=self.index_path,
                out_samples_path=self.samples_path,
                sample_ratio=self.sample_ratio,
                max_positions_per_game=self.max_positions_per_game,
                seed=self.seed,
            )

        self.samples = self._load_samples(self.samples_path)

        # Per-worker state (must be picklable at init)
        self._fh = None  # opened lazily inside each worker
        self.cache_size = int(cache_size)
        self._cache: Optional[OrderedDict[int, Optional[Tuple[str, List[chess.Move]]]]] = None

    def __getstate__(self):
        """
        Make dataset spawn-safe (macOS/Windows): drop non-picklable runtime state.
        """
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_cache"] = None
        return state

    def _load_samples(self, path: str) -> List[Tuple[int, int]]:
        def read_payload():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        payload = read_payload()
        if os.path.abspath(payload.get("pgn_path", "")) != os.path.abspath(self.pgn_path) or \
        os.path.abspath(payload.get("index_path", "")) != os.path.abspath(self.index_path):
            build_samples_file(
                pgn_path=self.pgn_path,
                index_path=self.index_path,
                out_samples_path=self.samples_path,
                sample_ratio=self.sample_ratio,
                max_positions_per_game=self.max_positions_per_game,
                seed=self.seed,
            )
            payload = read_payload()  # IMPORTANT: reload

        return [(int(a), int(b)) for a, b in payload["samples"]]

    def __len__(self) -> int:
        return len(self.samples)

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.pgn_path, "rb")
        return self._fh

    def _read_game_bytes(self, start: int, end: int) -> bytes:
        fh = self._get_fh()
        fh.seek(start)
        return fh.read(end - start)

    def _get_cache(self):
        if self.cache_size <= 0:
            return None
        if self._cache is None:
            self._cache = OrderedDict()
        return self._cache

    def _load_moves_for_game(self, game_list_idx: int) -> Optional[Tuple[str, List[chess.Move]]]:
        """
        Cached loader for (start_fen, moves) of one game.
        start_fen respects SetUp/FEN tags.
        """
        cache = self._get_cache()
        if cache is not None and game_list_idx in cache:
            value = cache.pop(game_list_idx)
            cache[game_list_idx] = value
            return value  # can be None or (start_fen, moves)

        g = self.games[game_list_idx]
        game_bytes = self._read_game_bytes(g["start"], g["end"])
        game = _parse_game_bytes(game_bytes)

        if game is None:
            value = None
        else:
            start_fen = game.board().fen()
            moves = list(game.mainline_moves())
            value = (start_fen, moves)

        if cache is not None:
            cache[game_list_idx] = value
            if len(cache) > self.cache_size:
                cache.popitem(last=False)

        return value

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game_list_idx, target_ply = self.samples[idx]
        g = self.games[game_list_idx]

        loaded = self._load_moves_for_game(game_list_idx)
        if loaded is None:
            print("Warning: invalid move")
            return {"valid": False}
        start_fen, moves = loaded
        board = chess.Board(start_fen)

        n = len(moves)
        if target_ply < 0 or target_ply >= n:
            print("Warning: invalid move")
            return {"valid": False}

        try:
            for ply in range(target_ply):
                board.push(moves[ply])
        except AssertionError:
            print("Warning: invalid move")
            return {"valid": False}

        mv = moves[target_ply]

        # labels + metadata from current board
        grid = board_to_grid_ids(board)
        labels64 = torch.from_numpy(grid.reshape(-1)).long()

        piece_probs = None
        if self.return_piece_probs:
            # one-hot: [64,13] float
            piece_probs = torch.nn.functional.one_hot(labels64, num_classes=13).float()

        turn = torch.tensor(1 if board.turn else 0, dtype=torch.long)
        ck  = int(board.has_kingside_castling_rights(chess.WHITE))
        cq  = int(board.has_queenside_castling_rights(chess.WHITE))
        ck2 = int(board.has_kingside_castling_rights(chess.BLACK))
        cq2 = int(board.has_queenside_castling_rights(chess.BLACK))
        castling = torch.tensor([ck, cq, ck2, cq2], dtype=torch.long)
        ep_square = torch.tensor(board.ep_square if board.ep_square is not None else -1, dtype=torch.long)

        # current move encoding
        fs, ts, pr = encode_move_from_board(board, mv)

        move_from = torch.tensor(fs, dtype=torch.long)
        move_to = torch.tensor(ts, dtype=torch.long)
        move_promo = torch.tensor(pr, dtype=torch.long)

        # render current board
        image = None
        if self.return_images:
            img = self.renderer.render(board, out_size=self.resolution)
            if self.augment is not None:
                np_img = np.asarray(img, dtype=np.uint8)
                np_img = self.augment(np_img)
                img = Image.fromarray(np_img)
            image = pil_to_tensor(img)  # [3,H,W]

        # next move = next ply in game (or -1)
        if target_ply + 1 < n:
            board_after = board.copy(stack=False)
            board_after.push(mv)
            mv2 = moves[target_ply + 1]
            nfs, nts, npr = encode_move_from_board(board_after, mv2)
            next_move_from = torch.tensor(nfs, dtype=torch.long)
            next_move_to = torch.tensor(nts, dtype=torch.long)
            next_move_promo = torch.tensor(npr, dtype=torch.long)
        else:
            next_move_from = torch.tensor(-1, dtype=torch.long)
            next_move_to = torch.tensor(-1, dtype=torch.long)
            next_move_promo = torch.tensor(-1, dtype=torch.long)
            
        
        board_check = chess.Board(board.fen())
        legal_pairs = {(m.from_square, m.to_square) for m in board_check.legal_moves}
        if (int(move_from.item()), int(move_to.item())) not in legal_pairs:
            print("Warning: invalid move")
            return {"valid": False}

        out = {
            "valid": True,
            "game_idx": torch.tensor(g["idx"], dtype=torch.long),
            "ply_idx": torch.tensor(target_ply, dtype=torch.long),
            "labels64": labels64,  # [64]

            "turn": turn,
            "castling": castling,
            "ep_square": ep_square,

            "move_from": move_from,
            "move_to": move_to,
            "move_promo": move_promo,

            "next_move_from": next_move_from,
            "next_move_to": next_move_to,
            "next_move_promo": next_move_promo,
        }

        if self.return_fen:
            out["fen"] = board.fen()

        if self.return_images:
            out["images"] = image
        if self.return_piece_probs:
            out["piece_probs"] = piece_probs

        return out
    


class ChessGameSequenceDataset(Dataset):
    """
    Game-level sequence dataset:
      __getitem__(i) returns a sequence of positions from ONE game.

    Each timestep t contains:
      - board state BEFORE playing move t
      - target = move t (from,to,promo)

    We optionally take a random contiguous window of length <= max_seq_len to
    keep memory bounded.
    """
    def __init__(
        self,
        pgn_path: str,
        index_path: str,
        sprites_dir: str,          # kept for API consistency (not used if return_images=False)
        resolution: int = 256,     # kept for API consistency
        seed: int = 0,
        max_seq_len: int = 64,
        random_window: bool = True,
        return_fen: bool = True,
        return_images: bool = False,
        return_piece_probs: bool = True,
        cache_size: int = 64,
    ):
        self.pgn_path = pgn_path
        self.index_path = index_path
        self.seed = int(seed)
        self.max_seq_len = int(max_seq_len)
        self.random_window = bool(random_window)

        self.return_fen = bool(return_fen)
        self.return_images = bool(return_images)  # supported but default False (much faster)
        self.return_piece_probs = bool(return_piece_probs)

        # Load offsets
        ensure_offsets(pgn_path, index_path)
        self.index = load_pgn_offsets(index_path)
        self.games = self.index["games"]

        # Runtime state per worker
        self._fh = None
        self.cache_size = int(cache_size)
        self._cache: Optional[OrderedDict[int, Optional[Tuple[str, List[chess.Move]]]]] = None

        # NOTE: We do NOT precompute samples here. Dataset length = number of games.
        # You can filter very short games if you want; for now keep all.
        self.game_list_indices = list(range(len(self.games)))

        self.return_legal = True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self.game_list_indices)

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.pgn_path, "rb")
        return self._fh

    def _read_game_bytes(self, start: int, end: int) -> bytes:
        fh = self._get_fh()
        fh.seek(start)
        return fh.read(end - start)

    def _get_cache(self):
        if self.cache_size <= 0:
            return None
        if self._cache is None:
            self._cache = OrderedDict()
        return self._cache

    def _load_moves_for_game(self, game_list_idx: int) -> Optional[Tuple[str, List[chess.Move]]]:
        cache = self._get_cache()
        if cache is not None and game_list_idx in cache:
            value = cache.pop(game_list_idx)
            cache[game_list_idx] = value
            return value

        g = self.games[game_list_idx]
        game_bytes = self._read_game_bytes(g["start"], g["end"])
        game = _parse_game_bytes(game_bytes)

        if game is None:
            value = None
        else:
            start_fen = game.board().fen()
            moves = list(game.mainline_moves())
            value = (start_fen, moves)

        if cache is not None:
            cache[game_list_idx] = value
            if len(cache) > self.cache_size:
                cache.popitem(last=False)

        return value

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        game_list_idx = self.game_list_indices[idx]
        g = self.games[game_list_idx]

        loaded = self._load_moves_for_game(game_list_idx)
        if loaded is None:
            print("Warning: invalid move (parse failed)")
            return {"valid": False}

        start_fen, moves = loaded
        n = len(moves)
        if n <= 0:
            #print("Warning: invalid move (no moves in game)")
            return {"valid": False}

        # Choose a contiguous window
        L = min(self.max_seq_len, n)
        if self.random_window and n > L:
            # per-call randomness is OK; workers each have their own RNG stream
            start = random.randint(0, n - L)
        else:
            start = 0

        end = start + L

        board = chess.Board(start_fen)
        # advance to window start
        try:
            for ply in range(start):
                board.push(moves[ply])
        except AssertionError:
            print("Warning: invalid move (push to start failed)")
            return {"valid": False}

        # Build sequence tensors
        turn_seq = []
        castling_seq = []
        ep_seq = []
        move_from_seq = []
        move_to_seq = []
        move_promo_seq = []
        labels64_seq = []

        legal_flat_seq = []

        for ply in range(start, end):
            mv = moves[ply]

            # --- state BEFORE move ---
            grid = board_to_grid_ids(board)
            labels64 = torch.from_numpy(grid.reshape(-1)).long()   # [64]
            labels64_seq.append(labels64)

            # metadata
            turn_seq.append(torch.tensor(1 if board.turn else 0, dtype=torch.long))

            ck  = int(board.has_kingside_castling_rights(chess.WHITE))
            cq  = int(board.has_queenside_castling_rights(chess.WHITE))
            ck2 = int(board.has_kingside_castling_rights(chess.BLACK))
            cq2 = int(board.has_queenside_castling_rights(chess.BLACK))
            castling_seq.append(torch.tensor([ck, cq, ck2, cq2], dtype=torch.long))

            ep_seq.append(torch.tensor(board.ep_square if board.ep_square is not None else -1, dtype=torch.long))

            # NEW: legal mask for this board state as flat 4096
            legal = torch.zeros(4096, dtype=torch.bool)
            for m in board.legal_moves:
                legal[m.from_square * 64 + m.to_square] = True
            legal_flat_seq.append(legal)
            

            # --- target move encoding ---
            fs, ts, pr = encode_move_from_board(board, mv)
    
            if fs < 0 or ts < 0:
                print("Warning: invalid move (encode failed)")
                return {"valid": False}
            move_from_seq.append(torch.tensor(fs, dtype=torch.long))
            move_to_seq.append(torch.tensor(ts, dtype=torch.long))
            move_promo_seq.append(torch.tensor(pr, dtype=torch.long))

            # advance
            try:
                board.push(mv)
            except AssertionError:
                print("Warning: invalid move (push failed)")
                return {"valid": False}

        out = {
            "valid": True,
            "game_idx": torch.tensor(g["idx"], dtype=torch.long),
            "seq_len": torch.tensor(L, dtype=torch.long),
            "move_from": torch.stack(move_from_seq, dim=0),   # [T]
            "move_to": torch.stack(move_to_seq, dim=0),       # [T]
            "move_promo": torch.stack(move_promo_seq, dim=0), # [T]
            "turn": torch.stack(turn_seq, dim=0),             # [T]
            "castling": torch.stack(castling_seq, dim=0),     # [T,4]
            "ep_square": torch.stack(ep_seq, dim=0),          # [T]
        }


        out["labels64"] = torch.stack(labels64_seq, dim=0)  # [T,64]
        out["legal_flat"] = torch.stack(legal_flat_seq, dim=0)  # [T,4096] bool

        # Images sequence is intentionally not implemented here (too slow / big).
        # If you really need it, we can add it later.

        return out