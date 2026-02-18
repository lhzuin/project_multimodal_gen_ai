import os
import random
import numpy as np
from PIL import Image
import chess

PIECE_SYMBOLS = ["P","N","B","R","Q","K","p","n","b","r","q","k"]

FILE_FOR_SYMBOL = {
    "P": "wP.png", "N": "wN.png", "B": "wB.png", "R": "wR.png", "Q": "wQ.png", "K": "wK.png",
    "p": "bP.png", "n": "bN.png", "b": "bB.png", "r": "bR.png", "q": "bQ.png", "k": "bK.png",
}


PIECE_TO_ID = {
    None: 0,
    chess.Piece.from_symbol("P"): 1, chess.Piece.from_symbol("N"): 2, chess.Piece.from_symbol("B"): 3,
    chess.Piece.from_symbol("R"): 4, chess.Piece.from_symbol("Q"): 5, chess.Piece.from_symbol("K"): 6,
    chess.Piece.from_symbol("p"): 7, chess.Piece.from_symbol("n"): 8, chess.Piece.from_symbol("b"): 9,
    chess.Piece.from_symbol("r"): 10, chess.Piece.from_symbol("q"): 11, chess.Piece.from_symbol("k"): 12,
}

ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

# Color pairs for board rendering (light_rgb, dark_rgb)
COLOR_PAIRS = [
    ((240, 217, 181), (181, 136, 99)),      # Original tan/brown
    ((235, 236, 208), (119, 149, 83)),      # Green
    ((240, 217, 181), (100, 150, 200)),     # Blue
    ((220, 220, 220), (100, 100, 100)),     # Gray
    ((230, 200, 240), (150, 100, 150)),     # Purple
]

def board_to_grid_ids(board: chess.Board) -> np.ndarray:
    """Return (8,8) IDs from rank 8->1, file a->h."""
    grid = np.zeros((8,8), dtype=np.int64)
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = chess.square(file, rank)
            grid[7-rank, file] = PIECE_TO_ID[board.piece_at(sq)]
    return grid

def encode_move_from_board(board: chess.Board, move: chess.Move):
    """
    Encode target move as (from_sq, to_sq, promo_id).
    promo_id: 0 none, 1 N,2 B,3 R,4 Q (consistent with common ordering)
    """
    promo = move.promotion
    promo_id = 0
    if promo is not None:
        if promo == chess.KNIGHT: promo_id = 1
        elif promo == chess.BISHOP: promo_id = 2
        elif promo == chess.ROOK: promo_id = 3
        elif promo == chess.QUEEN: promo_id = 4
        else: promo_id = 0
    return move.from_square, move.to_square, promo_id

class SpriteBoardRenderer:
    """
    Render a top-down chessboard image using piece sprites.
    Supports multiple sprite styles in subdirectories.
    """
    def __init__(
        self,
        sprites_dir: str,
        square_px: int = 64,
    ):
        self.square_px = int(square_px)
        self.color_pairs = COLOR_PAIRS
        
        # Load sprite styles from subdirectories
        self.sprite_styles = {}
        
        # Check if sprites_dir itself contains piece files (legacy single style)
        legacy_style_path = sprites_dir
        if self._has_piece_files(legacy_style_path):
            self.sprite_styles["default"] = self._load_sprites_from_dir(legacy_style_path)
        
        # Load sprite styles from subdirectories
        if os.path.isdir(sprites_dir):
            for item in os.listdir(sprites_dir):
                item_path = os.path.join(sprites_dir, item)
                if os.path.isdir(item_path) and self._has_piece_files(item_path):
                    self.sprite_styles[item] = self._load_sprites_from_dir(item_path)
        
        if not self.sprite_styles:
            raise FileNotFoundError(
                f"No sprite styles found in {sprites_dir}. "
                f"Organize sprites as: {sprites_dir}/style1/ with piece files, "
                f"or place piece files directly in {sprites_dir}/"
            )
    
    def _has_piece_files(self, directory: str) -> bool:
        """Check if directory contains all required piece files."""
        if not os.path.isdir(directory):
            return False
        for sym in PIECE_SYMBOLS:
            fname = FILE_FOR_SYMBOL[sym]
            if not os.path.exists(os.path.join(directory, fname)):
                return False
        return True
    
    def _load_sprites_from_dir(self, directory: str) -> dict:
        """Load all sprite images from a directory."""
        sprites = {}
        for sym, fname in FILE_FOR_SYMBOL.items():
            path = os.path.join(directory, fname)
            img = Image.open(path).convert("RGBA")
            sprites[sym] = img
        return sprites

    def render(self, board: chess.Board, out_size: int) -> Image.Image:
        """
        Render board to a square PIL image of size (out_size, out_size).
        Randomly selects color pair and sprite style.
        """
        # Randomly select color pair and sprite style
        light_rgb, dark_rgb = random.choice(self.color_pairs)
        sprites = random.choice(list(self.sprite_styles.values()))
        
        sq = self.square_px
        board_img = Image.new("RGBA", (8*sq, 8*sq), (0,0,0,0))

        # draw squares
        for r in range(8):
            for c in range(8):
                is_light = ((r + c) % 2 == 0)
                color = light_rgb if is_light else dark_rgb
                tile = Image.new("RGBA", (sq, sq), color + (255,))
                board_img.paste(tile, (c*sq, r*sq))

        # place pieces (r=0 is top = rank 8)
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                if piece is None:
                    continue
                sym = piece.symbol()  # "P" or "p", etc
                spr = sprites[sym]

                # resize sprite to fit square (keep aspect)
                spr_resized = spr.resize((sq, sq), resample=Image.Resampling.LANCZOS)

                r = 7 - rank
                c = file
                board_img.alpha_composite(spr_resized, (c*sq, r*sq))

        # final resize
        if out_size != 8*sq:
            board_img = board_img.resize((out_size, out_size), resample=Image.Resampling.LANCZOS)

        return board_img.convert("RGB")