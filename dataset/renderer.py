import os
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
    """
    def __init__(
        self,
        sprites_dir: str,
        square_px: int = 64,
        light_rgb=(240, 217, 181),
        dark_rgb=(181, 136, 99),
    ):
        self.square_px = int(square_px)
        self.light_rgb = tuple(light_rgb)
        self.dark_rgb = tuple(dark_rgb)

        self.sprites = {}

        for sym, fname in FILE_FOR_SYMBOL.items():
            path = os.path.join(sprites_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing sprite: {path}")
            img = Image.open(path).convert("RGBA")
            self.sprites[sym] = img

    def render(self, board: chess.Board, out_size: int) -> Image.Image:
        """
        Render board to a square PIL image of size (out_size, out_size).
        """
        sq = self.square_px
        board_img = Image.new("RGBA", (8*sq, 8*sq), (0,0,0,0))

        # draw squares
        for r in range(8):
            for c in range(8):
                is_light = ((r + c) % 2 == 0)
                color = self.light_rgb if is_light else self.dark_rgb
                tile = Image.new("RGBA", (sq, sq), color + (255,))
                board_img.paste(tile, (c*sq, r*sq))

        # place pieces (r=0 is top = rank 8)
        for rank in range(7, -1, -1):
            for file in range(8):
                piece = board.piece_at(chess.square(file, rank))
                if piece is None:
                    continue
                sym = piece.symbol()  # "P" or "p", etc
                spr = self.sprites[sym]

                # resize sprite to fit square (keep aspect)
                spr_resized = spr.resize((sq, sq), resample=Image.Resampling.LANCZOS)

                r = 7 - rank
                c = file
                board_img.alpha_composite(spr_resized, (c*sq, r*sq))

        # final resize
        if out_size != 8*sq:
            board_img = board_img.resize((out_size, out_size), resample=Image.Resampling.LANCZOS)

        return board_img.convert("RGB")