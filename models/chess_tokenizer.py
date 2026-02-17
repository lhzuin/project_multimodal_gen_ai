import os, json, tempfile, io
from typing import List, Tuple, Optional, Dict, Any

import chess
import chess.pgn

class ChessTokenizer:
    FILES = "abcdefgh"
    RANKS = "12345678"
    PIECE_TOKENS = ["P","N","B","R","Q","K"]

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SPECIALS = [PAD, BOS, EOS, UNK]

    def __init__(self, path: str):
        self.path = path
        self.SQUARES = [f + r for r in self.RANKS for f in self.FILES]

        if os.path.isfile(path):
            self._load(path)
        else:
            base_moves = self.build_minimal_physical_uci_vocab()
            self.id2move = self.SPECIALS + base_moves
            self.move2id = {m:i for i,m in enumerate(self.id2move)}

            self.id2piece = self.SPECIALS + self.PIECE_TOKENS
            self.piece2id = {p:i for i,p in enumerate(self.id2piece)}

            self._save(path)

        # cache ids
        self.pad_id = self.move2id[self.PAD]
        self.bos_id = self.move2id[self.BOS]
        self.eos_id = self.move2id[self.EOS]
        self.unk_id = self.move2id[self.UNK]

        self.p_pad_id = self.piece2id[self.PAD]
        self.p_bos_id = self.piece2id[self.BOS]
        self.p_eos_id = self.piece2id[self.EOS]
        self.p_unk_id = self.piece2id[self.UNK]

    # ---------- IO ----------
    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 3,
            "format": "uci_minimal_physical + piece_stream",
            "id2move": self.id2move,
            "id2piece": self.id2piece,
        }
        fd, tmp = tempfile.mkstemp(prefix="chesstok_", suffix=".json", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            try: os.remove(tmp)
            except OSError: pass
            raise

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.id2move = payload["id2move"]
        self.move2id = {m:i for i,m in enumerate(self.id2move)}

        self.id2piece = payload["id2piece"]
        self.piece2id = {p:i for i,p in enumerate(self.id2piece)}

        # sanity
        for tok in self.SPECIALS:
            if tok not in self.move2id or tok not in self.piece2id:
                raise ValueError(f"Tokenizer file missing special token: {tok}")

    # ---------- PGN -> (move_ids, piece_ids) ----------
    def encode_pgn_game(
        self,
        game: chess.pgn.Game,
        add_bos: bool = True,
        add_eos: bool = True,
        strict: bool = True,
    ) -> Tuple[List[int], List[int]]:
        board = game.board()
        move_ids: List[int] = []
        piece_ids: List[int] = []

        if add_bos:
            move_ids.append(self.bos_id)
            piece_ids.append(self.p_bos_id)

        for mv in game.mainline_moves():
            uci = mv.uci()

            # move token
            if uci in self.move2id:
                move_ids.append(self.move2id[uci])
            else:
                if strict: raise KeyError(f"UCI not in vocab: {uci}")
                move_ids.append(self.unk_id)

            # piece token (color is redundant; side-to-move is in history)
            piece = board.piece_at(mv.from_square)
            if piece is None:
                # should never happen in a valid game
                if strict: raise RuntimeError("No piece on from_square; PGN likely corrupt.")
                piece_ids.append(self.p_unk_id)
            else:
                p = piece.symbol().upper()  # "P","N","B","R","Q","K"
                piece_ids.append(self.piece2id.get(p, self.p_unk_id))

            board.push(mv)

        if add_eos:
            move_ids.append(self.eos_id)
            piece_ids.append(self.p_eos_id)

        return move_ids, piece_ids

    def encode_pgn_text(
        self,
        pgn_text: str,
        max_games: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        out = []
        f = io.StringIO(pgn_text)
        n = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            out.append(self.encode_pgn_game(game, **kwargs))
            n += 1
            if max_games is not None and n >= max_games:
                break
        return out

    # ---------- Your vocab generator (unchanged) ----------
    def build_minimal_physical_uci_vocab(self) -> List[str]:
        sq2xy = {s: (self.FILES.index(s[0]), self.RANKS.index(s[1])) for s in self.SQUARES}
        xy2sq = {(x, y): self.FILES[x] + self.RANKS[y] for x in range(8) for y in range(8)}
        def inb(x, y): return 0 <= x < 8 and 0 <= y < 8

        moves = set()
        KN = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
        KG = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        ROOK = [(1,0),(-1,0),(0,1),(0,-1)]
        BISH = [(1,1),(1,-1),(-1,1),(-1,-1)]

        def add_slides(frm, dirs):
            x, y = sq2xy[frm]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                while inb(nx, ny):
                    moves.add(frm + xy2sq[(nx, ny)])
                    nx += dx
                    ny += dy

        for frm in self.SQUARES:
            x, y = sq2xy[frm]
            for dx, dy in KN:
                nx, ny = x + dx, y + dy
                if inb(nx, ny): moves.add(frm + xy2sq[(nx, ny)])
            for dx, dy in KG:
                nx, ny = x + dx, y + dy
                if inb(nx, ny): moves.add(frm + xy2sq[(nx, ny)])
            add_slides(frm, ROOK)
            add_slides(frm, BISH)

        PROMO = "qrbn"
        for frm in self.SQUARES:
            x, y = sq2xy[frm]

            # white pawn
            if inb(x, y + 1):
                to = xy2sq[(x, y + 1)]
                if y == 6:
                    for p in PROMO: moves.add(frm + to + p)
                else:
                    moves.add(frm + to)
            if y == 1 and inb(x, y + 2):
                moves.add(frm + xy2sq[(x, y + 2)])
            for dx in (-1, 1):
                nx, ny = x + dx, y + 1
                if inb(nx, ny):
                    to = xy2sq[(nx, ny)]
                    if y == 6:
                        for p in PROMO: moves.add(frm + to + p)
                    else:
                        moves.add(frm + to)

            # black pawn
            if inb(x, y - 1):
                to = xy2sq[(x, y - 1)]
                if y == 1:
                    for p in PROMO: moves.add(frm + to + p)
                else:
                    moves.add(frm + to)
            if y == 6 and inb(x, y - 2):
                moves.add(frm + xy2sq[(x, y - 2)])
            for dx in (-1, 1):
                nx, ny = x + dx, y - 1
                if inb(nx, ny):
                    to = xy2sq[(nx, ny)]
                    if y == 1:
                        for p in PROMO: moves.add(frm + to + p)
                    else:
                        moves.add(frm + to)

        moves.update(["e1g1", "e1c1", "e8g8", "e8c8"])
        return sorted(moves)

