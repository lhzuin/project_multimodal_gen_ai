import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os

@dataclass
class GameOffset:
    start: int
    end: int   # exclusive
    idx: int

def build_pgn_offsets(pgn_path: str, out_index_path: str) -> List[GameOffset]:
    """
    Build random-access index for an uncompressed .pgn by recording byte offsets of each game.

    Heuristic: a new game starts at a line beginning with b'[Event '.
    This is robust for standard PGNs (Lichess, TWIC, etc.).

    Writes JSON with keys: {"pgn_path":..., "games":[{"idx":..,"start":..,"end":..}, ...]}
    """
    starts: List[int] = []
    with open(pgn_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b"[Event "):
                starts.append(pos)

        file_end = f.tell()

    games: List[GameOffset] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else file_end
        games.append(GameOffset(start=s, end=e, idx=i))


    payload = {
        "pgn_path": os.path.abspath(pgn_path),
        "pgn_size": os.path.getsize(pgn_path),
        "pgn_mtime": os.path.getmtime(pgn_path),
        "num_games": len(games),
        "games": [{"idx": g.idx, "start": g.start, "end": g.end} for g in games],
    }
    with open(out_index_path, "w", encoding="utf-8") as w:
        json.dump(payload, w)

    return games

def load_pgn_offsets(index_path: str) -> Dict[str, Any]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)
    



def ensure_offsets(pgn_path: str, index_path: str):
    if not os.path.exists(index_path):
        build_pgn_offsets(pgn_path, index_path)
        return

    try:
        idx = load_pgn_offsets(index_path)
        ok = (
            os.path.abspath(pgn_path) == idx.get("pgn_path")
            and os.path.getsize(pgn_path) == idx.get("pgn_size")
            and os.path.getmtime(pgn_path) == idx.get("pgn_mtime")
        )
    except Exception:
        ok = False

    if not ok:
        build_pgn_offsets(pgn_path, index_path)