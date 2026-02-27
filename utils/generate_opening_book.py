import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
import chess
import chess.pgn


def normalize_str(x: Optional[str]) -> str:
    return (x or "").strip()


def variation_key(headers: Dict[str, str]) -> str:
    """
    Use a stable key to 'jump games' by opening family.
    If Variation is empty, fall back to Opening or ECO.
    """
    eco = normalize_str(headers.get("ECO"))
    opening = normalize_str(headers.get("Opening"))
    var = normalize_str(headers.get("Variation"))

    if var:
        return f"{eco}|{opening}|{var}"
    if opening:
        return f"{eco}|{opening}"
    if eco:
        return eco
    return "UNKNOWN"


def extract_prefix_and_fen(
    game: chess.pgn.Game,
    max_full_moves: int = 6,
) -> Tuple[List[str], List[str], str]:
    """
    Returns:
      - san_prefix: list of SAN moves (length <= 2*max_full_moves)
      - uci_prefix: list of UCI moves (same length)
      - fen_after: FEN after applying prefix
    """
    board = game.board()
    san_prefix: List[str] = []
    uci_prefix: List[str] = []

    max_plies = 2 * max_full_moves
    ply = 0

    node = game
    while node.variations and ply < max_plies:
        node = node.variation(0)
        move = node.move
        if move is None:
            break

        # SAN must be computed BEFORE pushing
        san_prefix.append(board.san(move))
        uci_prefix.append(move.uci())
        board.push(move)
        ply += 1

    fen_after = board.fen()
    return san_prefix, uci_prefix, fen_after


def build_opening_suite_from_pgn(
    pgn_path: str,
    max_full_moves: int = 6,
    one_entry_per_variation_key: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parse PGN and build a deduplicated opening suite.

    Dedup strategy:
      - Primary grouping: variation_key (ECO|Opening|Variation)
      - Within group: dedup by (uci_prefix tuple, fen_after)
    If one_entry_per_variation_key=True, we keep only the first unique entry per key.
    Otherwise, we keep multiple entries per key (still deduped).
    """
    suite: List[Dict[str, Any]] = []

    # Track duplicates
    seen_within_key: Dict[str, set] = {}
    used_keys: set = set()

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_idx += 1

            headers = dict(game.headers)

            key = variation_key(headers)
            if one_entry_per_variation_key and key in used_keys:
                continue

            try:
                san_prefix, uci_prefix, fen_after = extract_prefix_and_fen(
                    game, max_full_moves=max_full_moves
                )
            except Exception:
                # Skip broken games/moves
                continue

            # Ignore games that are too short to supply the prefix if you want strictness
            # (optional) If you want exactly 12 plies, uncomment:
            # if len(uci_prefix) < 2 * max_full_moves:
            #     continue

            sig = (tuple(uci_prefix), fen_after)

            if key not in seen_within_key:
                seen_within_key[key] = set()
            if sig in seen_within_key[key]:
                continue

            seen_within_key[key].add(sig)
            used_keys.add(key)

            entry = {
                "key": key,  # stable identifier for lookups
                "eco": normalize_str(headers.get("ECO")),
                "opening": normalize_str(headers.get("Opening")),
                "variation": normalize_str(headers.get("Variation")),
                "event": normalize_str(headers.get("Event")),
                "site": normalize_str(headers.get("Site")),
                "date": normalize_str(headers.get("Date")),
                "white": normalize_str(headers.get("White")),
                "black": normalize_str(headers.get("Black")),
                "result": normalize_str(headers.get("Result")),
                "prefix_full_moves": max_full_moves,
                "san_prefix": san_prefix,
                "uci_prefix": uci_prefix,
                "fen_after_prefix": fen_after,
            }
            suite.append(entry)

    return suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to PGN file")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--full_moves", type=int, default=6, help="How many full moves to include (default 6)")
    ap.add_argument(
        "--one_per_key",
        action="store_true",
        help="Keep only one entry per (ECO|Opening|Variation) key",
    )
    args = ap.parse_args()

    suite = build_opening_suite_from_pgn(
        pgn_path=args.pgn,
        max_full_moves=args.full_moves,
        one_entry_per_variation_key=args.one_per_key,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(suite)} unique openings to {args.out}")


if __name__ == "__main__":
    main()