"""Streaming statistics helpers for PGN corpora."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


@dataclass(frozen=True)
class PgnPlyStats:
    """Statistics collected while scanning a PGN corpus."""

    total_plies: int
    valid_games: int
    skipped_games: int


class _PlyCountVisitor(chess.pgn.BaseVisitor):
    """Count mainline plies while fully parsing PGN variations."""

    def __init__(self) -> None:
        self.plies = 0
        self.variation_depth = 0
        self.has_error = False

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        """Count moves only when they belong to the mainline."""
        if self.variation_depth == 0:
            self.plies += 1

    def begin_variation(self) -> None:
        """Enter a side variation while continuing to parse it."""
        self.variation_depth += 1

    def end_variation(self) -> None:
        """Leave a side variation."""
        self.variation_depth -= 1

    def handle_error(self, error: Exception) -> None:
        """Mark the entire game as malformed without aborting the corpus scan."""
        self.has_error = True

    def result(self) -> tuple[int, bool]:
        return self.plies, self.has_error


def analyze_pgn_plies(pgn_path: str | Path) -> PgnPlyStats:
    """Count mainline plies, completely excluding malformed games.

    Variations are fully parsed and validated, but their moves are not included
    in the ply count.

    If a parsing error occurs anywhere in a game, including inside a variation,
    the entire game is discarded and contributes zero plies.

    The PGN is processed one game at a time without constructing a complete
    game tree, keeping memory usage low.
    """
    path = Path(pgn_path)

    total_plies = 0
    valid_games = 0
    skipped_games = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        buffering=1024 * 1024,
    ) as pgn_stream:
        while True:
            result = chess.pgn.read_game(
                pgn_stream,
                Visitor=_PlyCountVisitor,
            )

            if result is None:
                break

            game_plies, has_error = result

            if has_error:
                # Discard the WHOLE game, including valid moves parsed before
                # the malformed move.
                skipped_games += 1
                continue

            total_plies += game_plies
            valid_games += 1

    return PgnPlyStats(
        total_plies=total_plies,
        valid_games=valid_games,
        skipped_games=skipped_games,
    )


def count_pgn_plies(pgn_path: str | Path) -> int:
    """Return total mainline plies from valid games."""
    return analyze_pgn_plies(pgn_path).total_plies


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count mainline plies in a PGN file while ignoring malformed games."
        )
    )
    parser.add_argument(
        "pgn_path",
        help="Path to the PGN file to analyze.",
    )
    args = parser.parse_args()

    stats = analyze_pgn_plies(args.pgn_path)

    print(f"Total recorded mainline plies: {stats.total_plies:,}")
    print(f"Valid games:                  {stats.valid_games:,}")
    print(f"Malformed games skipped:      {stats.skipped_games:,}")


if __name__ == "__main__":
    main()