#!/usr/bin/env python3

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn
import matplotlib.pyplot as plt


def board_key(board: chess.Board, mode: str = "state") -> str:
    """
    Normalized representation of a chess board.

    mode = "pieces":
        Only piece placement.
        Ignores side to move, castling rights, and en passant.

    mode = "state":
        Piece placement + side to move + castling rights + en passant square.
        Recommended.

    mode = "full_fen":
        Full FEN, including halfmove and fullmove counters.
        Usually too strict.
    """
    if mode == "pieces":
        return board.board_fen()

    if mode == "state":
        fen_parts = board.fen().split()
        return " ".join(fen_parts[:4])

    if mode == "full_fen":
        return board.fen()

    raise ValueError(f"Unknown key mode: {mode}")


def read_positions_by_ply(
    pgn_path: Path,
    max_games: int | None,
    max_ply: int | None,
    key_mode: str,
    include_initial_position: bool,
):
    """
    First pass through the PGN.

    Returns:
        positions_by_ply:
            ply -> list of board-state keys observed at that ply.

        global_counter:
            board-state key -> number of times this state appears anywhere
            in the dataset, regardless of ply.

        num_games:
            number of games read.
    """

    positions_by_ply = defaultdict(list)
    global_counter = Counter()

    num_games = 0

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)

            if game is None:
                break

            num_games += 1

            if max_games is not None and num_games > max_games:
                num_games -= 1
                break

            board = game.board()

            if include_initial_position:
                key = board_key(board, key_mode)
                positions_by_ply[0].append(key)
                global_counter[key] += 1

            ply = 0

            for move in game.mainline_moves():
                board.push(move)
                ply += 1

                if max_ply is not None and ply > max_ply:
                    break

                key = board_key(board, key_mode)
                positions_by_ply[ply].append(key)
                global_counter[key] += 1

            if num_games % 1000 == 0:
                print(f"Processed {num_games} games...")

    return positions_by_ply, global_counter, num_games


def build_rows(positions_by_ply, global_counter):
    rows = []

    for ply in sorted(positions_by_ply.keys()):
        states = positions_by_ply[ply]

        total = len(states)

        unique_at_same_ply = len(set(states))
        repeated_at_same_ply = total - unique_at_same_ply

        globally_repeated_count = sum(
            1 for state in states
            if global_counter[state] >= 2
        )

        globally_unique_count = total - globally_repeated_count

        repeated_same_ply_pct = (
            100.0 * repeated_at_same_ply / total
            if total > 0
            else 0.0
        )

        globally_repeated_pct = (
            100.0 * globally_repeated_count / total
            if total > 0
            else 0.0
        )

        rows.append({
            "ply": ply,
            "move_number": ply / 2,
            "total_positions_at_ply": total,
            "unique_positions_at_same_ply": unique_at_same_ply,
            "repeated_positions_at_same_ply": repeated_at_same_ply,
            "repeated_percentage_at_same_ply": repeated_same_ply_pct,
            "globally_repeated_positions": globally_repeated_count,
            "globally_unique_positions": globally_unique_count,
            "globally_repeated_percentage": globally_repeated_pct,
        })

    return rows


def save_csv(rows, output_csv: Path):
    fieldnames = [
        "ply",
        "move_number",
        "total_positions_at_ply",
        "unique_positions_at_same_ply",
        "repeated_positions_at_same_ply",
        "repeated_percentage_at_same_ply",
        "globally_repeated_positions",
        "globally_unique_positions",
        "globally_repeated_percentage",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values, window: int):
    if window <= 1:
        return values

    smoothed = []

    for i in range(len(values)):
        left = max(0, i - window // 2)
        right = min(len(values), i + window // 2 + 1)

        smoothed.append(sum(values[left:right]) / (right - left))

    return smoothed


def plot_repetition(
    rows,
    output_png: Path,
    min_positions_per_ply: int,
    smooth_window: int,
):
    filtered = [
        row for row in rows
        if row["total_positions_at_ply"] >= min_positions_per_ply
    ]

    if not filtered:
        raise ValueError(
            f"No ply has at least {min_positions_per_ply} positions. "
            "Try lowering --min_positions_per_ply."
        )

    plies = [row["ply"] for row in filtered]

    global_repeated_pct = [
        row["globally_repeated_percentage"]
        for row in filtered
    ]

    same_ply_repeated_pct = [
        row["repeated_percentage_at_same_ply"]
        for row in filtered
    ]

    total_positions = [
        row["total_positions_at_ply"]
        for row in filtered
    ]

    global_repeated_pct = moving_average(global_repeated_pct, smooth_window)
    same_ply_repeated_pct = moving_average(same_ply_repeated_pct, smooth_window)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        plies,
        global_repeated_pct,
        linewidth=2,
        label="Repeated anywhere in dataset (%)",
    )

    ax1.plot(
        plies,
        same_ply_repeated_pct,
        linewidth=1.5,
        linestyle="--",
        label="Repeated at same ply only (%)",
    )

    ax1.set_xlabel("Ply number, i.e. half-move number")
    ax1.set_ylabel("Repeated positions (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        plies,
        total_positions,
        linewidth=1.2,
        linestyle=":",
        label="Total positions at this ply",
    )

    ax2.set_ylabel("Number of positions at this ply")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
    )

    plt.title(
        "Board-state repetition across game progression\n"
        "A state is counted as repeated if it appears anywhere in the PGN, even at another ply"
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze repeated chess board states by game progression in a PGN file."
    )

    parser.add_argument(
        "pgn_path",
        type=Path,
        help="Path to the .pgn file.",
    )

    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("board_state_repetition_by_ply.png"),
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("board_state_repetition_by_ply.csv"),
    )

    parser.add_argument(
        "--max_games",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_ply",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--key_mode",
        choices=["pieces", "state", "full_fen"],
        default="state",
    )

    parser.add_argument(
        "--exclude_initial_position",
        action="store_true",
    )

    parser.add_argument(
        "--min_positions_per_ply",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    positions_by_ply, global_counter, num_games = read_positions_by_ply(
        pgn_path=args.pgn_path,
        max_games=args.max_games,
        max_ply=args.max_ply,
        key_mode=args.key_mode,
        include_initial_position=not args.exclude_initial_position,
    )

    rows = build_rows(
        positions_by_ply=positions_by_ply,
        global_counter=global_counter,
    )

    save_csv(rows, args.output_csv)

    plot_repetition(
        rows=rows,
        output_png=args.output_png,
        min_positions_per_ply=args.min_positions_per_ply,
        smooth_window=args.smooth_window,
    )

    print(f"Processed {num_games} games.")
    print(f"Saved CSV to: {args.output_csv}")
    print(f"Saved plot to: {args.output_png}")


if __name__ == "__main__":
    main()