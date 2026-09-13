#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(csv_path: Path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "ply",
            "total_positions_at_ply",
            "repeated_percentage_at_same_ply",
            "globally_repeated_percentage",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            rows.append({
                "ply": int(row["ply"]),
                "total_positions_at_ply": int(row["total_positions_at_ply"]),
                "repeated_percentage_at_same_ply": float(row["repeated_percentage_at_same_ply"]),
                "globally_repeated_percentage": float(row["globally_repeated_percentage"]),
            })

    return rows


def moving_average(values, window: int):
    if window <= 1:
        return values

    smoothed = []

    for i in range(len(values)):
        left = max(0, i - window // 2)
        right = min(len(values), i + window // 2 + 1)

        smoothed.append(sum(values[left:right]) / (right - left))

    return smoothed


def plot_repetition_from_csv(
    csv_path: Path,
    output_png: Path,
    min_positions_per_ply: int,
    smooth_window: int,
    max_ply: int | None,
):
    rows = read_rows(csv_path)

    filtered = [
        row for row in rows
        if row["total_positions_at_ply"] >= min_positions_per_ply
    ]

    if max_ply is not None:
        filtered = [
            row for row in filtered
            if row["ply"] <= max_ply
        ]

    if not filtered:
        raise ValueError(
            f"No ply remains after filtering. "
            f"Try lowering --min_positions_per_ply or increasing --max_ply."
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

    # Larger fonts for Beamer readability.
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    })

    fig, ax1 = plt.subplots(figsize=(13, 7))

    ax1.plot(
        plies,
        global_repeated_pct,
        linewidth=4.0,
        label="Repeated anywhere in dataset (%)",
    )

    ax1.plot(
        plies,
        same_ply_repeated_pct,
        linewidth=3.0,
        linestyle="--",
        label="Repeated at same ply only (%)",
    )

    ax1.set_title(
        "Board-State Repetition Across Game Progression",
        fontweight="bold",
        pad=18,
    )

    ax1.set_xlabel("Ply number")
    ax1.set_ylabel("Repeated positions (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(min(plies), max(plies))
    ax1.grid(True, alpha=0.35, linewidth=1.0)

    ax2 = ax1.twinx()

    ax2.plot(
        plies,
        total_positions,
        linewidth=2.5,
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
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the board-state repetition plot from an existing CSV."
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to board_state_repetition_by_ply.csv.",
    )

    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("board_state_repetition_by_ply_clean.png"),
        help="Path where the cleaned plot will be saved.",
    )

    parser.add_argument(
        "--min_positions_per_ply",
        type=int,
        default=50,
        help="Ignore plies with fewer positions than this value.",
    )

    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving-average smoothing window. Use 1 for no smoothing.",
    )

    parser.add_argument(
        "--max_ply",
        type=int,
        default=None,
        help="Optional maximum ply to plot.",
    )

    args = parser.parse_args()

    plot_repetition_from_csv(
        csv_path=args.csv_path,
        output_png=args.output_png,
        min_positions_per_ply=args.min_positions_per_ply,
        smooth_window=args.smooth_window,
        max_ply=args.max_ply,
    )


if __name__ == "__main__":
    main()