"""Add ordinary early anchors from the existing, already partitioned games.

No new source download, expert labels, confidence gate, or engine queries.
Original rows remain unchanged. New anchors contain strictly truncated history.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import chess

from train.rl.contracts import PositionPrompt
from utils.dataset_rl.positions import position_key
from utils.dataset_rl.sources import exclusion_positions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def augment(splits, counts, excluded, *, seed=42, min_ply=12, max_ply=23):
    """Retain existing anchors and add disjoint, color-balanced game prefixes."""
    if min_ply < 1 or max_ply < min_ply:
        raise ValueError("invalid early ply window")
    if set(splits) != set(counts) or any(n < 0 or n % 2 for n in counts.values()):
        raise ValueError("each split needs an even nonnegative early count")
    used = set(excluded)
    original_keys = set()
    owners = {}
    entries = set()
    games = defaultdict(dict)
    for split, rows in splits.items():
        for row in rows:
            prompt = PositionPrompt.from_mapping(row)
            prompt.validate("encoder")
            if prompt.entry_id in entries:
                raise ValueError("duplicate original entry ID")
            entries.add(prompt.entry_id)
            key = position_key(prompt.board())
            if key in excluded:
                raise ValueError("original anchor overlaps tournament exclusion")
            if key in original_keys:
                raise ValueError("original anchors are not globally unique")
            original_keys.add(key)
            used.add(key)
            game = str(prompt.metadata["game_id"])
            if not game or owners.setdefault(game, split) != split:
                raise ValueError("source game overlaps held-out partitions")
            games[split].setdefault(game, []).append(row)
    candidates = defaultdict(dict)
    prefix_splits = defaultdict(set)
    for split, source_games in games.items():
        for game, rows in source_games.items():
            # A later anchor provides the longest available real history.
            row = max(rows, key=lambda r: len(r["moves_uci"]))
            limit = min(max_ply, len(row["moves_uci"]) - 1)
            if limit < min_ply:
                continue
            board = chess.Board(row["start_fen"])
            choices = []
            for ply, uci in enumerate(row["moves_uci"][:limit], 1):
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError("illegal source prefix")
                board.push(move)
                if ply >= min_ply and not board.is_game_over(claim_draw=True):
                    key = position_key(board)
                    prefix_splits[key].add(split)
                    choices.append((ply, key, "white" if board.turn else "black"))
            candidates[split][game] = (row, choices)
    # Common book positions in different source games must not bridge splits.
    shared_prefixes = {key for key, partitions in prefix_splits.items() if len(partitions) > 1}
    used.update(shared_prefixes)
    output, report = {}, {}
    for split, rows in splits.items():
        rng = random.Random(f"{seed}|early|{split}")
        order = sorted(candidates[split])
        rng.shuffle(order)
        added = []
        colors = Counter()
        for game in order:
            if len(added) == counts[split]:
                break
            row, choices = candidates[split][game]
            choices = list(choices)
            rng.shuffle(choices)
            for ply, key, color in choices:
                if colors[color] >= counts[split] // 2 or key in used:
                    continue
                identifier = f"lichess-{game}-early-ply-{ply}"
                if identifier in entries:
                    raise ValueError("early ID collision")
                new = {
                    "entry_id": identifier, "start_fen": row["start_fen"],
                    "moves_uci": row["moves_uci"][:ply],
                    "metadata": {"source": "lichess", "bucket": "early",
                                 "game_id": game, "selected_ply": ply,
                                 "parent_entry_id": row["entry_id"]},
                }
                PositionPrompt.from_mapping(new).validate("encoder")
                added.append(new)
                entries.add(identifier)
                colors[color] += 1
                used.add(key)
                break  # at most one added state per source game
            if len(added) == counts[split]:
                break
        if len(added) != counts[split]:
            raise ValueError(f"{split}: only {len(added)} eligible early states for quota {counts[split]}")
        output[split] = [*rows, *added]
        report[split] = {"original_rows": len(rows), "added_early": len(added),
                         "total_rows": len(output[split]), "added_colors": dict(colors),
                         "added_ply_counts": dict(sorted(Counter(len(r["moves_uci"]) for r in added).items())),
                         "bucket_counts": dict(Counter(r["metadata"]["bucket"] for r in output[split]))}
    return output, {"splits": report, "shared_prefix_positions_excluded": len(shared_prefixes),
                    "game_disjoint_splits": True, "globally_unique_anchors": True,
                    "seed": seed, "min_ply": min_ply, "max_ply": max_ply}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("processed_games/rl/lichess_mixed_v1"))
    parser.add_argument("--output", type=Path, default=Path("processed_games/rl/lichess_mixed_final"))
    parser.add_argument("--train-early", type=int, default=4000)
    parser.add_argument("--heldout-early", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--max-ply", type=int, default=23)
    parser.add_argument("--exclusion-openings", type=Path, action="append",
                        help="Opening suite to exclude; repeat for multiple suites.")
    parser.add_argument("--exclusion-games", type=Path, action="append", default=[],
                        help="Tournament games.jsonl whose effective start FENs must be excluded.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    splits = {name: json.loads((args.source / f"{name}.json").read_text()) for name in ("train", "validation", "test")}
    suites = args.exclusion_openings or [Path("utils/opening_suite_large.json")]
    excluded = exclusion_positions(tuple(str(path) for path in suites))
    for historical in args.exclusion_games:
        with historical.open(encoding="utf-8") as handle:
            excluded.update(position_key(chess.Board(json.loads(line)["start_fen"]))
                            for line in handle if line.strip())
    counts = {"train": args.train_early, "validation": args.heldout_early, "test": args.heldout_early}
    result, report = augment(splits, counts, excluded, seed=args.seed,
                             min_ply=args.min_ply, max_ply=args.max_ply)
    report.update(source=str(args.source), source_sha256={name: sha(args.source / f"{name}.json") for name in splits},
                  exclusion_inputs_sha256={str(path): sha(path) for path in [*suites, *args.exclusion_games]},
                  excluded_tournament_positions=len(excluded),
                  note="Early means ordinary source-game prefixes, without a quiet-position or confidence filter; original rows unchanged. No full tournament continuation or source-game exclusion is claimed.")
    args.output.mkdir(parents=True)
    for name, rows in result.items():
        (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    report["output_sha256"] = {name: sha(args.output / f"{name}.json") for name in result}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
