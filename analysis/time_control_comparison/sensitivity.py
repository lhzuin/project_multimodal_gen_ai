"""Recompute category summaries after conservative corpus-overlap exclusions.

Reads the original prediction database without writing to it. Reports and
bootstrap summaries are written to a separate, new output directory.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from analysis.time_control_comparison.aggregate import summarize


ROOT = Path(__file__).resolve().parents[2]


def matched_table(connection, selected, quota=481):
    """Use the same number of hash-ranked complete games from each category."""
    if quota < 1:
        raise ValueError("matched-game quota must be positive")
    connection.execute("CREATE TEMP TABLE matched_ids (game_id TEXT PRIMARY KEY)")
    counts = {}
    for category in ("classic", "rapid", "blitz"):
        ranked = sorted((g for g in selected if g["category"] == category), key=lambda g: (g["rank"], g["game_id"]))[:quota]
        if len(ranked) != quota:
            raise ValueError(f"{category}: fewer than {quota} games remain")
        counts[category] = len(ranked)
        connection.executemany("INSERT INTO matched_ids VALUES (?)", [(g["game_id"],) for g in ranked])
    query = """
        SELECT g.category,r.model,COUNT(*),AVG(r.top1_correct),AVG(-r.target_logp)
        FROM games g JOIN matched_ids m USING(game_id) JOIN predictions r USING(game_id)
        GROUP BY g.category,r.model
    """
    result = {c: {"games": n, "models": {}} for c, n in counts.items()}
    for category, model, positions, agreement, nll in connection.execute(query):
        result[category]["models"][model] = {"positions": positions, "agreement": agreement, "nll": nll}
    connection.execute("DROP TABLE matched_ids")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/twic_1662_1667_time_controls")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/category_overlap_audit/overlap_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/category_overlap_sensitivity")
    parser.add_argument("--matched-games-per-category", type=int, default=481)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    audit = json.loads(args.audit.read_text())
    excluded = {h["test_game_id"] for model in audit["models"].values() for h in model["confirmed_matches"]}
    selection = json.loads((args.source / "selection_manifest.json").read_text())
    connection = sqlite3.connect(f"file:{args.source / 'results.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=MEMORY")
    try:
        # Reproduce the existing matched-size table before adding exclusions.
        original = matched_table(connection, selection["games"], args.matched_games_per_category)
        connection.execute("CREATE TEMP TABLE exclusions (game_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO exclusions VALUES (?)", [(gid,) for gid in sorted(excluded)])
        connection.execute("CREATE TEMP VIEW games AS SELECT * FROM main.games WHERE game_id NOT IN (SELECT game_id FROM exclusions)")
        retained = [g for g in selection["games"] if g["game_id"] not in excluded]
        output = args.output_dir
        output.mkdir(parents=True)
        summarize(SimpleNamespace(connection=connection), SimpleNamespace(output_dir=output, seed=42, bootstrap=2000))
        matched = matched_table(connection, retained, args.matched_games_per_category)
        (output / "sensitivity.json").write_text(json.dumps({
            "excluded_test_games": sorted(excluded),
            "retained_games_by_category": dict(Counter(g["category"] for g in retained)),
            "matched_games_per_category": args.matched_games_per_category,
            "original_matched_size_results": original,
            "conservative_matched_size_results": matched,
            "source_database_opened_read_only": True,
            "inference_rerun": False,
        }, indent=2) + "\n")
        print("Original and exclusion-adjusted matched-size tables saved.")
        print("Conservative retained cohorts:", dict(Counter(g["category"] for g in retained)))
        print("Saved resummarized predictions and bootstrap intervals:", output)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
