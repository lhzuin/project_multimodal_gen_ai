"""Position-weighted metrics with paired complete-game bootstrap intervals."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from .common import CATEGORIES, PAIRS, write_csv, atomic_json, sha256
from pathlib import Path

MODEL_METRICS = ("agreement", "nll", "top5", "entropy", "pair_accuracy")
METRICS = tuple(f"{m}:{metric}" for m in CATEGORIES for metric in MODEL_METRICS) + tuple(
    f"{left}-{right}:disagreement" for left, right in PAIRS)


def interval(draws):
    return tuple(float(value) for value in np.percentile(draws, [2.5, 97.5], axis=0))


def game_arrays(connection):
    """Aggregate once in SQL, then retain only small per-game sufficient sums."""
    expressions = []
    aliases = {m: f"r{i}" for i, m in enumerate(CATEGORIES)}
    for model in CATEGORIES:
        alias = aliases[model]
        expressions += [f"SUM({alias}.top1_correct)", f"SUM(-{alias}.target_logp)",
                        f"SUM({alias}.top5_correct)", f"SUM({alias}.entropy)", f"SUM({alias}.pair_correct)"]
    expressions += [f"SUM({aliases[a]}.predicted_uci != {aliases[b]}.predicted_uci)" for a, b in PAIRS]
    joins = " ".join(
        f"JOIN predictions {alias} ON {alias}.game_id=p.game_id AND {alias}.ply=p.ply AND {alias}.model='{model}'"
        for model, alias in aliases.items())
    query = f"""
        SELECT g.category, p.game_id, p.bucket, p.turn_white, p.is_promotion,
               (p.n_legal > 1), COUNT(*), {', '.join(expressions)}
        FROM positions p JOIN games g USING(game_id) {joins}
        GROUP BY g.category, p.game_id, p.bucket, p.turn_white, p.is_promotion, (p.n_legal > 1)
    """
    grouped = defaultdict(dict)
    for row in connection.execute(query):
        category, gid, band, white, promotion, nonforced = tuple(row)[:6]
        values = np.asarray(tuple(row)[6:], dtype=np.float64)
        scopes = ["all", f"ply:{band}", "side:white" if white else "side:black"]
        if nonforced:
            scopes.append("nonforced")
        if promotion:
            scopes.append("promotion")
        for scope in scopes:
            per_game = grouped[(category, scope)]
            if gid not in per_game:
                per_game[gid] = np.zeros(len(values), dtype=np.float64)
            per_game[gid] += values
    return {key: np.stack([games[gid] for gid in sorted(games)]) for key, games in grouped.items()}


def bootstrap(array: np.ndarray, replicates: int, seed: int):
    """Resample identical complete-game indices for every metric and policy."""
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(METRICS)), dtype=np.float64)
    for start in range(0, replicates, 32):
        stop = min(start + 32, replicates)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        sampled = array[indices].sum(axis=1)
        draws[start:stop] = sampled[:, 1:] / sampled[:, :1]
    return draws


def summarize(store, config):
    atomic_json(config.output_dir / "summary_manifest.json", {"complete": False})
    arrays = game_arrays(store.connection)
    summaries, contrasts, disagreements = [], [], []
    primary_draws, primary_values = {}, {}
    metric_index = {name: i for i, name in enumerate(METRICS)}
    for (category, scope), array in sorted(arrays.items()):
        # Stable scope-specific RNG keeps intervals unchanged when a report adds another scope.
        salt = int.from_bytes(hashlib.sha256(f"{config.seed}:{category}:{scope}".encode()).digest()[:8], "big")
        draws = bootstrap(array, config.bootstrap, salt)
        totals = array.sum(axis=0)
        means = totals[1:] / totals[0]
        base = {"test_category": category, "scope": scope, "n_games": len(array), "n_positions": int(totals[0])}
        if scope == "all":
            primary_draws[category], primary_values[category] = draws, means
        for model in CATEGORIES:
            row = {**base, "model": model}
            for metric in MODEL_METRICS:
                i = metric_index[f"{model}:{metric}"]
                low, high = interval(draws[:, i])
                row.update({metric: float(means[i]), f"{metric}_ci_low": low, f"{metric}_ci_high": high,
                            f"{metric}_equal_game": float((array[:, i + 1] / array[:, 0]).mean())})
            summaries.append(row)
        for left, right in PAIRS:
            row = {**base, "model": left, "reference": right}
            for metric in ("agreement", "nll"):
                i, j = metric_index[f"{left}:{metric}"], metric_index[f"{right}:{metric}"]
                low, high = interval(draws[:, i] - draws[:, j])
                row.update({f"delta_{metric}": float(means[i] - means[j]),
                            f"delta_{metric}_ci_low": low, f"delta_{metric}_ci_high": high})
            contrasts.append(row)
            i = metric_index[f"{left}-{right}:disagreement"]
            low, high = interval(draws[:, i])
            disagreements.append({**base, "model": left, "reference": right,
                                  "disagreement": float(means[i]), "ci_low": low, "ci_high": high,
                                  "disagreement_equal_game": float((array[:, i + 1] / array[:, 0]).mean())})
    if set(primary_draws) != set(CATEGORIES):
        raise ValueError("every category requires evaluated games")
    macro_draws = np.mean([primary_draws[c] for c in CATEGORIES], axis=0)
    macro_values = np.mean([primary_values[c] for c in CATEGORIES], axis=0)
    macro = []
    for model in CATEGORIES:
        for metric in ("agreement", "nll"):
            i = metric_index[f"{model}:{metric}"]
            low, high = interval(macro_draws[:, i])
            macro.append({"model": model, "metric": metric, "category_macro_mean": float(macro_values[i]),
                          "ci_low": low, "ci_high": high})
    specialization = []
    for model in ("rapid", "blitz"):
        for metric in ("agreement", "nll"):
            i, j = metric_index[f"{model}:{metric}"], metric_index[f"classic:{metric}"]
            own = primary_draws[model][:, i] - primary_draws[model][:, j]
            classical = primary_draws["classic"][:, i] - primary_draws["classic"][:, j]
            value = (primary_values[model][i] - primary_values[model][j]
                     - primary_values["classic"][i] + primary_values["classic"][j])
            low, high = interval(own - classical)
            specialization.append({"model": model, "metric": metric,
                                   "contrast": "(model-classic on own category) - (model-classic on classic category)",
                                   "difference_in_differences": float(value), "ci_low": low, "ci_high": high})
    output = config.output_dir
    for name, rows in (("prediction_summary.csv", summaries), ("paired_deltas.csv", contrasts),
                       ("disagreement.csv", disagreements), ("category_macro.csv", macro),
                       ("specialization.csv", specialization)):
        write_csv(output / name, rows)
    # Compact game-level metadata lets later analyses reweight or cluster cohorts.
    games = []
    for row in store.connection.execute("SELECT * FROM games ORDER BY category, game_id"):
        headers = json.loads(row["headers_json"])
        games.append({"game_id": row["game_id"], "category": row["category"], "n_positions": row["n_positions"],
                      **{key: headers.get(key, "") for key in (
                          "Event", "Site", "Date", "Round", "White", "Black", "WhiteElo", "BlackElo", "Result", "TimeControl", "TWICIssue")}})
    write_csv(output / "games.csv", games)
    atomic_json(output / "summary_manifest.json", {
        "complete": True, "bootstrap_replicates": config.bootstrap, "seed": config.seed,
        "unit": "complete source game, stratified by test category",
        "weighting": "positions within category; equal category weighting only in category_macro.csv",
        "metric_units": "agreement/disagreement are fractions; NLL is nats per complete legal move",
        "scope": "fixed checkpoints and sample; intervals exclude training-seed variation and event/player dependence",
        "aggregation_code_sha256": sha256(Path(__file__)),
    })
    return summaries, contrasts, disagreements
