"""Position-weighted metrics and paired whole-game bootstrap comparisons."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from analysis.ply_comparison.plot_results import BUCKETS


def game_arrays(store):
    """Aggregate each complete game's paired predictions without loading move rows."""
    select = """
        SELECT b.game_id, b.bucket, COUNT(*),
               SUM(b.top1_correct), SUM(b.top5_correct), SUM(b.nll_legal),
               SUM(c.top1_correct), SUM(c.top5_correct), SUM(c.nll_legal),
               SUM(b.predicted_uci != c.predicted_uci),
               SUM(b.top1_correct=0 AND c.top1_correct=1),
               SUM(b.top1_correct=1 AND c.top1_correct=0)
        FROM baseline b JOIN candidate c USING(game_id,ply)
    """
    groups = {}
    overall = {}
    for row in store.db.execute(select + " GROUP BY b.game_id,b.bucket ORDER BY b.game_id,b.bucket"):
        values = np.asarray(tuple(row)[2:], dtype=np.float64)
        groups.setdefault(row[1], []).append(values)
        overall.setdefault(row[0], np.zeros(10, dtype=np.float64))[:] += values
    if not overall:
        raise ValueError("no paired predictions to summarize")
    groups["overall"] = list(overall.values())
    # A supplementary view prevents forced moves from dominating the reading
    # of agreement. The all-position result remains the primary comparison.
    nonforced = [np.asarray(tuple(row)[2:], dtype=np.float64) for row in store.db.execute(
        select + " WHERE b.n_legal>1 GROUP BY b.game_id ORDER BY b.game_id")]
    if nonforced:
        groups["nonforced"] = nonforced
    return {bucket: np.stack(values) for bucket, values in groups.items()}


def summarize_arrays(groups, *, replicates=2000, seed=42, candidate_id="decoder_rloo_v5"):
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    summaries, differences = [], []
    rng = np.random.default_rng(seed)
    for bucket in ("overall", *BUCKETS, "nonforced"):
        if bucket not in groups:
            continue
        values = groups[bucket]
        n_games, n_positions = len(values), int(values[:, 0].sum())
        totals = values.sum(axis=0) / n_positions
        for model, offset in (("decoder_baseline", 1), (candidate_id, 4)):
            nll = float(totals[offset + 2])
            summaries.append({
                "model_id": model, "bucket": bucket, "n_games": n_games, "n_positions": n_positions,
                "accuracy_top1": float(totals[offset]), "accuracy_top5": float(totals[offset + 1]),
                "nll_legal": nll, "perplexity_legal": math.exp(nll),
                "accuracy_top1_equal_game": float(np.mean(values[:, offset] / values[:, 0])),
                "nll_legal_equal_game": float(np.mean(values[:, offset + 2] / values[:, 0])),
            })
        # Only sufficient sums are resampled. This estimates the ratio of
        # totals, not an unweighted mean of within-game accuracy changes.
        paired = np.column_stack((values[:, 0], values[:, 4:7] - values[:, 1:4]))
        draws = np.empty((replicates, 3), dtype=np.float64)
        for start in range(0, replicates, 16):
            stop = min(start + 16, replicates)
            indices = rng.integers(0, n_games, size=(stop - start, n_games))
            sampled = paired[indices].sum(axis=1)
            draws[start:stop] = sampled[:, 1:] / sampled[:, :1]
        result = {"bucket": bucket, "n_games": n_games, "n_positions": n_positions}
        for i, name in enumerate(("accuracy_top1", "accuracy_top5", "nll_legal")):
            result[f"delta_{name}"] = float(totals[4 + i] - totals[1 + i])
            low, high = np.quantile(draws[:, i], [.025, .975])
            result[f"ci_low_{name}"] = float(low)
            result[f"ci_high_{name}"] = float(high)
        result.update(disagreement_rate=float(totals[7]), beneficial_switch_rate=float(totals[8]),
                      harmful_switch_rate=float(totals[9]))
        if not math.isclose(result["delta_accuracy_top1"], totals[8] - totals[9], abs_tol=1e-12):
            raise ValueError("inconsistent beneficial/harmful switch accounting")
        differences.append(result)
    return summaries, differences


def _csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(directory: Path, store, *, replicates, seed, candidate_id, full_corpus):
    summaries, differences = summarize_arrays(game_arrays(store), replicates=replicates,
                                               seed=seed, candidate_id=candidate_id)
    _csv(directory / "summary.csv", summaries)
    _csv(directory / "paired_deltas.csv", differences)
    document = {"full_corpus": full_corpus, "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                "delta_direction": "candidate minus saved supervised baseline",
                "summaries": summaries, "paired_deltas": differences,
                "baseline_compatibility_check": store.get("baseline_check")}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(directory / "summary.json")
    overall = differences[0]
    lo, hi = overall["ci_low_accuracy_top1"], overall["ci_high_accuracy_top1"]
    conclusion = ("increases" if lo > 0 else "decreases" if hi < 0
                  else "has an unresolved change in")
    lines = ["# Decoder agreement with recorded human moves", "",
             f"{'Full corpus' if full_corpus else 'SMOKE TEST: restricted corpus'}; {overall['n_games']:,} games and {overall['n_positions']:,} human decisions.", "",
             f"The candidate {conclusion} legal top-1 agreement: change {100*overall['delta_accuracy_top1']:+.3f} percentage points, paired 95% interval [{100*lo:+.3f}, {100*hi:+.3f}].", "",
             "| Progression | Games | Decisions | Baseline top-1 | Candidate top-1 | Change, pp [95% interval] | NLL change [95% interval] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for delta in differences:
        bucket = delta["bucket"]
        base, candidate = [r for r in summaries if r["bucket"] == bucket]
        lines.append(f"| {bucket} | {delta['n_games']:,} | {delta['n_positions']:,} | {100*base['accuracy_top1']:.3f}% | {100*candidate['accuracy_top1']:.3f}% | {100*delta['delta_accuracy_top1']:+.3f} [{100*delta['ci_low_accuracy_top1']:+.3f}, {100*delta['ci_high_accuracy_top1']:+.3f}] | {delta['delta_nll_legal']:+.4f} [{delta['ci_low_nll_legal']:+.4f}, {delta['ci_high_nll_legal']:+.4f}] |")
    lines += ["", "All changes are candidate minus baseline. Positive accuracy changes and negative NLL changes favor agreement with human decisions. Top-5 results, equal-game averages, policy disagreement, and beneficial/harmful switches are in the CSV files. The nonforced row excludes decisions with only one legal move; it is supplementary.", "",
              f"Inputs and legal-move scoring reuse the original TWIC evaluator, with up to 256 preceding-history tokens, greedy legal top-1, and untempered legal NLL. The baseline's saved predictions are reused; one game per represented TWIC issue checks inference compatibility. Paired bootstrap resampling retains each whole game ({replicates:,} draws, seed {seed}). Later bands contain fewer games. Intervals condition on these checkpoints and this corpus; recurring openings and event/player dependence are not independently resampled.", "",
              "Together with the tournament result, this measures whether stronger complete-game play accompanies increased or reduced imitation on recorded human positions. It does not establish individualized human style or alignment on the policy's own generated trajectories.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    return document
