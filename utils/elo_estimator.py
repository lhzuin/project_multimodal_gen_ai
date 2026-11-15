"""Fit an approximate logistic Elo rating from grouped match scores.

Use --matches for JSON rows [opponent_elo, wins, draws, losses], or repeat
--summary for native tournament summary.json files. Draws contribute half a point.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

Z_95 = 1.959963984540054  # 95% normal quantile


# ------------------------------------------------------------
# Elo expected score
# ------------------------------------------------------------
def expected_score(player_elo: float, opponent_elo: float) -> float:
    value = (player_elo - opponent_elo) * math.log(10.0) / 400.0
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


# ------------------------------------------------------------
# Convert matches into grouped sufficient statistics
# Each group becomes (opp_elo, n_games, observed_mean_score)
# ------------------------------------------------------------
def grouped_stats(matches: List[Tuple[int, int, int, int]]) -> List[Tuple[float, int, float]]:
    if not matches:
        raise ValueError("at least one opponent group is required")
    groups = []
    for opp_elo, wins, draws, losses in matches:
        if not math.isfinite(opp_elo) or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in (wins, draws, losses)):
            raise ValueError("ratings must be finite and match counts nonnegative integers")
        n = wins + draws + losses
        if n == 0:
            raise ValueError("each opponent group needs at least one game")
        score = wins + 0.5 * draws
        s = score / n
        groups.append((float(opp_elo), n, s))
    return groups


# ------------------------------------------------------------
# Score function U(R) = d logL / dR
# and observed Fisher information I(R) = -d² logL / dR²
#
# For logistic Elo:
# p'(R) = c * p * (1-p),  where c = ln(10)/400
#
# Grouped score:
# U(R) = c * sum_j N_j * (s_j - p_j)
#
# Observed / expected information:
# I(R) = c² * sum_j N_j * p_j * (1-p_j)
# ------------------------------------------------------------
def score_and_info(player_elo: float, groups: List[Tuple[float, int, float]]) -> Tuple[float, float]:
    c = math.log(10.0) / 400.0
    score = 0.0
    info = 0.0

    for opp_elo, n, s_obs in groups:
        p = expected_score(player_elo, opp_elo)
        score += n * (s_obs - p)
        info += n * p * (1.0 - p)

    score *= c
    info *= c * c
    return score, info


# ------------------------------------------------------------
# Fast MLE with Newton-Raphson
# ------------------------------------------------------------
def estimate_elo_mle(matches: List[Tuple[int, int, int, int]], init: float = 1800.0,
                     tol: float = 1e-10, max_iter: int = 100) -> Tuple[float, float]:
    """
    Returns:
        elo_hat, observed_information_at_hat
    """
    groups = grouped_stats(matches)
    if all(score == 0 for _, _, score in groups) or all(score == 1 for _, _, score in groups):
        raise ValueError("an all-win or all-loss sample has no finite Elo estimate")
    lower = min(elo for elo, _, _ in groups) - 4000
    upper = max(elo for elo, _, _ in groups) + 4000
    elo = min(upper, max(lower, float(init)))

    for _ in range(max_iter):
        u, info = score_and_info(elo, groups)
        if info <= 0:
            raise RuntimeError("Information became non-positive; check input data.")
        if abs(u / info) < tol:
            break
        if u > 0:
            lower = elo
        else:
            upper = elo
        new_elo = elo + u / info
        if not lower < new_elo < upper:
            new_elo = (lower + upper) / 2

        if abs(new_elo - elo) < tol:
            elo = new_elo
            break
        elo = new_elo

    _, info = score_and_info(elo, groups)
    return elo, info


# ------------------------------------------------------------
# Standard error and CI from Fisher information
# Var(elo_hat) ≈ 1 / I(elo_hat)
# ------------------------------------------------------------
def elo_summary(matches: List[Tuple[int, int, int, int]]) -> dict:
    elo_hat, info = estimate_elo_mle(matches)
    se = math.sqrt(1.0 / info)
    ci_low = elo_hat - Z_95 * se
    ci_high = elo_hat + Z_95 * se

    total_games = sum(w + d + l for _, w, d, l in matches)
    total_score = sum(w + 0.5 * d for _, w, d, l in matches)
    avg_score = total_score / total_games

    return {
        "elo_hat": elo_hat,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "total_games": total_games,
        "total_score": total_score,
        "avg_score": avg_score,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--matches", type=Path)
    inputs.add_argument("--summary", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.matches:
        matches = json.loads(args.matches.read_text(encoding="utf-8"))
    else:
        matches = []
        for path in args.summary:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not saved["stockfish_config"].get("limit_strength"):
                raise ValueError(f"{path}: opponent does not use a calibrated UCI Elo")
            matches.append((saved["stockfish_config"]["uci_elo"], saved["wins"], saved["draws"], saved["losses"]))
    print(json.dumps(elo_summary(matches), indent=2))



if __name__ == "__main__":
    main()
