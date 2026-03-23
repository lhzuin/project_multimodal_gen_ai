import math
from typing import List, Tuple

# ============================================================
# Input data: (opponent_elo, wins, draws, losses)
# ============================================================

DISTILL_ENCODER_MATCHES = [ # Distill Encoder
(1600, 331, 432, 349),
(1700, 290, 406, 416),
(1800, 203, 441, 468),
]

DECODER_MATCHES = [ # Decoder
    (1350, 250, 342, 520),
    (1400, 198, 314, 600),
    (1600,  71, 270, 771),
    (1500, 107, 274, 731),
]


ENCODER_SEQ_MATCHES = [ # Encoder

    (2000, 386,	342,384),
    (2100, 414,	312, 386),
    (2200, 252, 364, 496)
]

MATCHES = ENCODER_SEQ_MATCHES

Z_95 = 1.959963984540054  # 95% normal quantile


# ------------------------------------------------------------
# Elo expected score
# ------------------------------------------------------------
def expected_score(player_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


# ------------------------------------------------------------
# Convert matches into grouped sufficient statistics
# Each group becomes (opp_elo, n_games, observed_mean_score)
# ------------------------------------------------------------
def grouped_stats(matches: List[Tuple[int, int, int, int]]) -> List[Tuple[float, int, float]]:
    groups = []
    for opp_elo, wins, draws, losses in matches:
        n = wins + draws + losses
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
    elos = [elo for elo,_,_,_ in matches]
    elo = int(sum(elos)/len(elos))

    for _ in range(max_iter):
        u, info = score_and_info(elo, groups)
        if info <= 0:
            raise RuntimeError("Information became non-positive; check input data.")
        step = u / info
        new_elo = elo + step

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


def main():
    out = elo_summary(MATCHES)

    print("================================================")
    print("FINAL ELO ESTIMATION")
    print("================================================")
    print(f"Total games          : {out['total_games']}")
    print(f"Total score          : {out['total_score']:.1f}/{out['total_games']}")
    print(f"Average score        : {out['avg_score']:.6f}")
    print()
    print(f"Estimated Elo        : {out['elo_hat']:.2f}")
    print(f"Std. error           : {out['se']:.2f}")
    print(f"95% CI               : [{out['ci_low']:.2f}, {out['ci_high']:.2f}]")
    print()
    print(f"Poster format (1σ)   : {out['elo_hat']:.0f} ± {out['se']:.0f}")
    print(f"Poster format (95%)  : {out['elo_hat']:.0f} ± {((out['ci_high'] - out['ci_low'])/2):.0f}")


if __name__ == "__main__":
    main()