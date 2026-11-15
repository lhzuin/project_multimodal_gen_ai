"""Game-cluster aggregation for the disjoint TWIC checkpoint comparison."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path

from analysis.ply_comparison.plot_results import BUCKETS, FAMILIES, MODELS

RELATIVE_BUCKETS = ("q0_20", "q20_40", "q40_60", "q60_80", "q80_100")


def relative_bucket_for_ply(ply: int, game_plies: int) -> str:
    if not 0 <= ply < game_plies:
        raise ValueError("ply must be inside a nonempty game")
    return RELATIVE_BUCKETS[min(4, 5 * ply // game_plies)]


def bucket_for_ply(ply: int) -> str:
    if ply < 0:
        raise ValueError("ply must be nonnegative")
    for start, stop, label in (
        (0, 20, "0_20"), (20, 40, "20_40"), (40, 60, "40_60"),
        (60, 80, "60_80"), (80, 100, "80_100"),
    ):
        if start <= ply < stop:
            return label
    return "100_plus"


@dataclass
class MetricSums:
    n: int = 0
    top1: float = 0.0
    top5: float = 0.0
    nll: float = 0.0

    def add(self, top1: bool, top5: bool, nll: float) -> None:
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("NLL must be finite and nonnegative")
        self.n += 1
        self.top1 += int(top1)
        self.top5 += int(top5)
        self.nll += nll


@dataclass
class ComparisonAccumulator:
    # Sparse per-game state keeps bootstrap units intact without retaining
    # millions of per-ply rows in memory.
    games: dict[int, dict[tuple[str, str], MetricSums]] = field(default_factory=dict)

    def add(self, game_id: int, ply: int, model_id: str, *, top1: bool,
            top5: bool, nll: float) -> None:
        self.add_bucket(game_id, bucket_for_ply(ply), model_id,
                        top1=top1, top5=top5, nll=nll)

    def add_bucket(self, game_id: int, bucket: str, model_id: str, *,
                   top1: bool, top5: bool, nll: float) -> None:
        if model_id not in MODELS:
            raise ValueError(f"unknown model: {model_id}")
        key = (model_id, bucket)
        stats = self.games.setdefault(game_id, {}).setdefault(key, MetricSums())
        stats.add(top1, top5, nll)

    def _game_arrays(self, bucket: str):
        # Imported only for aggregation; inference already requires NumPy.
        import numpy as np

        rows = []
        for game_id, game_stats in self.games.items():
            if (MODELS[0], bucket) not in game_stats:
                continue
            counts = [game_stats[(model, bucket)].n for model in MODELS]
            if len(set(counts)) != 1:
                raise ValueError(f"unequal model counts in game {game_id}, bucket {bucket}")
            rows.append([counts[0]] + [game_stats[(model, bucket)].top1 for model in MODELS]
                        + [game_stats[(model, bucket)].top5 for model in MODELS]
                        + [game_stats[(model, bucket)].nll for model in MODELS])
        if not rows:
            raise ValueError(f"no observations for bucket {bucket}")
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _interval(samples):
        import numpy as np

        low, high = np.percentile(samples, [2.5, 97.5])
        return float(low), float(high)

    def summarize(self, *, bootstrap_replicates: int = 2000, seed: int = 42,
                  buckets=BUCKETS):
        import numpy as np

        if bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        rng = np.random.default_rng(seed)
        summaries = []
        deltas = []
        for bucket in buckets:
            array = self._game_arrays(bucket)
            n_games = len(array)
            n_plies = int(array[:, 0].sum())
            totals = array[:, 1:].sum(axis=0) / n_plies
            for i, model in enumerate(MODELS):
                game_top1 = array[:, 1 + i] / array[:, 0]
                game_nll = array[:, 9 + i] / array[:, 0]
                summaries.append({
                    "model_id": model, "bucket": bucket,
                    "n_games": n_games, "n_plies": n_plies,
                    "accuracy_top1": float(totals[i]),
                    "accuracy_top5": float(totals[4 + i]),
                    "nll_legal": float(totals[8 + i]),
                    "perplexity_legal": float(math.exp(totals[8 + i])),
                    "accuracy_top1_equal_game": float(game_top1.mean()),
                    "nll_legal_equal_game": float(game_nll.mean()),
                })

            # Bootstrap complete games, retaining all their plies and the
            # original/balanced pair on every draw. Chunking caps memory.
            draws = np.empty((bootstrap_replicates, 2, 2), dtype=np.float64)
            for start in range(0, bootstrap_replicates, 32):
                stop = min(start + 32, bootstrap_replicates)
                indices = rng.integers(0, n_games, size=(stop - start, n_games))
                sampled = array[indices].sum(axis=1)
                denominator = sampled[:, 0]
                for family_index in range(2):
                    original = family_index * 2
                    balanced = original + 1
                    draws[start:stop, family_index, 0] = (
                        sampled[:, 1 + balanced] - sampled[:, 1 + original]
                    ) / denominator
                    draws[start:stop, family_index, 1] = (
                        sampled[:, 9 + balanced] - sampled[:, 9 + original]
                    ) / denominator

            for family_index, family in enumerate(FAMILIES):
                original = family_index * 2
                balanced = original + 1
                delta_acc = float(totals[balanced] - totals[original])
                delta_nll = float(totals[8 + balanced] - totals[8 + original])
                acc_low, acc_high = self._interval(draws[:, family_index, 0])
                nll_low, nll_high = self._interval(draws[:, family_index, 1])
                deltas.append({
                    "family": family, "bucket": bucket,
                    "n_games": n_games, "n_plies": n_plies,
                    "delta_accuracy_top1": delta_acc,
                    "ci_low_accuracy_top1": acc_low,
                    "ci_high_accuracy_top1": acc_high,
                    "delta_nll_legal": delta_nll,
                    "ci_low_nll_legal": nll_low,
                    "ci_high_nll_legal": nll_high,
                })
        return summaries, deltas

    def summarize_without_intervals(self, buckets):
        import numpy as np

        summaries = []
        for bucket in buckets:
            array = self._game_arrays(bucket)
            n_plies = int(array[:, 0].sum())
            totals = array[:, 1:].sum(axis=0) / n_plies
            for i, model in enumerate(MODELS):
                summaries.append({
                    "model_id": model, "bucket": bucket,
                    "n_games": len(array), "n_plies": n_plies,
                    "accuracy_top1": float(totals[i]),
                    "accuracy_top5": float(totals[4 + i]),
                    "nll_legal": float(totals[8 + i]),
                    "perplexity_legal": float(np.exp(totals[8 + i])),
                    "accuracy_top1_equal_game": float((array[:, 1 + i] / array[:, 0]).mean()),
                    "nll_legal_equal_game": float((array[:, 9 + i] / array[:, 0]).mean()),
                })
        return summaries


def write_summary_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = (
    "model_id", "bucket", "n_games", "n_plies", "accuracy_top1",
    "accuracy_top5", "nll_legal", "perplexity_legal",
    "accuracy_top1_equal_game", "nll_legal_equal_game",
)
DELTA_FIELDS = (
    "family", "bucket", "n_games", "n_plies", "delta_accuracy_top1",
    "ci_low_accuracy_top1", "ci_high_accuracy_top1", "delta_nll_legal",
    "ci_low_nll_legal", "ci_high_nll_legal",
)
