import unittest

from analysis.ply_comparison.aggregate import ComparisonAccumulator, bucket_for_ply
from analysis.ply_comparison.plot_results import BUCKETS, MODELS


class AggregateTests(unittest.TestCase):
    def test_ply_bucket_boundaries(self):
        cases = {
            0: "0_20", 19: "0_20", 20: "20_40", 39: "20_40",
            40: "40_60", 59: "40_60", 60: "60_80", 79: "60_80",
            80: "80_100", 99: "80_100", 100: "100_plus",
        }
        for ply, expected in cases.items():
            self.assertEqual(bucket_for_ply(ply), expected)

    def test_paired_game_bootstrap_and_decision_weighted_means(self):
        acc = ComparisonAccumulator()
        representative_plies = [0, 20, 40, 60, 80, 100]
        for ply in representative_plies:
            # Game 0 has one target, game 1 has three. The decision-weighted
            # baseline accuracy is 3/4, while equal-game weighting is 1/2.
            for game_id, repetitions in ((0, 1), (1, 3)):
                for offset in range(repetitions):
                    for model in MODELS:
                        baseline = model.endswith("baseline")
                        correct = game_id == 1 if baseline else True
                        acc.add(game_id, ply + offset, model,
                                top1=correct, top5=True,
                                nll=1.0 if baseline else 0.5)
        summary, deltas = acc.summarize(bootstrap_replicates=50, seed=7)
        self.assertEqual(len(summary), 24)
        self.assertEqual(len(deltas), 12)
        for bucket in BUCKETS:
            baseline = next(row for row in summary if row["model_id"] == "encoder_baseline"
                            and row["bucket"] == bucket)
            balanced = next(row for row in summary if row["model_id"] == "encoder_balanced"
                            and row["bucket"] == bucket)
            self.assertEqual((baseline["n_games"], baseline["n_plies"]), (2, 4))
            self.assertEqual(baseline["accuracy_top1"], 0.75)
            self.assertEqual(balanced["accuracy_top1"], 1.0)
            delta = next(row for row in deltas if row["family"] == "encoder"
                         and row["bucket"] == bucket)
            self.assertAlmostEqual(delta["delta_accuracy_top1"], 0.25)
            self.assertAlmostEqual(delta["delta_nll_legal"], -0.5)


if __name__ == "__main__":
    unittest.main()
