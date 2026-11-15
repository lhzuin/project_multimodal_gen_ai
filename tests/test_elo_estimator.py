"""Elo fitting uses grouped game counts and handles extreme initial guesses."""

import math
import unittest

from utils.elo_estimator import elo_summary, estimate_elo_mle, expected_score


class EloTests(unittest.TestCase):
    def test_equal_score_recovers_opponent_rating(self):
        result = elo_summary([(2000, 20, 60, 20)])
        self.assertAlmostEqual(result["elo_hat"], 2000)
        self.assertEqual(result["total_games"], 100)
        self.assertEqual(result["avg_score"], .5)
        self.assertLess(result["ci_low"], result["elo_hat"])
        self.assertGreater(result["ci_high"], result["elo_hat"])

    def test_extreme_initial_guesses_converge_to_the_same_estimate(self):
        matches = [(1600, 30, 40, 30), (1800, 10, 40, 50)]
        low, _ = estimate_elo_mle(matches, init=-10000)
        high, _ = estimate_elo_mle(matches, init=10000)
        self.assertAlmostEqual(low, high)
        self.assertTrue(math.isfinite(low))
        self.assertEqual(expected_score(1e6, 0), 1)
        self.assertEqual(expected_score(-1e6, 0), 0)

    def test_invalid_or_separated_samples_fail_explicitly(self):
        for matches in ([], [(2000, 0, 0, 0)], [(2000, -1, 2, 0)],
                        [(2000, 10, 0, 0)], [(2000, 0, 0, 10)]):
            with self.subTest(matches=matches), self.assertRaises(ValueError):
                elo_summary(matches)


if __name__ == "__main__":
    unittest.main()
