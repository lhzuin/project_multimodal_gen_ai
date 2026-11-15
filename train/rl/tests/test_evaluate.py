import contextlib
import io
import unittest

from train.grpo.tests.helpers import MemoryServer, TinyPolicy
from train.rl.evaluate import comparison_summary, evaluate_policy, select_prompts
from train.rloo import RLOOConfig


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_only_and_paired_identical_policy(self):
        rows = [{"entry_id": str(i)} for i in range(6)]
        prompts = select_prompts(rows, 4, 12)
        self.assertEqual(prompts, select_prompts(rows, 4, 12))
        policy = TinyPolicy()
        original = policy.model.weight.detach().clone()
        config = RLOOConfig(
            ply_depth=2,
            rollouts_per_entry=2,
            training_batch_size=2,
            max_concurrent_games=2,
            sampling_batch_size=2,
            poll_timeout_ms=1,
            device="cpu",
            randomize_policy_color=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = await evaluate_policy(MemoryServer(), policy, prompts, config)
            candidate = await evaluate_policy(MemoryServer(), policy, prompts, config)
        self.assertEqual(
            comparison_summary(baseline, candidate)["paired_reward_difference"], 0
        )
        self.assertTrue(policy.model.weight.equal(original))
        self.assertIsNone(policy.model.weight.grad)
        with self.assertRaises(ValueError):
            comparison_summary(baseline, {})

    async def test_summary_uses_position_means_and_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            select_prompts([{"entry_id": "a"}, {"entry_id": "a"}], 2, 1)
        a = {"a": {"mean_reward": -0.1}, "b": {"mean_reward": 0.1}}
        b = {"a": {"mean_reward": 0.1}, "b": {"mean_reward": 0.3}}
        self.assertAlmostEqual(
            comparison_summary(a, b)["paired_reward_difference"], 0.2
        )

    async def test_multiple_anchors_share_a_source_game_uncertainty_cluster(self):
        baseline = {str(i): {"mean_reward": 0, "game_id": str(i // 2)} for i in range(4)}
        candidate = {str(i): {"mean_reward": .2 if i < 2 else -.2, "game_id": str(i // 2)} for i in range(4)}
        result = comparison_summary(baseline, candidate)
        self.assertEqual(result["positions"], 4)
        self.assertEqual(result["independent_game_clusters"], 2)
        self.assertAlmostEqual(result["paired_standard_error"], .2)
        self.assertAlmostEqual(result["paired_reward_difference"], 0)

    async def test_legacy_independent_position_standard_error_is_preserved(self):
        baseline = {str(i): {"mean_reward": 0} for i in range(4)}
        candidate = {str(i): {"mean_reward": .2 if i < 2 else -.2} for i in range(4)}
        result = comparison_summary(baseline, candidate)
        self.assertAlmostEqual(result["paired_standard_error"], .2 / 3 ** .5)

    async def test_game_identity_must_match_and_more_than_one_game_is_needed(self):
        baseline = {str(i): {"mean_reward": 0, "game_id": "same"} for i in range(2)}
        with self.assertRaisesRegex(ValueError, "two independent"):
            comparison_summary(baseline, baseline)
        candidate = {key: {**row, "game_id": "other"} for key, row in baseline.items()}
        with self.assertRaisesRegex(ValueError, "different source-game"):
            comparison_summary(baseline, candidate)
