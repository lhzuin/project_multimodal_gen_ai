import math
import unittest

import torch

from train.rl.contracts import PolicyDecision, Trajectory
from train.rloo.config import RLOOConfig
from train.rloo.objective import (
    leave_one_out_advantages,
    reinforce_loss,
    training_items,
)


def trajectory(game_id, score, length=1, status="completed", sampled_kl=0.0):
    decision = PolicyDecision({}, 0, -1, (-1 - sampled_kl,))
    return Trajectory(
        game_id,
        "group",
        [decision] * length,
        {
            "status": status,
            "reward": {"centipawns": score, "perspective": "policy"},
        },
    )


class ObjectiveTests(unittest.TestCase):
    def test_no_policy_actions_still_contribute_to_other_samples_baseline(self):
        no_action = trajectory("a", -100)
        no_action.decisions = []
        items, metrics = training_items(
            [no_action, trajectory("b", 100)],
            RLOOConfig(
                rollouts_per_entry=2,
                kl_beta=0,
                reward_mode="linear",
                reward_scale_cp=100,
            ),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].advantage, 2)
        self.assertEqual(items[0].weight, 0.5)
        self.assertEqual(metrics["no_policy_trajectories"], 1)

    def test_baseline_excludes_own_return(self):
        result = leave_one_out_advantages(torch.tensor([1.0, 3.0, 5.0]))
        torch.testing.assert_close(result, torch.tensor([-3.0, 0.0, 3.0]))

    def test_linear_rewards_preserve_absolute_scale(self):
        cfg = RLOOConfig(
            rollouts_per_entry=2, reward_mode="linear", reward_scale_cp=100, kl_beta=0
        )
        small, _ = training_items([trajectory("a", 100), trajectory("b", 101)], cfg)
        large, _ = training_items([trajectory("a", 100), trajectory("b", 200)], cfg)
        self.assertAlmostEqual(large[0].advantage, 100 * small[0].advantage)

    def test_trajectory_actions_are_summed_not_length_averaged(self):
        cfg = RLOOConfig(rollouts_per_entry=2, kl_beta=0)
        items, _ = training_items(
            [trajectory("a", 100), trajectory("b", 300, length=3)], cfg
        )
        self.assertEqual([item.weight for item in items], [0.5] * 4)

    def test_kl_cost_is_in_return_before_leave_one_out(self):
        cfg = RLOOConfig(rollouts_per_entry=2, kl_beta=0.1)
        items, metrics = training_items(
            [
                trajectory("a", 0, sampled_kl=2),
                trajectory("b", 0, sampled_kl=0),
            ],
            cfg,
        )
        self.assertAlmostEqual(items[0].advantage, -0.2)
        self.assertAlmostEqual(items[1].advantage, 0.2)
        self.assertAlmostEqual(metrics["mean_return"], -0.1)

    def test_mate_constant_is_bounded(self):
        cfg = RLOOConfig(rollouts_per_entry=2, kl_beta=0)
        items, _ = training_items(
            [trajectory("a", 100000), trajectory("b", -100000)], cfg
        )
        self.assertAlmostEqual(items[0].advantage, 2)
        self.assertAlmostEqual(items[1].advantage, -2)

    def test_equal_returns_produce_zero_without_dividing_by_variance(self):
        items, metrics = training_items(
            [
                trajectory("a", 10),
                trajectory("b", 10),
            ],
            RLOOConfig(rollouts_per_entry=2, kl_beta=0),
        )
        self.assertEqual([item.advantage for item in items], [0, 0])
        self.assertEqual(metrics["zero_variance_groups"], 1)

    def test_failed_and_incomplete_groups_are_dropped(self):
        cfg = RLOOConfig(rollouts_per_entry=2, kl_beta=0)
        for group in (
            [trajectory("a", 1)],
            [trajectory("a", 1), trajectory("b", 2, status="failed")],
        ):
            items, metrics = training_items(group, cfg)
            self.assertEqual(items, [])
            self.assertEqual(metrics["dropped_groups"], 1)

    def test_reference_is_required_when_kl_enabled(self):
        rollout = trajectory("a", 1)
        rollout.decisions = [PolicyDecision({}, 0, -1, None)]
        with self.assertRaisesRegex(ValueError, "reference probabilities"):
            training_items(
                [rollout, trajectory("b", 2)], RLOOConfig(rollouts_per_entry=2)
            )

    def test_diagnostics_distinguish_flat_reward_from_kl_signal(self):
        items, metrics = training_items(
            [trajectory("a", 100000, sampled_kl=2), trajectory("b", 100000)],
            RLOOConfig(rollouts_per_entry=2, kl_beta=0.1),
        )
        self.assertEqual(metrics["reward_zero_variance_groups"], 1)
        self.assertEqual(metrics["reward_low_spread_groups"], 1)
        self.assertEqual(metrics["zero_variance_groups"], 0)
        self.assertEqual(metrics["reward_saturation_fraction"], 1)
        self.assertEqual(metrics["mean_reward_group_std"], 0)
        self.assertAlmostEqual(metrics["mean_abs_advantage"], 0.2)
        self.assertAlmostEqual(metrics["mean_abs_kl_penalty"], 0.1)
        self.assertAlmostEqual(items[0].advantage, -0.2)

    def test_reward_diagnostics_empty_and_linear_groups(self):
        config = RLOOConfig(rollouts_per_entry=2, kl_beta=0, reward_mode="linear", reward_scale_cp=100)
        _, metrics = training_items([trajectory("a", 100), trajectory("b", 300)], config)
        self.assertEqual(metrics["mean_reward_group_spread"], 2)
        self.assertEqual(metrics["mean_reward_group_std"], 1)
        self.assertEqual(metrics["reward_saturation_fraction"], 0)
        _, metrics = training_items([], config)
        for key in ("mean_abs_advantage", "mean_reward_group_std", "reward_saturation_fraction"):
            self.assertEqual(metrics[key], 0)

    def test_reinforce_gradient_matches_analytic_two_action_case(self):
        logits = torch.tensor([0.2, -0.3], requires_grad=True)
        loss = reinforce_loss(
            logits.log_softmax(0),
            torch.tensor([1.0, -1.0]),
            torch.tensor([0.5, 0.5]),
        )
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-0.5, 0.5]))

    def test_expected_rloo_gradient_matches_exact_kl_regularized_objective(self):
        # Enumerate every pair of independent on-policy samples. This checks
        # that treating sampled KL as a detached return has the right expected
        # gradient, including the KL sign, rather than just checking its value.
        logits = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
        log_probs = logits.log_softmax(0)
        reference = torch.tensor([-0.4, 0.7], dtype=torch.float64).log_softmax(0)
        rewards = torch.tensor([-0.4, 0.7], dtype=torch.float64)
        beta = 0.13
        returns = rewards - beta * (log_probs.detach() - reference)
        expected_surrogate = logits.new_zeros(())
        for first in range(2):
            for second in range(2):
                indices = [first, second]
                probability = log_probs.detach()[indices].sum().exp()
                advantage = leave_one_out_advantages(returns[indices])
                expected_surrogate = expected_surrogate + probability * reinforce_loss(
                    log_probs[indices], advantage, logits.new_full((2,), 0.5)
                )
        actual = torch.autograd.grad(expected_surrogate, logits, retain_graph=True)[0]
        exact_loss = -(log_probs.exp() * (rewards - beta * (log_probs - reference))).sum()
        expected = torch.autograd.grad(exact_loss, logits)[0]
        torch.testing.assert_close(actual, expected)

    def test_invalid_config_and_nonfinite_rewards_are_rejected(self):
        with self.assertRaises(TypeError):
            RLOOConfig(update_epochs=2)
        with self.assertRaises(ValueError):
            RLOOConfig(reward_scale_cp=0)
        with self.assertRaises(ValueError):
            training_items(
                [trajectory("a", math.nan), trajectory("b", 2)],
                RLOOConfig(rollouts_per_entry=2),
            )
