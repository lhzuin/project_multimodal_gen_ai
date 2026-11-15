import math
import unittest

import torch

from train.grpo.contracts import PolicyDecision, Trajectory
from train.grpo.objective import clipped_grpo_loss, exact_legal_kl, group_training_items


def trajectory(game, group, score, count, status="completed"):
    decision = PolicyDecision({}, 0, -1.0, None)
    return Trajectory(
        game,
        group,
        [decision] * count,
        {
            "status": status,
            "reward": {"centipawns": score, "perspective": "policy"},
        },
    )


class ObjectiveTests(unittest.TestCase):
    def test_group_advantages_and_equal_trajectory_weight(self):
        items, metrics = group_training_items(
            [
                trajectory("a", "g", 100, 1),
                trajectory("b", "g", 300, 3),
            ],
            1e-6,
        )
        self.assertAlmostEqual(items[0].advantage, -1, places=6)
        self.assertAlmostEqual(items[-1].advantage, 1, places=6)
        self.assertAlmostEqual(items[0].weight, 0.5)
        self.assertAlmostEqual(sum(i.weight for i in items[1:]), 0.5)
        self.assertEqual(metrics["valid_groups"], 1)

    def test_zero_variance_has_zero_advantages(self):
        items, metrics = group_training_items(
            [
                trajectory("a", "g", 0, 1),
                trajectory("b", "g", 0, 1),
            ],
            1e-6,
        )
        self.assertEqual([i.advantage for i in items], [0, 0])
        self.assertEqual(metrics["zero_variance_groups"], 1)

    def test_failed_rollout_discards_entire_group(self):
        items, metrics = group_training_items(
            [
                trajectory("a", "g", 100, 1),
                trajectory("b", "g", 0, 1, "failed"),
            ],
            1e-6,
        )
        self.assertEqual(items, [])
        self.assertEqual(metrics["dropped_groups"], 1)

    def test_clipping_and_gradient(self):
        new = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
        loss, metrics = clipped_grpo_loss(
            new,
            torch.zeros(2),
            torch.tensor([1.0, -1.0]),
            torch.tensor([0.5, 0.5]),
            torch.zeros(2),
            clip_epsilon=0.2,
            kl_beta=0,
        )
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.zeros(2))
        self.assertEqual(metrics["clip_fraction"].item(), 1)

    def test_exact_kl_is_zero_at_reference_and_positive_elsewhere(self):
        ref = torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)
        self.assertAlmostEqual(exact_legal_kl(ref, ref).item(), 0, places=6)
        other = torch.log_softmax(torch.tensor([0.0, 1.0]), dim=0)
        self.assertGreater(exact_legal_kl(other, ref).item(), 0)
