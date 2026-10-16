"""Logging tests use a fake W&B module: no network, account, or SDK required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from train.grpo.tests.helpers import MemoryServer, TinyPolicy
from train.rl.launch import run_training
from train.rl.metrics import MetricsLogger
from train.rl.optimization import clip_gradients
from train.rloo import RLOOConfig, RLOOTrainer


class MetricsTests(unittest.TestCase):
    def test_jsonl_flush_append_metadata_and_disabled_wandb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "metrics.jsonl"
            settings = {"jsonl_path": str(path), "stdout": False}
            with patch("train.rl.metrics.importlib.import_module") as imported:
                for resumed in (False, True):
                    with MetricsLogger(
                        settings,
                        run_config={"epochs": 1},
                        algorithm="rloo",
                        run_id="same-run",
                        resumed=resumed,
                    ) as logger:
                        logger({"completed_batches": 1 + int(resumed), "loss": -0.2})
                        # Read while open: every completed record is flushed.
                        self.assertEqual(
                            len(path.read_text().splitlines()), 1 + int(resumed)
                        )
                imported.assert_not_called()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual({r["run_id"] for r in rows}, {"same-run"})
            self.assertNotEqual(rows[0]["session_id"], rows[1]["session_id"])
            sessions = [
                json.loads(line)
                for line in Path(str(path) + ".sessions.jsonl").read_text().splitlines()
            ]
            self.assertEqual([s["resumed"] for s in sessions], [False, True])
            self.assertEqual(sessions[0]["config"], {"epochs": 1})

    def test_wandb_resume_and_success_cleanup(self):
        run = Mock()
        module = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
        with patch("train.rl.metrics.importlib.import_module", return_value=module):
            with MetricsLogger(
                {"stdout": False, "wandb": {"enabled": True, "mode": "online"}},
                run_config={},
                algorithm="grpo",
                run_id="abc",
                resumed=True,
            ) as logger:
                logger({"completed_batches": 8.0})
        self.assertEqual(module.init.call_args.kwargs["id"], "abc")
        self.assertEqual(module.init.call_args.kwargs["resume"], "allow")
        run.define_metric.assert_any_call("*", step_metric="completed_batches")
        run.log.assert_called_once_with({"completed_batches": 8.0})
        run.finish.assert_called_once_with(exit_code=0)

    def test_offline_fallback_and_logging_failure_preserve_jsonl(self):
        run = Mock()
        run.log.side_effect = RuntimeError("disconnected")
        module = SimpleNamespace(
            init=Mock(side_effect=[RuntimeError("no network"), run]), Settings=Mock()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with patch("train.rl.metrics.importlib.import_module", return_value=module):
                with self.assertWarns(UserWarning):
                    with MetricsLogger(
                        {
                            "stdout": False,
                            "jsonl_path": str(path),
                            "wandb": {"enabled": True, "mode": "online"},
                        },
                        run_config={},
                        algorithm="rloo",
                    ) as logger:
                        logger({"completed_batches": 1.0})
                        logger({"completed_batches": 2.0})
            self.assertEqual(module.init.call_args.kwargs["mode"], "offline")
            self.assertEqual(len(path.read_text().splitlines()), 2)
            run.log.assert_called_once()
            run.finish.assert_called_once_with(exit_code=1)

    def test_cleanup_on_training_exception(self):
        run = Mock()
        module = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
        with patch("train.rl.metrics.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with MetricsLogger(
                    {"wandb": {"enabled": True}}, run_config={}, algorithm="rloo"
                ):
                    raise RuntimeError("training failed")
        run.finish.assert_called_once_with(exit_code=1)

    def test_typo_and_conflicting_resume_id_fail(self):
        with self.assertRaises(ValueError):
            MetricsLogger({"json_path": "typo"}, run_config={}, algorithm="rloo")
        with self.assertRaises(ValueError):
            MetricsLogger(
                {"wandb": {"id": "different"}},
                run_config={},
                algorithm="rloo",
                run_id="original",
            )
        for settings in (
            {"stdout": "false"},
            {"wandb": {"enabled": "false"}},
            {"wandb": {"mode": "typo"}},
            {"wandb": {"init_timeout_s": -1}},
        ):
            with self.assertRaises(ValueError):
                MetricsLogger(settings, run_config={}, algorithm="rloo")

    def test_global_clipping_metrics_match_actual_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        result = clip_gradients([parameter], 1.0)
        self.assertEqual(result["grad_norm_pre_clip"], 5.0)
        self.assertEqual(result["grad_clipped"], 1.0)
        self.assertAlmostEqual(result["grad_clip_scale"], 0.2, places=6)
        self.assertAlmostEqual(
            parameter.grad.norm().item(),
            result["grad_norm_post_clip_estimate"],
            places=6,
        )
        parameter.grad = torch.tensor([0.1, 0.1])
        self.assertEqual(clip_gradients([parameter], 1.0)["grad_clipped"], 0.0)


class LauncherMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_journal_checkpoint_identity_and_resumed_progress(self):
        def factory(**kwargs):
            kwargs.update(policy=TinyPolicy(), checkpoint_path=None)
            return RLOOTrainer(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "prompts.json"
            database.write_text(json.dumps([{"entry_id": "a"}, {"entry_id": "b"}]))
            checkpoint = root / "run.latest.pt"
            config = {
                "database": {"manifest": str(database)},
                "checkpoint_path": "injected",
                "model_type": "encoder",
                "server": {"mode": "remote"},
                "rloo": {
                    "training_batch_size": 2,
                    "rollouts_per_entry": 2,
                    "ply_depth": 2,
                    "device": "cpu",
                    "poll_timeout_ms": 1,
                },
                "output_checkpoint": str(checkpoint),
                "logging": {"stdout": False},
            }
            with patch(
                "train.rl.launch.AsyncRolloutClient",
                side_effect=lambda **kw: MemoryServer(),
            ):
                await run_training(
                    config,
                    trainer_type=factory,
                    config_type=RLOOConfig,
                    algorithm="rloo",
                )
                config.update(resume_path=str(checkpoint), epochs=2)
                await run_training(
                    config,
                    trainer_type=factory,
                    config_type=RLOOConfig,
                    algorithm="rloo",
                )
            rows = [
                json.loads(line)
                for line in checkpoint.with_suffix(".metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([r["completed_batches"] for r in rows], [1, 2])
            self.assertEqual([r["epoch_progress"] for r in rows], [1, 2])
            self.assertEqual(len({r["run_id"] for r in rows}), 1)
            for row in rows:
                self.assertGreater(row["batch_seconds"], 0)
                self.assertIn("grad_clip_scale", row)
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["logging_run_id"], rows[0]["run_id"])
