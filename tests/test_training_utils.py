import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from train.train_chessdecoder import Trainer as DecoderTrainer
from train.train_encoder_seq import Trainer as EncoderTrainer
from train.training_utils import (
    CheckpointManager,
    SafeWandbRun,
    capture_rng_state,
    model_weights,
    restore_rng_state,
    select_device,
)


class CheckpointManagerTests(unittest.TestCase):
    def test_saves_full_latest_and_rotates_weight_snapshots(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            manager = CheckpointManager(root / "model.pt", keep_last=2)

            for epoch in range(4):
                weights = {"weight": torch.tensor([float(epoch)])}
                state = {"model": weights, "optimizer": {"step": epoch}, "epoch": epoch}
                manager.save_latest(state, weights, stage_name="fine tune", epoch=epoch)

            latest = manager.load(manager.latest_path)
            self.assertEqual(latest["epoch"], 3)
            self.assertEqual(latest["optimizer"], {"step": 3})
            self.assertTrue(torch.equal(model_weights(latest)["weight"], torch.tensor([3.0])))

            snapshots = sorted(root.glob("model.*.epoch-*.pt"))
            self.assertEqual(
                [path.name for path in snapshots],
                ["model.fine-tune.epoch-0002.pt", "model.fine-tune.epoch-0003.pt"],
            )
            self.assertEqual(set(manager.load(snapshots[-1])), {"weight"})
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_supports_auto_resume_and_best_weights(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            manager = CheckpointManager(Path(raw_tmp) / "experiment.latest.pt")
            weights = {"bias": torch.tensor([2.0])}
            manager.save_latest({"model": weights}, weights, stage_name="stage1", epoch=0)
            manager.save_best(weights)

            self.assertEqual(manager.resolve_resume_path("auto"), manager.latest_path)
            self.assertEqual(manager.best_path.name, "experiment.best.pt")
            self.assertTrue(
                torch.equal(manager.load_best_weights()["bias"], torch.tensor([2.0]))
            )


class RuntimeStateTests(unittest.TestCase):
    def test_rng_state_round_trip(self):
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(1))

        random.random()
        np.random.rand()
        torch.rand(1)
        restore_rng_state(state)

        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_safe_wandb_run_does_not_propagate_logging_errors(self):
        class BrokenRun:
            id = "run-id"

            def log(self, _values):
                raise ConnectionError("offline")

            def finish(self):
                raise ConnectionError("offline")

        run = SafeWandbRun(BrokenRun())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run.log({"loss": 1.0})
            run.log({"loss": 0.5})
            run.finish()

        self.assertEqual(run.id, "run-id")
        self.assertEqual(len(caught), 2)

    def test_explicit_cpu_device_is_always_available(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_auto_device_prefers_cuda_then_mps(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(select_device("auto"), torch.device("cuda"))
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(select_device("auto"), torch.device("mps"))

    def test_both_trainers_restore_full_mid_epoch_state(self):
        for trainer_class in (EncoderTrainer, DecoderTrainer):
            with (
                self.subTest(trainer=trainer_class.__module__),
                tempfile.TemporaryDirectory() as raw_tmp,
            ):
                checkpoint_path = str(Path(raw_tmp) / "latest.pt")
                cfg = OmegaConf.create(
                    {
                        "seed": 42,
                        "acc_epsilon": 0.001,
                        "checkpoint_save_path": checkpoint_path,
                        "checkpoint_load_path": None,
                        "checkpoint_keep_last": 2,
                        "early_stopping": {"patience": 3, "min_epochs": 1},
                        "trainer": {"grad_accum_steps": 1, "grad_clip_norm": 1.0},
                        "optim": {
                            "_target_": "torch.optim.AdamW",
                            "lr": 0.01,
                            "weight_decay": 0.0,
                        },
                    }
                )
                source = trainer_class(cfg, logger=None, device=torch.device("cpu"))
                source.model = torch.nn.Linear(2, 1)
                source.optimizer = torch.optim.AdamW(source.model.parameters(), lr=0.01)
                source.scheduler = torch.optim.lr_scheduler.StepLR(
                    source.optimizer, step_size=1
                )
                loss = source.model(torch.ones(1, 2)).sum()
                loss.backward()
                source.optimizer.step()
                source.scheduler.step()
                source.stage_name = "stage1"
                source.stage_index = 0
                source.current_epoch = 4
                source.next_batch_idx = 7
                source.global_step = 13
                source.epoch_generator_state = source.train_generator.get_state()
                source.best_val_loss = 0.25
                source.best_val_acc = 0.75
                source.epochs_no_improve = 2
                source.save_checkpoint(epoch_complete=False)

                resumed_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                resumed_cfg.checkpoint_load_path = "auto"
                resumed = trainer_class(resumed_cfg, logger=None, device=torch.device("cpu"))
                resumed.model = torch.nn.Linear(2, 1)
                resumed.optimizer = torch.optim.AdamW(resumed.model.parameters(), lr=0.01)
                resumed.scheduler = torch.optim.lr_scheduler.StepLR(
                    resumed.optimizer, step_size=1
                )

                progress = resumed._restore_training_state("stage1", 0)
                self.assertEqual(progress[:2], (4, 7))
                self.assertEqual(progress[2:5], (0.25, 0.75, 2))
                self.assertEqual(resumed.global_step, 13)
                self.assertEqual(
                    resumed.scheduler.last_epoch, source.scheduler.last_epoch
                )
                self.assertTrue(resumed.optimizer.state)
                for expected, actual in zip(source.model.parameters(), resumed.model.parameters()):
                    self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
