from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset.balanced_training_plan import (
    BalancedPlanError,
    BalancedTrainingPlan,
    decoder_label_mask_from_ply_mask,
    decoder_token_slice_for_ply_window,
)
from dataset.collate import CollateGameSequences
from dataset.collate_llm import CollateLLMMoves


SELECTED_DTYPE = np.dtype(
    [
        ("game_id", "<u4"),
        ("pgn_start", "<u8"),
        ("pgn_end", "<u8"),
        ("ply_count", "<u4"),
        ("window_start", "<u4"),
        ("window_stop", "<u4"),
        ("anchor_ply", "<u4"),
        ("active_targets", "<u4"),
        ("game_mass", "<f8"),
    ]
)


def _write_tiny_plan(root: Path, profile: str = "encoder") -> tuple[Path, Path]:
    plan_dir = root / "plan"
    plan_dir.mkdir()
    reservoir = root / "reservoir.pgn"
    reservoir.write_bytes(b"0123456789")

    selected = np.zeros(1, dtype=SELECTED_DTYPE)
    selected[0] = (0, 0, 10, 5, 1, 4, 2, 2, 1.0)
    np.save(plan_dir / "selected_games.npy", selected, allow_pickle=False)
    np.save(plan_dir / "game_ply_offsets.npy", np.array([0, 5], dtype=np.uint64))

    bits = np.array([False, True, False, True, False], dtype=np.uint8)
    (plan_dir / "loss_mask.bin").write_bytes(
        np.packbits(bits, bitorder="little").tobytes()
    )

    plan = {
        "format": "balanced_chess_training_plan_v1",
        "reservoir": {"path": str(reservoir), "valid_games": 1, "valid_plies": 5},
        "profile": profile,
        "max_window_plies": 3,
        "selected_games_file": "selected_games.npy",
        "game_ply_offsets_file": "game_ply_offsets.npy",
        "loss_mask_file": "loss_mask.bin",
        "loss_mask_bitorder": "little",
        "report_file": "report.json",
        "budget": {"training_items": 1, "supervised_plies": 2},
    }
    (plan_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    report = {"sources": {"reservoir": {"size_bytes": 10}}}
    (plan_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return plan_dir, reservoir


class BalancedPlanReaderTests(unittest.TestCase):
    def test_reads_selected_window_and_bitpacked_loss_mask(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            plan_dir, reservoir = _write_tiny_plan(Path(raw_tmp))
            plan = BalancedTrainingPlan(
                plan_dir, expected_profile="encoder", reservoir_pgn=reservoir
            )
            self.assertEqual(len(plan), 1)
            row = plan.selected_game(0)
            self.assertEqual((int(row["window_start"]), int(row["window_stop"])), (1, 4))
            np.testing.assert_array_equal(
                plan.local_loss_mask(0), np.array([True, False, True])
            )

    def test_rejects_wrong_profile_and_wrong_reservoir(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            plan_dir, reservoir = _write_tiny_plan(root)
            with self.assertRaises(BalancedPlanError):
                BalancedTrainingPlan(plan_dir, expected_profile="decoder")

            wrong = root / "wrong.pgn"
            wrong.write_bytes(b"short")
            with self.assertRaises(BalancedPlanError):
                BalancedTrainingPlan(
                    plan_dir, expected_profile="encoder", reservoir_pgn=wrong
                )

    def test_decoder_ply_to_token_contract(self) -> None:
        self.assertEqual(decoder_token_slice_for_ply_window(10, 13), slice(10, 14))
        mask = decoder_label_mask_from_ply_mask(np.array([True, False, True]))
        np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


class EncoderCollateMaskTests(unittest.TestCase):
    @staticmethod
    def _item(loss_mask=None):
        item = {
            "valid": True,
            "game_idx": torch.tensor(7),
            "seq_len": torch.tensor(3),
            "labels64": torch.zeros((3, 64), dtype=torch.long),
            "legal_flat": torch.ones((3, 4096), dtype=torch.bool),
            "turn": torch.tensor([0, 1, 0]),
            "castling": torch.zeros((3, 4), dtype=torch.long),
            "ep_square": torch.full((3,), -1, dtype=torch.long),
            "move_from": torch.tensor([1, 2, 3]),
            "move_to": torch.tensor([2, 3, 4]),
            "move_promo": torch.zeros(3, dtype=torch.long),
        }
        if loss_mask is not None:
            item["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
        return item

    def test_planned_mask_is_padded_separately_from_attention(self) -> None:
        batch = CollateGameSequences()([self._item([True, False, True])])
        self.assertTrue(torch.equal(batch["attn_mask"], torch.tensor([[True, True, True]])))
        self.assertTrue(torch.equal(batch["loss_mask"], torch.tensor([[True, False, True]])))

    def test_legacy_batch_schema_remains_unmasked(self) -> None:
        batch = CollateGameSequences()([self._item()])
        self.assertNotIn("loss_mask", batch)


class DecoderCollateMaskTests(unittest.TestCase):
    @staticmethod
    def _item(loss_mask=None):
        item = {
            "valid": True,
            "move_ids": torch.tensor([5, 6, 7, 8], dtype=torch.long),
            "piece_ids": torch.tensor([4, 5, 6, 7], dtype=torch.long),
            "start_turn": torch.tensor(0, dtype=torch.long),
        }
        if loss_mask is not None:
            item["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
        return item

    def test_plan_masks_labels_but_keeps_causal_inputs_visible(self) -> None:
        batch = CollateLLMMoves(pad_id=0, piece_pad_id=0, max_len=8)(
            [self._item([True, False, True, False])]
        )
        self.assertTrue(torch.equal(batch["input_ids"], torch.tensor([[5, 6, 7, 8]])))
        self.assertTrue(torch.equal(batch["attention_mask"], torch.ones((1, 4), dtype=torch.bool)))
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([[6, -100, 8, -100]])))
        self.assertTrue(torch.equal(batch["piece_labels"], torch.tensor([[5, -100, 7, -100]])))

    def test_legacy_labels_preserve_historical_final_pad_target(self) -> None:
        batch = CollateLLMMoves(pad_id=0, piece_pad_id=0, max_len=8)([self._item()])
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([[6, 7, 8, 0]])))
        self.assertNotIn("loss_mask", batch)

    def test_mixed_planned_and_legacy_items_fail_fast(self) -> None:
        collate = CollateLLMMoves(pad_id=0, piece_pad_id=0, max_len=8)
        with self.assertRaises(ValueError):
            collate([self._item([True, True, True, False]), self._item()])


if __name__ == "__main__":
    unittest.main()
