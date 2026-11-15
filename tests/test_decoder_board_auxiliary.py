"""Behavioral tests for board targets, normalized losses and decoder integration."""

import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import chess
import chess.pgn
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.nn import functional as F

from dataset.collate_llm import CollateLLMMoves
from dataset.collate_llm_board import CollateLLMBoardMoves
from dataset.offsets import build_pgn_offsets
from dataset.pgn_move_board_dataset import PGNMoveBoardDataset
from dataset.pgn_move_dataset import PGNMoveDataset
from dataset.renderer import board_to_grid_ids
from models.chess_decoder_board_player import ChessDecoderBoardPlayer
from models.chess_decoder_player import ChessDecoderPlayer
from train.decoder_board_objective import DecoderBoardObjective
from train.train_chessdecoder import Trainer, param_groups_for
from train.train_chessdecoder_board import BoardTrainer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = str(ROOT / "tokenizers/chess_uci_vocab.json")
PGN = '[Event "test"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *\n'


def tiny_model(cls=ChessDecoderBoardPlayer):
    return cls(
        tokenizer_path=TOKENIZER, model_dim=16, n_heads=2, n_layers=1,
        mlp_ratio=2, dropout=0, max_seq_len=16,
    )


def write_game(root, text=PGN):
    pgn, index = root / "games.pgn", root / "index.json"
    pgn.write_text(text)
    build_pgn_offsets(str(pgn), str(index))
    return dict(pgn_path=str(pgn), index_path=str(index), tokenizer_path=TOKENIZER)


def read_item(dataset):
    """Close worker-style cached handles when reading a one-off test sample."""
    try:
        return dataset[0]
    finally:
        if dataset._fh is not None:
            dataset._fh.close()
            dataset._fh = None


def write_plan(root, pgn, *, start=2, stop=5):
    """Build one real balanced sample with a deliberately masked middle target."""
    plan = root / "plan"
    plan.mkdir()
    dtype = np.dtype([
        ("game_id", "<u4"), ("pgn_start", "<u8"), ("pgn_end", "<u8"),
        ("ply_count", "<u4"), ("window_start", "<u4"), ("window_stop", "<u4"),
        ("anchor_ply", "<u4"), ("active_targets", "<u4"), ("game_mass", "<f8"),
    ])
    row = np.zeros(1, dtype=dtype)
    row[0] = (0, 0, pgn.stat().st_size, 6, start, stop, start, 2, 1)
    np.save(plan / "selected_games.npy", row)
    np.save(plan / "game_ply_offsets.npy", np.array([0, 6], dtype=np.uint64))
    mask = np.zeros(6, dtype=np.uint8)
    mask[start] = mask[stop - 1] = 1
    (plan / "loss_mask.bin").write_bytes(np.packbits(mask, bitorder="little").tobytes())
    (plan / "plan.json").write_text(json.dumps({
        "format": "balanced_chess_training_plan_v1", "profile": "decoder",
        "reservoir": {"path": str(pgn), "valid_games": 1, "valid_plies": 6},
        "max_window_plies": stop - start,
        "selected_games_file": "selected_games.npy",
        "game_ply_offsets_file": "game_ply_offsets.npy",
        "loss_mask_file": "loss_mask.bin", "loss_mask_bitorder": "little",
        "report_file": "report.json", "budget": {"training_items": 1, "supervised_plies": 2},
    }))
    (plan / "report.json").write_text(json.dumps({"sources": {"reservoir": {"size_bytes": pgn.stat().st_size}}}))
    return str(plan)


class BoardDatasetTests(unittest.TestCase):
    def test_full_game_labels_are_after_input_move_and_eos_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = write_game(Path(tmp))
            item = read_item(PGNMoveBoardDataset(**kwargs))
            legacy = read_item(PGNMoveDataset(**kwargs))
            for key in legacy:
                if isinstance(legacy[key], torch.Tensor):
                    torch.testing.assert_close(item[key], legacy[key])
            board = chess.Board()
            game = chess.pgn.read_game(io.StringIO(PGN))
            states = [torch.from_numpy(board_to_grid_ids(board).reshape(64))]
            for move in game.mainline_moves():
                board.push(move)
                states.append(torch.from_numpy(board_to_grid_ids(board).reshape(64)))
            torch.testing.assert_close(item["board_labels"][:-1], torch.stack(states))
            self.assertTrue((item["board_labels"][-1] == -100).all())

    def test_balanced_crop_replays_prefix_and_preserves_base_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = write_game(root)
            kwargs.update(balanced_plan_dir=write_plan(root, Path(kwargs["pgn_path"])), block_size=4)
            item = read_item(PGNMoveBoardDataset(**kwargs))
            base_item = read_item(PGNMoveDataset(**kwargs))
            batch = CollateLLMBoardMoves(0, 0, 4)([item])
            base_batch = CollateLLMMoves(0, 0, 4)([base_item])
            for key in base_batch:
                if isinstance(base_batch[key], torch.Tensor):
                    torch.testing.assert_close(batch[key], base_batch[key])
            board = chess.Board()
            board.push_uci("e2e4")
            board.push_uci("e7e5")
            torch.testing.assert_close(batch["board_labels"][0, 0], torch.from_numpy(board_to_grid_ids(board).reshape(64)))
            self.assertTrue((batch["board_labels"][0, 1] == -100).all())
            self.assertTrue((batch["board_labels"][0, -1] == -100).all())
            self.assertEqual(int((batch["board_labels"] != -100).sum()), 2 * 64)
            self.assertTrue(batch["attention_mask"].all())

    def test_legacy_random_crop_matches_absolute_token_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = PGNMoveBoardDataset(**write_game(Path(tmp)), block_size=3)
            with mock.patch.object(ds.rng, "randrange", return_value=3):
                item = read_item(ds)
            game = chess.pgn.read_game(io.StringIO(PGN))
            expected = ds._additional_item_fields(game, 0, 8)["board_labels"][3:6]
            torch.testing.assert_close(item["board_labels"], expected)

    def test_special_moves_and_custom_fen_use_true_chess_state(self):
        # Compare manually expected squares, independently of target generation.
        cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", {"e1": 0, "h1": 0, "g1": 6, "f1": 4}),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", {"e5": 0, "d5": 0, "d6": 1}),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", {"a7": 0, "a8": 5}),
            ("4k3/8/8/8/8/8/r7/R3K3 b - - 0 1", "a2a1", {"a2": 0, "a1": 10}),
        ]
        for fen, uci, expected in cases:
            with self.subTest(uci=uci), tempfile.TemporaryDirectory() as tmp:
                board = chess.Board(fen)
                game = chess.pgn.Game.from_board(board)
                game.add_variation(chess.Move.from_uci(uci))
                ds = PGNMoveBoardDataset(**write_game(Path(tmp), str(game)))
                target = read_item(ds)["board_labels"][1]
                for square, value in expected.items():
                    sq = chess.parse_square(square)
                    i = (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)
                    self.assertEqual(int(target[i]), value)

    def test_collate_padding_invalid_items_and_target_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = PGNMoveBoardDataset(**write_game(Path(tmp)))
            item = read_item(ds)
            short = {**item, "move_ids": item["move_ids"][:3], "piece_ids": item["piece_ids"][:3], "board_labels": item["board_labels"][:3]}
            collate = CollateLLMBoardMoves(0, 0, 16)
            batch = collate([item, {"valid": False}, short])
            self.assertTrue((batch["board_labels"][1, 3:] == -100).all())
            self.assertEqual(collate([{"valid": False}]), {"valid": False})
            with self.assertRaises(ValueError):
                collate([{**item, "board_labels": item["board_labels"][:-1]}])


class BoardObjectiveTests(unittest.TestCase):
    def inputs(self, uniform=False):
        create = torch.zeros if uniform else torch.randn
        out = {"logits": create(2, 3, 17, requires_grad=True), "piece_logits": create(2, 3, 10, requires_grad=True), "board_logits": create(2, 3, 64, 13, requires_grad=True)}
        return out, torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3, 64, dtype=torch.long)

    def test_uniform_predictions_have_unit_loss_and_seventy_thirty_contributions(self):
        values = DecoderBoardObjective()(*self.inputs(uniform=True))
        for key, expected in (("loss", 1), ("current_normalized", 1), ("board_normalized", 1), ("weighted_current", .7), ("weighted_board", .3)):
            self.assertAlmostEqual(float(values[key].detach()), expected, places=6)

    def test_exact_formula_preserves_original_move_piece_ratio(self):
        out, move, piece, board = self.inputs()
        values = DecoderBoardObjective()(out, move, piece, board)
        old = F.cross_entropy(out["logits"].reshape(-1, 17), move.reshape(-1)) + .3 * F.cross_entropy(out["piece_logits"].reshape(-1, 10), piece.reshape(-1))
        board_ce = F.cross_entropy(out["board_logits"].reshape(-1, 13), board.reshape(-1))
        expected = .7 * old / (math.log(17) + .3 * math.log(10)) + .3 * board_ce / math.log(13)
        torch.testing.assert_close(values["loss"], expected)

    def test_ignored_tokens_have_zero_gradient_and_padding_does_not_change_loss(self):
        out, move, piece, board = self.inputs()
        move[:, 1] = piece[:, 1] = -100
        objective = DecoderBoardObjective()
        full = objective(out, move, piece, board)["loss"]
        cropped = objective({k: v[:, [0, 2]] for k, v in out.items()}, move[:, [0, 2]], piece[:, [0, 2]], board[:, [0, 2]])["loss"]
        torch.testing.assert_close(full, cropped)
        full.backward()
        for logits in out.values():
            self.assertTrue((logits.grad[:, 1] == 0).all())
            self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_all_ignored_targets_produce_finite_differentiable_zero(self):
        out, move, piece, board = self.inputs()
        move.fill_(-100)
        piece.fill_(-100)
        board.fill_(-100)
        loss = DecoderBoardObjective()(out, move, piece, board)["loss"]
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue((out["board_logits"].grad == 0).all())

    def test_weight_endpoints_and_no_normalization_ablation(self):
        out, move, piece, board = self.inputs()
        values = DecoderBoardObjective(0, normalization="none")({k: v for k, v in out.items() if k != "board_logits"}, move, piece, board)
        torch.testing.assert_close(values["loss"], values["current_raw"])
        values = DecoderBoardObjective(1)(out, move, piece, board)
        torch.testing.assert_close(values["loss"], values["board_normalized"])

    def test_invalid_weights_and_shapes_fail_fast(self):
        for weight in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                DecoderBoardObjective(weight)
        with self.assertRaises(ValueError):
            DecoderBoardObjective(piece_loss_weight=-1)
        with self.assertRaises(ValueError):
            DecoderBoardObjective(normalization="dynamic")
        out, move, piece, board = self.inputs()
        with self.assertRaises(ValueError):
            DecoderBoardObjective()(out, move, piece, board[..., :63])


class BoardModelTests(unittest.TestCase):
    def test_auxiliary_model_shares_policy_features_and_probe_captures(self):
        model = tiny_model().eval()
        plain = tiny_model(ChessDecoderPlayer).eval()
        plain.load_state_dict(model.inference_state_dict(), strict=True)
        inputs = dict(input_ids=torch.tensor([[1, 2, 3]]),
                      piece_input_ids=torch.tensor([[1, 2, 3]]),
                      start_turn=torch.tensor([1]))
        with torch.no_grad():
            expected = plain(**inputs, capture_layers=(1,))
            actual = model(**inputs, capture_layers=(1,), return_board_logits=True)
        for key in ("logits", "piece_logits"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(actual["hidden_states"][1], expected["hidden_states"][1], rtol=0, atol=0)
        self.assertEqual(tuple(actual["board_logits"].shape), (1, 3, 64, 13))

    def test_original_inference_parity_and_policy_export(self):
        base, variant = tiny_model(ChessDecoderPlayer).eval(), tiny_model().eval()
        base.load_state_dict(variant.inference_state_dict(), strict=True)
        self.assertFalse(hasattr(base, "board_head"))
        inputs = {"input_ids": torch.tensor([[1, 5, 6]]), "piece_input_ids": torch.tensor([[1, 4, 4]]), "start_turn": torch.tensor([0])}
        with mock.patch.object(variant.board_head, "forward", wraps=variant.board_head.forward) as head:
            expected, actual = base(**inputs), variant(**inputs)
            head.assert_not_called()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertNotIn("board_logits", actual)
        base.load_state_dict(variant.inference_state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            base.load_state_dict(variant.state_dict(), strict=True)

    def test_training_and_validation_outputs_are_causal(self):
        model = tiny_model().train()
        tokens = torch.tensor([[1, 5, 6, 7]])
        out = model(tokens)
        self.assertEqual(out["board_logits"].shape, (1, 4, 64, 13))
        changed = tokens.clone()
        changed[0, -1] = 8
        torch.testing.assert_close(out["board_logits"][:, :-1], model(changed)["board_logits"][:, :-1])
        model.eval()
        self.assertIn("board_logits", model(tokens, return_board_logits=True))

    def test_parameter_groups_include_board_head_and_have_no_duplicates(self):
        model = tiny_model()
        model.set_trainable_groups(["decoder", "lm_head", "piece_head", "board_head"])
        groups = model.get_param_groups()
        params = [p for name in ("decoder", "lm_head", "piece_head", "board_head") for p in groups[name]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertTrue(all(p.requires_grad for p in model.board_head.parameters()))

    def test_board_loss_alone_trains_shared_decoder_and_auxiliary_head(self):
        model = tiny_model().train()
        out = model(torch.tensor([[1, 5, 6]]))
        labels = torch.ones(1, 3, dtype=torch.long)
        board = torch.zeros(1, 3, 64, dtype=torch.long)
        DecoderBoardObjective(board_loss_weight=1)(out, labels, labels, board)["loss"].backward()
        self.assertGreater(float(model.board_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.decoder.layers[0].linear1.weight.grad.abs().sum()), 0)
        self.assertEqual(float(model.piece_head.weight.grad.abs().sum()), 0)


class BoardTrainingTests(unittest.TestCase):
    def config(self, root, config_name="chess_decoder_player_balanced_board_v1"):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.1"):
            cfg = compose(config_name=config_name)
        # Unit tests have no running Hydra job; materialize its cwd resolver.
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("${hydra:runtime.cwd}", str(ROOT)))
        cfg.checkpoint_save_path = str(root / "variant.latest.pt")
        cfg.model.instance.tokenizer_path = TOKENIZER
        cfg.model.instance.model_dim = 16
        cfg.model.instance.n_heads = 2
        cfg.model.instance.n_layers = 1
        cfg.model.instance.dropout = 0
        cfg.model.instance.max_seq_len = 16
        cfg.log = False
        return cfg

    def test_real_optimizer_step_validation_metrics_and_full_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            kwargs = write_game(root)
            item = read_item(PGNMoveBoardDataset(**kwargs, balanced_plan_dir=write_plan(root, Path(kwargs["pgn_path"])), block_size=4))
            batch = CollateLLMBoardMoves(0, 0, 16)([item])
            trainer = BoardTrainer(cfg, None, torch.device("cpu"))
            trainer.init_model()
            trainer.stage_name = "stage1"
            trainer.model.set_trainable_groups(list(cfg.train_stages[0].groups))
            trainer.optimizer = torch.optim.AdamW(param_groups_for(trainer.model, cfg, cfg.train_stages[0]))
            trainer.train_loader = trainer.val_loader = [batch]
            before = trainer.model.board_head.weight.detach().clone()
            decoder_before = trainer.model.decoder.layers[0].linear1.weight.detach().clone()
            result = trainer.run_epoch("stage1")
            self.assertTrue(math.isfinite(result[0]))
            self.assertFalse(torch.equal(before, trainer.model.board_head.weight))
            self.assertFalse(torch.equal(decoder_before, trainer.model.decoder.layers[0].linear1.weight))
            validation = trainer.run_validation()
            self.assertTrue(math.isfinite(validation[0]))
            self.assertEqual(trainer._board_counts["square_total"], 128)
            self.assertEqual(trainer._board_counts["board_total"], 2)
            trainer.save_checkpoint(epoch_complete=False)
            cfg.checkpoint_load_path = "auto"
            resumed = BoardTrainer(cfg, None, torch.device("cpu"))
            resumed.init_model()
            resumed.optimizer = torch.optim.AdamW(param_groups_for(resumed.model, cfg, cfg.train_stages[0]))
            progress = resumed._restore_training_state("stage1", 0)
            self.assertEqual(progress[:2], (0, 1))
            self.assertEqual(resumed.global_step, trainer.global_step)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value)
            self.assertEqual(len(resumed.optimizer.state), len(trainer.optimizer.state))

    def test_default_initialization_is_from_scratch_and_zero_weight_skips_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            self.assertIsNone(cfg.checkpoint_load_path)
            self.assertNotIn("initial_weights_path", cfg)
            cfg.auxiliary_board.loss_weight = 0
            trainer = BoardTrainer(cfg, None, torch.device("cpu"))
            with mock.patch.object(trainer.checkpoints, "load") as load:
                trainer.init_model()
                load.assert_not_called()
            with mock.patch.object(trainer.model.board_head, "forward") as head:
                trainer._forward_model(input_ids=torch.tensor([[1, 5]]))
                head.assert_not_called()

    def test_original_trainer_loss_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp), "chess_decoder_player_balanced_v1")
            trainer = Trainer(cfg, None, torch.device("cpu"))
            out, move, piece, _ = BoardObjectiveTests().inputs()
            expected = F.cross_entropy(out["logits"].reshape(-1, 17), move.reshape(-1)) + .3 * F.cross_entropy(out["piece_logits"].reshape(-1, 10), piece.reshape(-1))
            trainer.model = mock.Mock(return_value=out)
            batch = {
                "labels": move, "piece_labels": piece,
                "input_ids": torch.ones_like(move), "piece_input_ids": torch.ones_like(piece),
                "attention_mask": torch.ones_like(move, dtype=torch.bool),
                "start_turn": torch.zeros(move.shape[0], dtype=torch.long),
            }
            loss, _, _, count = trainer._forward_loss_and_metrics(batch)
            torch.testing.assert_close(loss, expected)
            self.assertEqual(count, move.numel())

    def test_original_decoder_trains_and_evaluates_without_board_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root, "chess_decoder_player_balanced_v1")
            self.assertNotIn("auxiliary_board", cfg)
            batch = CollateLLMMoves(0, 0, 16)([read_item(PGNMoveDataset(**write_game(root)))])
            self.assertNotIn("board_labels", batch)
            trainer = Trainer(cfg, None, torch.device("cpu"))
            trainer.model = tiny_model(ChessDecoderPlayer)
            self.assertFalse(hasattr(trainer.model, "board_head"))
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.001)
            trainer.train_loader = trainer.val_loader = [batch]
            self.assertTrue(math.isfinite(trainer.run_epoch("original")[0]))
            self.assertTrue(math.isfinite(trainer.run_validation()[0]))


if __name__ == "__main__":
    unittest.main()
