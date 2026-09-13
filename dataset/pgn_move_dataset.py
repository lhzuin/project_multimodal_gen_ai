import io
import random
from typing import Any, Optional

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from dataset.balanced_training_plan import (
    BalancedPlanError,
    BalancedTrainingPlan,
    decoder_label_mask_from_ply_mask,
    decoder_token_slice_for_ply_window,
)
from dataset.offsets import load_pgn_offsets
from models.chess_tokenizer import ChessTokenizer


class PGNMoveDataset(Dataset):
    """Move-sequence dataset with optional TG2 balanced-plan support.

    Legacy mode (``balanced_plan_dir=None``) is intentionally unchanged: dataset
    length follows the PGN index and long games use a fresh random token crop.

    Balanced mode delegates game selection, human-ply window placement, and the
    supervision mask to a frozen preprocessing plan.  The returned ``loss_mask`` is
    aligned with next-token *label positions*: masked moves stay in ``move_ids`` and
    therefore remain visible as causal context.
    """

    def __init__(
        self,
        pgn_path: str,
        index_path: str,
        tokenizer_path: str,
        block_size: Optional[int] = None,
        strict: bool = True,
        seed: int = 0,
        balanced_plan_dir: Optional[str] = None,
    ):
        self.pgn_path = pgn_path
        self.index_path = index_path
        self.balanced_plan = (
            BalancedTrainingPlan(
                balanced_plan_dir,
                expected_profile="decoder",
                reservoir_pgn=pgn_path,
            )
            if balanced_plan_dir is not None
            else None
        )

        # The balanced plan already stores exact PGN byte offsets.  Avoid loading a
        # potentially multi-million-entry JSON index when it is not needed.
        if self.balanced_plan is None:
            self.index = load_pgn_offsets(index_path)
            self.games = self.index["games"]
        else:
            self.index = None
            self.games = None

        self.tok = ChessTokenizer(tokenizer_path)
        self.block_size = block_size
        self.strict = strict
        self.rng = random.Random(seed)
        self._fh = None

        if self.balanced_plan is not None and self.block_size is not None:
            max_window = self.balanced_plan.max_window_plies
            if max_window is not None and max_window + 1 > self.block_size:
                raise BalancedPlanError(
                    "decoder plan/context mismatch: planned windows allow "
                    f"{max_window} supervised plies, which require {max_window + 1} "
                    f"tokens, but block_size={self.block_size}"
                )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def __len__(self):
        if self.balanced_plan is not None:
            return len(self.balanced_plan)
        return len(self.games)

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.pgn_path, "rb")
        return self._fh

    def _read_game_bytes(self, start: int, end: int) -> bytes:
        fh = self._get_fh()
        fh.seek(start)
        return fh.read(end - start)

    @staticmethod
    def _parse_game(raw: bytes):
        txt = raw.decode("utf-8", errors="ignore")
        return chess.pgn.read_game(io.StringIO(txt))

    def _balanced_item(self, idx: int):
        row = self.balanced_plan.selected_game(idx)
        raw = self._read_game_bytes(int(row["pgn_start"]), int(row["pgn_end"]))
        game = self._parse_game(raw)
        if game is None:
            raise BalancedPlanError(
                f"planned decoder item {idx} no longer parses from the reservoir PGN"
            )

        try:
            full_move_ids, full_piece_ids = self.tok.encode_pgn_game(
                game, strict=self.strict
            )
        except (KeyError, RuntimeError) as exc:
            # The planner selected only valid PGNs, so an encoding failure here is a
            # contract/configuration error rather than a sample to silently skip.
            raise BalancedPlanError(
                f"failed to tokenize planned decoder item {idx}: {exc}"
            ) from exc

        if not full_move_ids or full_move_ids[0] != self.tok.bos_id:
            raise BalancedPlanError("decoder tokenizer did not emit the expected BOS token")
        if full_move_ids[-1] != self.tok.eos_id:
            raise BalancedPlanError("decoder tokenizer did not emit the expected EOS token")

        moves = list(game.mainline_moves())
        n_moves = len(moves)
        if n_moves != int(row["ply_count"]):
            raise BalancedPlanError(
                f"planned decoder item {idx} expected {int(row['ply_count'])} plies, "
                f"found {n_moves}; the plan and reservoir PGN do not match"
            )
        if len(full_move_ids) != n_moves + 2 or len(full_piece_ids) != n_moves + 2:
            raise BalancedPlanError("decoder tokenizer emitted an unexpected token/ply count")

        window_start = int(row["window_start"])
        window_stop = int(row["window_stop"])
        token_slice = decoder_token_slice_for_ply_window(window_start, window_stop)
        move_ids = full_move_ids[token_slice]
        piece_ids = full_piece_ids[token_slice]

        ply_loss_mask = self.balanced_plan.local_loss_mask(idx)
        label_loss_mask = decoder_label_mask_from_ply_mask(ply_loss_mask)
        if len(move_ids) != len(label_loss_mask):
            raise BalancedPlanError(
                "decoder token/loss-mask alignment failed; expected one predecessor "
                "token plus one token per planned target ply"
            )
        if self.block_size is not None and len(move_ids) > self.block_size:
            raise BalancedPlanError(
                f"planned decoder item has {len(move_ids)} tokens but block_size={self.block_size}"
            )

        # Match the legacy random-crop turn convention exactly.  If the first input is
        # BOS, no moves have been played.  Otherwise token index ``window_start`` is
        # move ply ``window_start - 1``, so replay that many moves before determining
        # the side associated with the first token position.
        ply_before = 0 if window_start == 0 else window_start - 1
        board = game.board()
        for move in moves[:ply_before]:
            board.push(move)
        start_turn = 0 if board.turn == chess.WHITE else 1

        return {
            "valid": True,
            "move_ids": torch.tensor(move_ids, dtype=torch.long),
            "piece_ids": torch.tensor(piece_ids, dtype=torch.long),
            "start_turn": torch.tensor(start_turn, dtype=torch.long),
            "loss_mask": torch.from_numpy(label_loss_mask.copy()).bool(),
        }

    def __getitem__(self, idx: int):
        if self.balanced_plan is not None:
            return self._balanced_item(idx)

        # ------------------------------
        # Legacy path: preserve behavior
        # ------------------------------
        g = self.games[idx]
        raw = self._read_game_bytes(g["start"], g["end"])
        txt = raw.decode("utf-8", errors="ignore")

        game = chess.pgn.read_game(io.StringIO(txt))
        if game is None:
            return {"valid": False}

        try:
            move_ids, piece_ids = self.tok.encode_pgn_game(game, strict=self.strict)
        except KeyError:
            print(f"Encoding error in game idx={idx}, skipping")
            return {"valid": False}

        # optional: sample a fixed-length window (block_size)
        if self.block_size is not None:
            # we need at least 2 tokens to make (x,y)
            if len(move_ids) < 2:
                return {"valid": False}
            if len(move_ids) > self.block_size:
                start = self.rng.randrange(0, len(move_ids) - self.block_size + 1)
                move_ids = move_ids[start:start + self.block_size]
                piece_ids = piece_ids[start:start + self.block_size]
            else:
                start = 0

        bos_id = self.tok.bos_id
        # global index of first move token (exclude BOS if present)
        global_first_token_idx = start
        if global_first_token_idx == 0 and len(move_ids) > 0 and move_ids[0] == bos_id:
            # window begins with BOS, first move token is next one in the window
            ply_before = 0
        else:
            # if the full stream had BOS, then token index 1 corresponds to ply 0.
            # So ply_before = (global_first_token_idx - 1) if BOS was in original stream.
            had_bos = (len(move_ids) > 0 and move_ids[0] == bos_id) or True  # encode_pgn_game adds BOS by default
            ply_before = max(0, global_first_token_idx - 1) if had_bos else global_first_token_idx

        board = game.board()
        moves = list(game.mainline_moves())
        # replay exactly ply_before moves
        for k in range(min(ply_before, len(moves))):
            board.push(moves[k])

        start_turn = 0 if board.turn == chess.WHITE else 1
        return {
            "valid": True,
            "move_ids": torch.tensor(move_ids, dtype=torch.long),
            "piece_ids": torch.tensor(piece_ids, dtype=torch.long),
            "start_turn": torch.tensor(start_turn, dtype=torch.long),
        }
