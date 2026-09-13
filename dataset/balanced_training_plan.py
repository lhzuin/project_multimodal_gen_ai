"""Runtime reader for TG2 balanced-training plan artifacts.

The heavy preprocessing lives in :mod:`utils.balanced_training_plan`.  This module is
intentionally small and dependency-light: datasets use it only to read a frozen plan,
resolve the planned PGN byte range/window for an item, and recover the corresponding
loss mask.

A plan changes *which targets contribute loss*; it does not change model inputs.  In
particular, decoder moves with a zero loss bit remain visible as causal context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = "balanced_chess_training_plan_v1"
MASK_BITORDER = "little"


class BalancedPlanError(RuntimeError):
    """Raised when a frozen balancing plan is incompatible with the active dataset."""


class BalancedTrainingPlan:
    """Lazy, spawn-safe reader for a balanced-training plan directory.

    Parameters
    ----------
    plan_dir:
        Directory produced by ``python -m utils.balanced_training_plan``.
    expected_profile:
        Optional profile guard (``"encoder"`` or ``"decoder"``).
    reservoir_pgn:
        PGN that the dataset will read.  When the planner report contains the source
        file size, it is checked here because the stored byte offsets are meaningful
        only for the exact reservoir bytes (the file may be moved/renamed safely).
    """

    def __init__(
        self,
        plan_dir: str | Path,
        *,
        expected_profile: str | None = None,
        reservoir_pgn: str | Path | None = None,
    ) -> None:
        self.plan_dir = Path(plan_dir).expanduser().resolve()
        plan_path = self.plan_dir / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"balanced plan metadata not found: {plan_path}")

        with plan_path.open("r", encoding="utf-8") as fh:
            self.metadata: dict[str, Any] = json.load(fh)

        version = self.metadata.get("format")
        if version != FORMAT_VERSION:
            raise BalancedPlanError(
                f"unsupported balanced plan format {version!r}; expected {FORMAT_VERSION!r}"
            )

        self.profile = str(self.metadata.get("profile", ""))
        if expected_profile is not None and self.profile != expected_profile:
            raise BalancedPlanError(
                f"plan profile is {self.profile!r}, but dataset expects {expected_profile!r}"
            )

        self._selected_games_path = self.plan_dir / self.metadata.get(
            "selected_games_file", "selected_games.npy"
        )
        self._game_ply_offsets_path = self.plan_dir / self.metadata.get(
            "game_ply_offsets_file", "game_ply_offsets.npy"
        )
        self._loss_mask_path = self.plan_dir / self.metadata.get(
            "loss_mask_file", "loss_mask.bin"
        )

        for path in (
            self._selected_games_path,
            self._game_ply_offsets_path,
            self._loss_mask_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"balanced plan artifact not found: {path}")

        bitorder = self.metadata.get("loss_mask_bitorder", MASK_BITORDER)
        if bitorder != MASK_BITORDER:
            raise BalancedPlanError(
                f"unsupported loss-mask bitorder {bitorder!r}; expected {MASK_BITORDER!r}"
            )

        reservoir = self.metadata.get("reservoir", {})
        self.valid_games = int(reservoir["valid_games"])
        self.valid_plies = int(reservoir["valid_plies"])
        self.max_window_plies = self.metadata.get("max_window_plies")
        if self.max_window_plies is not None:
            self.max_window_plies = int(self.max_window_plies)

        expected_mask_bytes = (self.valid_plies + 7) // 8
        actual_mask_bytes = self._loss_mask_path.stat().st_size
        if actual_mask_bytes != expected_mask_bytes:
            raise BalancedPlanError(
                "loss_mask.bin size mismatch: "
                f"expected {expected_mask_bytes:,} bytes for {self.valid_plies:,} plies, "
                f"found {actual_mask_bytes:,}"
            )

        self._expected_reservoir_size = self._load_expected_reservoir_size()
        self.reservoir_pgn = (
            Path(reservoir_pgn).expanduser().resolve() if reservoir_pgn is not None else None
        )
        if self.reservoir_pgn is not None:
            self._validate_reservoir(self.reservoir_pgn)

        # NumPy memmaps are opened lazily.  Dropping them during pickling keeps the
        # object safe under DataLoader spawn semantics on macOS/Windows.
        self._selected_games: np.ndarray | None = None
        self._game_ply_offsets: np.ndarray | None = None
        self._loss_mask_bytes: np.memmap | None = None

        # Cheap structural validation without materializing the arrays.
        if len(self.selected_games) != int(self.metadata["budget"]["training_items"]):
            raise BalancedPlanError("selected_games.npy length disagrees with plan.json budget")
        if len(self.game_ply_offsets) != self.valid_games + 1:
            raise BalancedPlanError(
                "game_ply_offsets.npy must contain one prefix-sum boundary per valid game plus one"
            )
        if int(self.game_ply_offsets[-1]) != self.valid_plies:
            raise BalancedPlanError(
                "final game-ply prefix sum disagrees with the reservoir valid-ply count"
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_selected_games"] = None
        state["_game_ply_offsets"] = None
        state["_loss_mask_bytes"] = None
        return state

    def _load_expected_reservoir_size(self) -> int | None:
        report_name = self.metadata.get("report_file", "report.json")
        report_path = self.plan_dir / report_name
        if not report_path.is_file():
            return None
        try:
            with report_path.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
            return int(report["sources"]["reservoir"]["size_bytes"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _validate_reservoir(self, pgn_path: Path) -> None:
        if not pgn_path.is_file():
            raise FileNotFoundError(f"reservoir PGN not found: {pgn_path}")
        if self._expected_reservoir_size is not None:
            actual = pgn_path.stat().st_size
            if actual != self._expected_reservoir_size:
                raise BalancedPlanError(
                    "reservoir PGN size does not match the file used to build the plan: "
                    f"expected {self._expected_reservoir_size:,} bytes, found {actual:,}. "
                    "Do not use plan byte offsets with a different PGN."
                )

    @property
    def selected_games(self) -> np.ndarray:
        if self._selected_games is None:
            self._selected_games = np.load(
                self._selected_games_path, mmap_mode="r", allow_pickle=False
            )
        return self._selected_games

    @property
    def game_ply_offsets(self) -> np.ndarray:
        if self._game_ply_offsets is None:
            self._game_ply_offsets = np.load(
                self._game_ply_offsets_path, mmap_mode="r", allow_pickle=False
            )
        return self._game_ply_offsets

    @property
    def loss_mask_bytes(self) -> np.memmap:
        if self._loss_mask_bytes is None:
            self._loss_mask_bytes = np.memmap(
                self._loss_mask_path, dtype=np.uint8, mode="r"
            )
        return self._loss_mask_bytes

    def __len__(self) -> int:
        return int(len(self.selected_games))

    def selected_game(self, item_index: int) -> np.void:
        if item_index < 0 or item_index >= len(self):
            raise IndexError(item_index)
        return self.selected_games[item_index]

    def local_loss_mask(self, item_index: int) -> np.ndarray:
        """Return the bool loss mask for the selected item's planned ply window."""

        row = self.selected_game(item_index)
        game_id = int(row["game_id"])
        start = int(row["window_start"])
        stop = int(row["window_stop"])
        ply_count = int(row["ply_count"])

        if not (0 <= game_id < self.valid_games):
            raise BalancedPlanError(f"selected row has invalid game_id={game_id}")
        if not (0 <= start < stop <= ply_count):
            raise BalancedPlanError(
                f"invalid planned window [{start}, {stop}) for game with {ply_count} plies"
            )

        base = int(self.game_ply_offsets[game_id])
        indices = base + np.arange(start, stop, dtype=np.uint64)
        byte_indices = indices >> np.uint64(3)
        bit_indices = (indices & np.uint64(7)).astype(np.uint8)
        values = (self.loss_mask_bytes[byte_indices] >> bit_indices) & np.uint8(1)
        mask = np.asarray(values, dtype=bool)

        expected_active = int(row["active_targets"])
        actual_active = int(mask.sum())
        if actual_active != expected_active:
            raise BalancedPlanError(
                f"plan row expects {expected_active} active targets, recovered {actual_active}"
            )
        return mask


def decoder_token_slice_for_ply_window(window_start: int, window_stop: int) -> slice:
    """Map a human-ply window to the causal token slice used by the decoder.

    The tokenizer emits ``[BOS, move_0, ..., move_(n-1), EOS]``.  To supervise target
    plies ``[a, b)`` we need one predecessor/context token followed by those moves:
    full-token indices ``[a, b]`` inclusive, i.e. Python slice ``a:b+1``.
    """

    start = int(window_start)
    stop = int(window_stop)
    if start < 0 or stop <= start:
        raise ValueError(f"invalid ply window [{start}, {stop})")
    return slice(start, stop + 1)


def decoder_label_mask_from_ply_mask(ply_mask: np.ndarray) -> np.ndarray:
    """Align a per-ply plan mask with next-token loss positions.

    For ``L`` planned target plies the decoder receives ``L+1`` tokens: one predecessor
    token plus the ``L`` moves.  Label position ``j`` predicts target ply ``j`` in the
    planned window.  The final token has no planned next-move target and is always
    masked out.
    """

    source = np.asarray(ply_mask, dtype=bool)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("decoder ply loss mask must be a non-empty 1-D array")
    out = np.zeros(source.size + 1, dtype=bool)
    out[:-1] = source
    return out
