"""Build a fixed-compute, diversity-aware chess training plan from an expanded PGN.

The planner is intended for the TG2 late-game robustness experiment.  It leaves the
human move target unchanged and changes only *which human decisions contribute loss*.
The target distribution is a mixture

    q_lambda(i) = (1 - lambda) / N + lambda * q_bal(i)

where q_bal assigns equal mass to broad absolute-ply bands.  For early bands, where
board-state repetition is substantial, q_bal is also uniform over distinct states;
after ``state_balance_until`` it is uniform over occurrences inside the band.

The output is a training contract, not a rewritten PGN.  It contains:

* a fixed number of selected game items (controlling DataLoader/optimizer steps),
* one deterministic context window per selected game,
* a bit-packed per-ply loss mask (controlling which targets contribute loss), and
* audit reports describing the realized distribution.

For a causal decoder, moves whose loss bit is zero MUST remain in ``input_ids`` and
attention context; only their labels are ignored.  This preserves autoregressive state
tracking while reallocating supervision.

Run from the repository root, e.g.::

    python -m utils.balanced_training_plan \
        --reservoir-pgn processed_games/games_xl_expanded.pgn \
        --baseline-pgn processed_games/games_xl.pgn \
        --profile decoder \
        --lambda-balance 0.5 \
        --max-window-plies 255 \
        --batch-size 32 \
        --epochs 25 \
        --output-dir processed_games/balanced/decoder_xl

The implementation is deliberately streaming and disk-backed.  At no point does it
store a Python dictionary containing one entry for every board state or every ply.
Early state identities are partitioned by a 128-bit deterministic fingerprint and
counted one partition at a time.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "balanced_chess_training_plan_v1"
DEFAULT_BIN_STARTS = (0, 20, 40, 60, 80, 100)
DEFAULT_STATE_BALANCE_UNTIL = 40
DEFAULT_PARTITIONS = 256
DEFAULT_PROGRESS_SECONDS = 20.0
MASK_BITORDER = "little"
UINT64_MASK = (1 << 64) - 1

# On-disk temporary record for an early-state occurrence.
# h1/h2 = 128-bit state fingerprint, game = valid-reservoir game id,
# ply = absolute target ply, band = absolute-ply band id.
EARLY_RECORD_STRUCT = struct.Struct("<QQIHB")
EARLY_RECORD_DTYPE = np.dtype(
    [
        ("h1", "<u8"),
        ("h2", "<u8"),
        ("game", "<u4"),
        ("ply", "<u2"),
        ("band", "u1"),
    ],
    align=False,
)
assert EARLY_RECORD_DTYPE.itemsize == EARLY_RECORD_STRUCT.size == 23

# Valid reservoir game table.  Store true byte offsets so the result plugs directly
# into the project's existing binary PGN readers/index convention.  Keeping both start
# and end means a downstream Dataset can read exactly one game without rescanning.
GAME_RECORD_STRUCT = struct.Struct("<QQIQQ")
GAME_RECORD_DTYPE = np.dtype(
    [
        ("pgn_start", "<u8"),
        ("pgn_end", "<u8"),
        ("ply_count", "<u4"),
        ("fingerprint_hi", "<u8"),
        ("fingerprint_lo", "<u8"),
    ],
    align=False,
)
assert GAME_RECORD_DTYPE.itemsize == GAME_RECORD_STRUCT.size == 36

SELECTED_GAME_DTYPE = np.dtype(
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
    ],
    align=False,
)


LOGGER = logging.getLogger("balanced_training_plan")


class PlanError(RuntimeError):
    """Raised when the requested training contract cannot be realized safely."""


@dataclass(frozen=True)
class FileIdentity:
    path: str
    size_bytes: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileIdentity":
        stat = path.stat()
        return cls(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


@dataclass(frozen=True)
class Budget:
    target_games: int
    target_plies: int
    source: str
    baseline_valid_games: int | None = None
    baseline_raw_plies: int | None = None
    baseline_effective_plies: int | None = None
    baseline_train_fraction: float | None = None


@dataclass(frozen=True)
class PlannerConfig:
    reservoir_pgn: Path
    output_dir: Path
    profile: str
    lambda_balance: float
    bin_starts: tuple[int, ...]
    state_balance_until: int
    seed: int
    max_window_plies: int | None
    anchor_policy: str
    partitions: int
    target_games: int | None
    target_plies: int | None
    baseline_pgn: Path | None
    baseline_train_fraction: float
    exclude_pgns: tuple[Path, ...]
    batch_size: int | None
    epochs: int | None
    gradient_accumulation_steps: int
    drop_last: bool
    overwrite: bool
    keep_intermediate: bool
    progress_seconds: float

    def validate(self) -> None:
        if self.profile not in {"encoder", "decoder"}:
            raise ValueError("profile must be 'encoder' or 'decoder'")
        if not (0.0 <= self.lambda_balance <= 1.0):
            raise ValueError("lambda_balance must lie in [0, 1]")
        if not self.bin_starts or self.bin_starts[0] != 0:
            raise ValueError("bin_starts must begin at 0")
        if any(b <= a for a, b in zip(self.bin_starts, self.bin_starts[1:])):
            raise ValueError("bin_starts must be strictly increasing")
        if self.state_balance_until not in self.bin_starts:
            raise ValueError(
                "state_balance_until must be one of the band boundaries in --bins; "
                "this keeps the statistical contract unambiguous"
            )
        if self.state_balance_until < 0 or self.state_balance_until > 65535:
            raise ValueError("state_balance_until must be between 0 and 65535")
        if self.partitions <= 0 or self.partitions & (self.partitions - 1):
            raise ValueError("partitions must be a positive power of two")
        if self.partitions > 65536:
            raise ValueError("partitions is unreasonably large")
        if self.max_window_plies is not None and self.max_window_plies <= 0:
            raise ValueError("max_window_plies must be positive")
        if self.anchor_policy not in {"center", "right"}:
            raise ValueError("anchor_policy must be 'center' or 'right'")
        if self.target_games is not None and self.target_games <= 0:
            raise ValueError("target_games must be positive")
        if self.target_plies is not None and self.target_plies <= 0:
            raise ValueError("target_plies must be positive")
        if not (0.0 < self.baseline_train_fraction <= 1.0):
            raise ValueError("baseline_train_fraction must lie in (0, 1]")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.progress_seconds <= 0:
            raise ValueError("progress_seconds must be positive")
        if self.target_games is None or self.target_plies is None:
            if self.baseline_pgn is None:
                raise ValueError(
                    "provide --baseline-pgn, or provide both --target-games and --target-plies"
                )


@dataclass
class ScanStats:
    valid_games: int = 0
    raw_plies: int = 0
    effective_plies: int = 0
    parse_errors: int = 0
    zero_ply_games: int = 0
    excluded_games: int = 0


@dataclass
class StateAudit:
    selected_occurrences: list[int]
    selected_unique_states: list[int]


class ProgressMeter:
    """Rate-limited progress/ETA logger based on a monotonically increasing total."""

    def __init__(self, label: str, total: int | None, interval: float) -> None:
        self.label = label
        self.total = total
        self.interval = interval
        self.started = time.monotonic()
        self.last_log = self.started

    def maybe_log(self, current: int, *, extra: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_log < self.interval:
            return
        elapsed = max(now - self.started, 1e-9)
        rate = current / elapsed
        if self.total and self.total > 0:
            frac = min(max(current / self.total, 0.0), 1.0)
            eta = (self.total - current) / rate if rate > 0 else float("inf")
            base = (
                f"{self.label}: {100.0 * frac:6.2f}% "
                f"({current:,}/{self.total:,}), {rate:,.1f}/s, ETA {_format_seconds(eta)}"
            )
        else:
            base = f"{self.label}: {current:,}, {rate:,.1f}/s"
        if extra:
            base += f" | {extra}"
        LOGGER.info(base)
        self.last_log = now


class PartitionedEarlyWriter:
    """Buffered writer for compact early-state records partitioned by hash prefix."""

    def __init__(
        self,
        directory: Path,
        partitions: int,
        *,
        flush_bytes_per_partition: int = 256 * 1024,
    ) -> None:
        self.directory = directory
        self.partitions = partitions
        self.flush_bytes_per_partition = flush_bytes_per_partition
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffers = [bytearray() for _ in range(partitions)]
        self.records = 0

    def _path(self, partition: int) -> Path:
        return self.directory / f"part_{partition:04d}.bin"

    def append(self, h1: int, h2: int, game_id: int, ply: int, band: int) -> None:
        if game_id >= 2**32:
            raise PlanError("more than 2^32 valid games are not supported")
        if ply >= 2**16:
            raise PlanError(
                f"encountered a game with ply >= 65536; increase the early-record ply field"
            )
        if band >= 2**8:
            raise PlanError("more than 255 absolute-ply bands are not supported")
        partition = h1 & (self.partitions - 1)
        buf = self.buffers[partition]
        buf.extend(EARLY_RECORD_STRUCT.pack(h1, h2, game_id, ply, band))
        self.records += 1
        if len(buf) >= self.flush_bytes_per_partition:
            self._flush_one(partition)

    def _flush_one(self, partition: int) -> None:
        buf = self.buffers[partition]
        if not buf:
            return
        with self._path(partition).open("ab") as fh:
            fh.write(buf)
        buf.clear()

    def close(self) -> None:
        for partition in range(self.partitions):
            self._flush_one(partition)


class PackedBitset:
    """Read-only helper for ``loss_mask.bin`` files emitted by this planner."""

    def __init__(self, path: str | Path, n_bits: int) -> None:
        self.path = Path(path)
        self.n_bits = int(n_bits)
        expected = (self.n_bits + 7) // 8
        actual = self.path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"bitset size mismatch: expected {expected} bytes for {n_bits} bits, got {actual}"
            )
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def get(self, indices: np.ndarray | Sequence[int] | int) -> np.ndarray | bool:
        scalar = np.isscalar(indices)
        idx = np.asarray([indices] if scalar else indices, dtype=np.uint64)
        if idx.size and int(idx.max()) >= self.n_bits:
            raise IndexError("bit index is outside the plan")
        values = (self._data[idx >> 3] >> (idx & 7).astype(np.uint8)) & np.uint8(1)
        if scalar:
            return bool(values[0])
        return values.astype(bool, copy=False)


class BalancedPlanReader:
    """Small integration helper that exposes the training-plan contract."""

    def __init__(self, plan_dir: str | Path) -> None:
        self.plan_dir = Path(plan_dir)
        with (self.plan_dir / "plan.json").open("r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        if self.metadata.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported plan format {self.metadata.get('format')!r}; expected {FORMAT_VERSION!r}"
            )
        self.selected_games = np.load(self.plan_dir / "selected_games.npy", mmap_mode="r")
        self.game_ply_offsets = np.load(self.plan_dir / "game_ply_offsets.npy", mmap_mode="r")
        self.loss_mask = PackedBitset(
            self.plan_dir / "loss_mask.bin",
            int(self.metadata["reservoir"]["valid_plies"]),
        )

    def __len__(self) -> int:
        return int(len(self.selected_games))

    def local_loss_mask(self, item_index: int) -> np.ndarray:
        row = self.selected_games[item_index]
        game_id = int(row["game_id"])
        start = int(row["window_start"])
        stop = int(row["window_stop"])
        flat = int(self.game_ply_offsets[game_id]) + np.arange(start, stop, dtype=np.uint64)
        return np.asarray(self.loss_mask.get(flat), dtype=bool)


# ---------------------------------------------------------------------------
# Deterministic hashing / sampling helpers
# ---------------------------------------------------------------------------


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64).copy()
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x


def _splitmix64_scalar(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & UINT64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (x ^ (x >> 31)) & UINT64_MASK


def _uniform_scalar(seed: int, game_id: int, stream: int) -> float:
    x = (
        (seed & UINT64_MASK)
        ^ ((game_id + 1) * 0xD6E8FEB86659FD93 & UINT64_MASK)
        ^ ((stream + 1) * 0xA5A3564E27F8862B & UINT64_MASK)
    )
    z = _splitmix64_scalar(x)
    # Open interval (0, 1), using 53 high-quality bits.
    return ((z >> 11) + 0.5) * (1.0 / (1 << 53))


def _uniform_for_game_ply(game_ids: np.ndarray, ply: int, seed: int, stream: int) -> np.ndarray:
    gid = np.asarray(game_ids, dtype=np.uint64)
    ply_key = np.uint64(((ply + 1) * 0x9E3779B97F4A7C15) & UINT64_MASK)
    stream_key = np.uint64(((stream + 1) * 0xA5A3564E27F8862B) & UINT64_MASK)
    x = (
        np.uint64(seed & UINT64_MASK)
        ^ ((gid + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93))
        ^ ply_key
        ^ stream_key
    )
    z = _splitmix64_np(x)
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def _state_fingerprint(board: Any, chess_module: Any) -> tuple[int, int]:
    """Return a stable 128-bit fingerprint of the first four FEN state fields.

    We encode public python-chess board attributes instead of allocating a FEN string at
    every early ply.  The payload determines piece placement, side to move, castling
    rights, and the *legal* en-passant square, matching the project's historical use of
    the first four fields of ``Board.fen()`` under python-chess's default legal-EP mode.
    """

    legal_ep = board.ep_square if board.ep_square is not None and board.has_legal_en_passant() else 64
    payload = struct.pack(
        "<8Q?QB",
        int(board.pawns),
        int(board.knights),
        int(board.bishops),
        int(board.rooks),
        int(board.queens),
        int(board.kings),
        int(board.occupied_co[chess_module.WHITE]),
        int(board.occupied_co[chess_module.BLACK]),
        bool(board.turn),
        int(board.clean_castling_rights()),
        int(legal_ep),
    )
    digest = hashlib.blake2b(payload, digest_size=16, person=b"ch-state-bal-v1").digest()
    return struct.unpack("<QQ", digest)


def _game_fingerprint(game: Any, moves: Sequence[Any] | None = None) -> tuple[int, int]:
    """Fingerprint a game by initial position plus its mainline UCI move sequence."""

    h = hashlib.blake2b(digest_size=16, person=b"ch-game-bal-v1")
    initial_fen = game.board().fen()
    h.update(initial_fen.encode("ascii", errors="strict"))
    h.update(b"\0")
    iterable = moves if moves is not None else game.mainline_moves()
    for move in iterable:
        h.update(move.uci().encode("ascii"))
        h.update(b" ")
    return struct.unpack("<QQ", h.digest())


def _fingerprint_bytes(pair: tuple[int, int]) -> bytes:
    return struct.pack("<QQ", pair[0], pair[1])


# ---------------------------------------------------------------------------
# Basic distribution utilities
# ---------------------------------------------------------------------------


def parse_bin_starts(value: str) -> tuple[int, ...]:
    try:
        starts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bins must be comma-separated integers") from exc
    if not starts:
        raise argparse.ArgumentTypeError("--bins cannot be empty")
    return starts


def band_index(ply: int, starts: Sequence[int]) -> int:
    return int(np.searchsorted(np.asarray(starts), ply, side="right") - 1)


def _band_bounds(starts: Sequence[int], band: int) -> tuple[int, int | None]:
    start = int(starts[band])
    stop = int(starts[band + 1]) if band + 1 < len(starts) else None
    return start, stop


def _count_interval_for_lengths(lengths: np.ndarray, start: int, stop: int | None) -> np.ndarray:
    values = np.asarray(lengths, dtype=np.int64)
    if stop is None:
        return np.maximum(values - start, 0)
    return np.maximum(np.minimum(values, stop) - start, 0)


def _interval_overlap_counts(
    window_starts: np.ndarray,
    window_stops: np.ndarray,
    start: int,
    stop: int | None,
) -> np.ndarray:
    lo = np.maximum(window_starts, start)
    hi = window_stops if stop is None else np.minimum(window_stops, stop)
    return np.maximum(hi - lo, 0)


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# PGN scanning
# ---------------------------------------------------------------------------


def _import_chess() -> tuple[Any, Any]:
    try:
        import chess  # type: ignore
        import chess.pgn  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "python-chess is required to build a plan. Install the same dependency used by the project."
        ) from exc
    return chess, chess.pgn


def _quiet_python_chess() -> None:
    # python-chess logs malformed PGNs; the planner records/skips them itself.
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)


def _iter_pgn_game_bytes(path: Path) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(start_byte, end_byte, raw_game)`` using the repository's PGN contract.

    ``dataset.offsets.build_pgn_offsets`` identifies game starts by binary lines that
    begin with ``[Event ``.  Reusing that convention here is important: offsets emitted
    by this planner can be consumed directly by ``PGNMoveDataset`` and the encoder's
    binary random-access readers, including PGNs containing non-ASCII header text.

    Only one game's bytes are retained at a time.  Text before the first ``[Event``
    header is ignored, matching the project's offset-index behavior.
    """

    with path.open("rb") as source:
        start: int | None = None
        chunks: list[bytes] = []
        while True:
            line_start = source.tell()
            line = source.readline()
            if not line:
                if start is not None:
                    yield start, source.tell(), b"".join(chunks)
                return

            if line.startswith(b"[Event "):
                if start is not None:
                    yield start, line_start, b"".join(chunks)
                start = line_start
                chunks = [line]
            elif start is not None:
                chunks.append(line)


def _parse_game_bytes(raw: bytes, chess_pgn: Any) -> Any:
    """Parse one indexed PGN game with the same UTF-8 tolerance as project readers."""

    return chess_pgn.read_game(io.StringIO(raw.decode("utf-8", errors="ignore")))


def _build_exclusion_set(paths: Sequence[Path], progress_seconds: float) -> set[bytes]:
    if not paths:
        return set()
    _, chess_pgn = _import_chess()
    _quiet_python_chess()
    excluded: set[bytes] = set()
    for path in paths:
        path = path.resolve()
        meter = ProgressMeter(f"Index exclusions {path.name}", path.stat().st_size, progress_seconds)
        for _, end, raw in _iter_pgn_game_bytes(path):
            try:
                game = _parse_game_bytes(raw, chess_pgn)
            except Exception:
                meter.maybe_log(end, extra=f"fingerprints={len(excluded):,}")
                continue
            if game is None or game.errors:
                meter.maybe_log(end, extra=f"fingerprints={len(excluded):,}")
                continue
            moves = list(game.mainline_moves())
            if moves:
                excluded.add(_fingerprint_bytes(_game_fingerprint(game, moves)))
            meter.maybe_log(end, extra=f"fingerprints={len(excluded):,}")
        meter.maybe_log(path.stat().st_size, extra=f"fingerprints={len(excluded):,}", force=True)
    return excluded


def _scan_baseline(
    path: Path,
    *,
    max_window_plies: int | None,
    progress_seconds: float,
) -> ScanStats:
    _, chess_pgn = _import_chess()
    _quiet_python_chess()
    stats = ScanStats()
    file_size = path.stat().st_size
    meter = ProgressMeter(f"Scan baseline {path.name}", file_size, progress_seconds)

    for _, end, raw in _iter_pgn_game_bytes(path):
        try:
            game = _parse_game_bytes(raw, chess_pgn)
        except Exception:
            stats.parse_errors += 1
            meter.maybe_log(end, extra=f"games={stats.valid_games:,}, bad={stats.parse_errors:,}")
            continue
        if game is None or game.errors:
            stats.parse_errors += 1
            meter.maybe_log(end, extra=f"games={stats.valid_games:,}, bad={stats.parse_errors:,}")
            continue
        plies = sum(1 for _ in game.mainline_moves())
        if plies <= 0:
            stats.zero_ply_games += 1
            continue
        stats.valid_games += 1
        stats.raw_plies += plies
        stats.effective_plies += min(plies, max_window_plies) if max_window_plies else plies
        meter.maybe_log(
            end,
            extra=f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, bad={stats.parse_errors:,}",
        )
    meter.maybe_log(
        file_size,
        extra=f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, bad={stats.parse_errors:,}",
        force=True,
    )
    return stats

def _resolve_budget(config: PlannerConfig) -> tuple[Budget, ScanStats | None]:
    baseline_stats: ScanStats | None = None
    if config.baseline_pgn is not None:
        baseline_stats = _scan_baseline(
            config.baseline_pgn,
            max_window_plies=config.max_window_plies,
            progress_seconds=config.progress_seconds,
        )

    if config.target_games is not None:
        games = config.target_games
        game_source = "explicit"
    else:
        assert baseline_stats is not None
        games = max(1, int(round(baseline_stats.valid_games * config.baseline_train_fraction)))
        game_source = "baseline"

    if config.target_plies is not None:
        plies = config.target_plies
        ply_source = "explicit"
    else:
        assert baseline_stats is not None
        plies = max(1, int(round(baseline_stats.effective_plies * config.baseline_train_fraction)))
        ply_source = "baseline_effective"

    return (
        Budget(
            target_games=games,
            target_plies=plies,
            source=f"games:{game_source};plies:{ply_source}",
            baseline_valid_games=baseline_stats.valid_games if baseline_stats else None,
            baseline_raw_plies=baseline_stats.raw_plies if baseline_stats else None,
            baseline_effective_plies=baseline_stats.effective_plies if baseline_stats else None,
            baseline_train_fraction=config.baseline_train_fraction if baseline_stats else None,
        ),
        baseline_stats,
    )


def _scan_reservoir(
    config: PlannerConfig,
    intermediate_dir: Path,
    exclusion_set: set[bytes],
) -> tuple[Path, ScanStats, int]:
    chess, chess_pgn = _import_chess()
    _quiet_python_chess()

    games_raw = intermediate_dir / "reservoir_games.raw"
    partitions_dir = intermediate_dir / "early_partitions"
    early_writer = PartitionedEarlyWriter(partitions_dir, config.partitions)
    stats = ScanStats()
    file_size = config.reservoir_pgn.stat().st_size
    meter = ProgressMeter(
        f"Scan reservoir {config.reservoir_pgn.name}", file_size, config.progress_seconds
    )

    with games_raw.open("wb") as game_table:
        for pgn_start, pgn_end, raw in _iter_pgn_game_bytes(config.reservoir_pgn):
            try:
                game = _parse_game_bytes(raw, chess_pgn)
            except Exception:
                stats.parse_errors += 1
                meter.maybe_log(
                    pgn_end,
                    extra=(
                        f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, "
                        f"bad={stats.parse_errors:,}, excluded={stats.excluded_games:,}"
                    ),
                )
                continue
            if game is None or game.errors:
                stats.parse_errors += 1
                meter.maybe_log(
                    pgn_end,
                    extra=(
                        f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, "
                        f"bad={stats.parse_errors:,}, excluded={stats.excluded_games:,}"
                    ),
                )
                continue

            moves = list(game.mainline_moves())
            ply_count = len(moves)
            if ply_count <= 0:
                stats.zero_ply_games += 1
                continue

            game_fp = _game_fingerprint(game, moves)
            if _fingerprint_bytes(game_fp) in exclusion_set:
                stats.excluded_games += 1
                continue

            game_id = stats.valid_games
            if game_id >= 2**32:
                raise PlanError("more than 2^32 valid games are not supported")
            if ply_count >= 2**32:
                raise PlanError("a game has more than 2^32 plies")

            # Replay only the early portion for state-frequency correction.  The target
            # at ply t is the move played *from* the board before move t.
            if config.state_balance_until > 0:
                board = game.board()
                upto = min(ply_count, config.state_balance_until)
                for ply in range(upto):
                    h1, h2 = _state_fingerprint(board, chess)
                    b = band_index(ply, config.bin_starts)
                    early_writer.append(h1, h2, game_id, ply, b)
                    board.push(moves[ply])

            game_table.write(
                GAME_RECORD_STRUCT.pack(
                    int(pgn_start),
                    int(pgn_end),
                    int(ply_count),
                    int(game_fp[0]),
                    int(game_fp[1]),
                )
            )
            stats.valid_games += 1
            stats.raw_plies += ply_count
            stats.effective_plies += (
                min(ply_count, config.max_window_plies)
                if config.max_window_plies
                else ply_count
            )
            meter.maybe_log(
                pgn_end,
                extra=(
                    f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, "
                    f"bad={stats.parse_errors:,}, excluded={stats.excluded_games:,}"
                ),
            )

    early_writer.close()
    meter.maybe_log(
        file_size,
        extra=(
            f"games={stats.valid_games:,}, plies={stats.raw_plies:,}, "
            f"bad={stats.parse_errors:,}, excluded={stats.excluded_games:,}, "
            f"early_records={early_writer.records:,}"
        ),
        force=True,
    )
    return games_raw, stats, early_writer.records


# ---------------------------------------------------------------------------
# Early-state frequency processing
# ---------------------------------------------------------------------------


def _early_band_ids(config: PlannerConfig) -> tuple[int, ...]:
    return tuple(
        b
        for b, start in enumerate(config.bin_starts)
        if start < config.state_balance_until
    )


def _sort_partition_in_place(path: Path) -> np.ndarray:
    records = np.fromfile(path, dtype=EARLY_RECORD_DTYPE)
    if records.size:
        records.sort(order=["band", "h1", "h2"], kind="quicksort")
        tmp = path.with_suffix(path.suffix + ".sorted.tmp")
        records.tofile(tmp)
        os.replace(tmp, path)
    return records


def _group_band_records(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return group-start indices and occurrence counts for sorted state hashes."""
    n = len(sub)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (sub["h1"][1:] != sub["h1"][:-1]) | (
        sub["h2"][1:] != sub["h2"][:-1]
    )
    starts = np.flatnonzero(change).astype(np.int64, copy=False)
    counts = np.diff(np.append(starts, n)).astype(np.int64, copy=False)
    return starts, counts


def _process_early_partitions_first_pass(
    config: PlannerConfig,
    intermediate_dir: Path,
    num_games: int,
) -> tuple[np.memmap, np.ndarray, list[Counter[int]]]:
    early_bands = _early_band_ids(config)
    early_col = {band: col for col, band in enumerate(early_bands)}
    invfreq_path = intermediate_dir / "early_invfreq_sum.npy"
    invfreq = np.lib.format.open_memmap(
        invfreq_path,
        mode="w+",
        dtype=np.float64,
        shape=(num_games, len(early_bands)),
    )
    invfreq[:] = 0.0
    unique_states = np.zeros(len(config.bin_starts), dtype=np.int64)
    states_by_frequency = [Counter() for _ in config.bin_starts]
    partitions_dir = intermediate_dir / "early_partitions"
    meter = ProgressMeter("Count early state frequencies", config.partitions, config.progress_seconds)

    for partition in range(config.partitions):
        path = partitions_dir / f"part_{partition:04d}.bin"
        if not path.exists() or path.stat().st_size == 0:
            meter.maybe_log(partition + 1)
            continue
        records = _sort_partition_in_place(path)
        bands = records["band"]
        for band in early_bands:
            left = int(np.searchsorted(bands, band, side="left"))
            right = int(np.searchsorted(bands, band, side="right"))
            if left == right:
                continue
            sub = records[left:right]
            starts, counts = _group_band_records(sub)
            unique_states[band] += len(starts)
            freq_values, n_states = np.unique(counts, return_counts=True)
            states_by_frequency[band].update(
                {int(freq): int(num) for freq, num in zip(freq_values, n_states)}
            )

            freq_per_occurrence = np.repeat(counts, counts).astype(np.float64, copy=False)
            contributions = 1.0 / freq_per_occurrence
            np.add.at(invfreq[:, early_col[band]], sub["game"], contributions)
        del records
        meter.maybe_log(partition + 1)

    invfreq.flush()
    meter.maybe_log(config.partitions, force=True)
    return invfreq, unique_states, states_by_frequency


def _fill_selected_early_frequencies(
    config: PlannerConfig,
    intermediate_dir: Path,
    selected_game_ids: np.ndarray,
    num_games: int,
) -> np.memmap:
    rank = np.full(num_games, -1, dtype=np.int32)
    rank[selected_game_ids] = np.arange(len(selected_game_ids), dtype=np.int32)
    path = intermediate_dir / "selected_early_frequency.npy"
    freq = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint32,
        shape=(len(selected_game_ids), config.state_balance_until),
    )
    freq[:] = 0

    early_bands = _early_band_ids(config)
    partitions_dir = intermediate_dir / "early_partitions"
    meter = ProgressMeter("Materialize selected early frequencies", config.partitions, config.progress_seconds)
    for partition in range(config.partitions):
        part = partitions_dir / f"part_{partition:04d}.bin"
        if not part.exists() or part.stat().st_size == 0:
            meter.maybe_log(partition + 1)
            continue
        records = np.fromfile(part, dtype=EARLY_RECORD_DTYPE)
        bands = records["band"]
        for band in early_bands:
            left = int(np.searchsorted(bands, band, side="left"))
            right = int(np.searchsorted(bands, band, side="right"))
            if left == right:
                continue
            sub = records[left:right]
            _, counts = _group_band_records(sub)
            freq_per_occurrence = np.repeat(counts, counts).astype(np.uint32, copy=False)
            selected_ranks = rank[sub["game"]]
            keep = selected_ranks >= 0
            if np.any(keep):
                freq[selected_ranks[keep], sub["ply"][keep]] = freq_per_occurrence[keep]
        del records
        meter.maybe_log(partition + 1)
    freq.flush()
    meter.maybe_log(config.partitions, force=True)
    return freq


# ---------------------------------------------------------------------------
# Game masses, game selection, and windows
# ---------------------------------------------------------------------------


def _load_game_table(raw_path: Path, output_dir: Path) -> np.ndarray:
    size = raw_path.stat().st_size
    if size % GAME_RECORD_DTYPE.itemsize:
        raise PlanError("temporary game table is truncated or corrupt")
    games = np.fromfile(raw_path, dtype=GAME_RECORD_DTYPE)
    np.save(output_dir / "reservoir_games.npy", games, allow_pickle=False)
    return games


def _write_game_ply_offsets(games: np.ndarray, output_dir: Path) -> np.memmap:
    path = output_dir / "game_ply_offsets.npy"
    offsets = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint64, shape=(len(games) + 1,)
    )
    offsets[0] = 0
    np.cumsum(games["ply_count"].astype(np.uint64), out=offsets[1:])
    offsets.flush()
    return offsets


def _band_counts_from_lengths(lengths: np.ndarray, bin_starts: Sequence[int]) -> np.ndarray:
    totals = np.zeros(len(bin_starts), dtype=np.int64)
    for band in range(len(bin_starts)):
        start, stop = _band_bounds(bin_starts, band)
        totals[band] = int(_count_interval_for_lengths(lengths, start, stop).sum(dtype=np.int64))
    return totals


def _build_game_masses(
    config: PlannerConfig,
    games: np.ndarray,
    invfreq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    intermediate_dir: Path,
) -> np.memmap:
    n_games = len(games)
    total_plies = int(games["ply_count"].sum(dtype=np.uint64))
    active_bands = np.flatnonzero(band_totals > 0)
    b_count = len(active_bands)
    if b_count == 0 or total_plies == 0:
        raise PlanError("reservoir contains no valid plies")

    early_bands = _early_band_ids(config)
    early_col = {band: col for col, band in enumerate(early_bands)}
    masses = np.lib.format.open_memmap(
        intermediate_dir / "game_mass.npy", mode="w+", dtype=np.float64, shape=(n_games,)
    )
    lengths = games["ply_count"].astype(np.float64)
    masses[:] = (1.0 - config.lambda_balance) * lengths / float(total_plies)

    for band in active_bands:
        if band in early_col:
            u = int(unique_states[band])
            if u <= 0:
                raise PlanError(f"early band {band} has no distinct states")
            masses[:] += (
                config.lambda_balance
                / float(b_count)
                / float(u)
                * invfreq[:, early_col[band]]
            )
        else:
            start, stop = _band_bounds(config.bin_starts, int(band))
            counts = _count_interval_for_lengths(games["ply_count"], start, stop)
            masses[:] += (
                config.lambda_balance
                / float(b_count)
                * counts.astype(np.float64)
                / float(band_totals[band])
            )

    total_mass = float(masses.sum(dtype=np.float64))
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise PlanError("invalid game-mass normalization")
    if abs(total_mass - 1.0) > 1e-9:
        LOGGER.info("Renormalizing game masses (raw sum %.12f)", total_mass)
        masses[:] /= total_mass
    masses.flush()
    return masses


def _weighted_sample_games_without_replacement(
    game_mass: np.ndarray,
    target_games: int,
    seed: int,
) -> np.ndarray:
    n = len(game_mass)
    if target_games > n:
        raise PlanError(
            f"target_games={target_games:,} exceeds reservoir valid games={n:,}"
        )
    if target_games == n:
        return np.arange(n, dtype=np.uint32)
    ids = np.arange(n, dtype=np.uint64)
    u_bits = _splitmix64_np(ids ^ np.uint64(seed & UINT64_MASK) ^ np.uint64(0xC6BC279692B5CC83))
    u = ((u_bits >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))
    weights = np.asarray(game_mass, dtype=np.float64)
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise PlanError("all valid games must have finite positive target mass")
    keys = -np.log(u) / weights
    chosen = np.argpartition(keys, target_games - 1)[:target_games]
    # Sort by reservoir id, not by random priority.  DataLoader shuffling remains a
    # training concern and this makes the manifest stable/readable.
    return np.sort(chosen.astype(np.uint32, copy=False))


def _balanced_bin_mass_for_game(
    config: PlannerConfig,
    game_id: int,
    game_length: int,
    band: int,
    invfreq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
) -> float:
    active_count = int(np.count_nonzero(band_totals))
    if active_count == 0:
        return 0.0
    early_bands = _early_band_ids(config)
    if band in early_bands:
        col = early_bands.index(band)
        u = int(unique_states[band])
        if u == 0:
            return 0.0
        return (
            config.lambda_balance
            / active_count
            / u
            * float(invfreq[game_id, col])
        )
    start, stop = _band_bounds(config.bin_starts, band)
    if stop is None:
        count = max(game_length - start, 0)
    else:
        count = max(min(game_length, stop) - start, 0)
    if count == 0 or band_totals[band] == 0:
        return 0.0
    return config.lambda_balance / active_count * count / float(band_totals[band])


def _choose_anchor(
    *,
    config: PlannerConfig,
    game_id: int,
    selected_rank: int,
    game_length: int,
    game_mass: float,
    invfreq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    early_freq: np.ndarray,
    total_plies: int,
) -> int:
    u_component = _uniform_scalar(config.seed, game_id, 11) * game_mass
    natural_mass = (1.0 - config.lambda_balance) * game_length / float(total_plies)
    if u_component < natural_mass or config.lambda_balance == 0.0:
        u = _uniform_scalar(config.seed, game_id, 12)
        return min(int(u * game_length), game_length - 1)

    residual = u_component - natural_mass
    chosen_band = None
    cumulative = 0.0
    for band in range(len(config.bin_starts)):
        mass = _balanced_bin_mass_for_game(
            config,
            game_id,
            game_length,
            band,
            invfreq,
            unique_states,
            band_totals,
        )
        cumulative += mass
        if residual <= cumulative + 1e-18:
            chosen_band = band
            break
    if chosen_band is None:
        # Floating point tail: choose the latest supported band.
        supported = [
            b
            for b in range(len(config.bin_starts))
            if _balanced_bin_mass_for_game(
                config, game_id, game_length, b, invfreq, unique_states, band_totals
            )
            > 0
        ]
        chosen_band = supported[-1]

    start, stop = _band_bounds(config.bin_starts, chosen_band)
    end = game_length if stop is None else min(game_length, stop)
    if end <= start:
        raise PlanError("anchor sampler selected an empty game-band intersection")

    if start < config.state_balance_until:
        freqs = np.asarray(early_freq[selected_rank, start:end], dtype=np.float64)
        if np.any(freqs <= 0):
            raise PlanError(
                f"missing early frequency for selected game {game_id} in [{start}, {end})"
            )
        weights = 1.0 / freqs
        threshold = _uniform_scalar(config.seed, game_id, 13) * float(weights.sum())
        index = int(np.searchsorted(np.cumsum(weights), threshold, side="right"))
        return min(start + index, end - 1)

    u = _uniform_scalar(config.seed, game_id, 14)
    return start + min(int(u * (end - start)), end - start - 1)


def _window_for_anchor(length: int, anchor: int, width: int | None, policy: str) -> tuple[int, int]:
    if width is None or length <= width:
        return 0, length
    if policy == "right":
        start = max(0, anchor - width + 1)
    elif policy == "center":
        start = anchor - width // 2
    else:  # defensive; validated earlier
        raise ValueError(policy)
    start = min(max(start, 0), length - width)
    stop = start + width
    if not (start <= anchor < stop):
        raise AssertionError("constructed window does not contain anchor")
    return start, stop


def _build_selected_games(
    config: PlannerConfig,
    games: np.ndarray,
    game_mass: np.ndarray,
    selected_game_ids: np.ndarray,
    invfreq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    early_freq: np.ndarray,
) -> np.ndarray:
    total_plies = int(games["ply_count"].sum(dtype=np.uint64))
    selected = np.empty(len(selected_game_ids), dtype=SELECTED_GAME_DTYPE)
    meter = ProgressMeter("Choose q-weighted windows", len(selected_game_ids), config.progress_seconds)
    for rank, game_id_u32 in enumerate(selected_game_ids):
        game_id = int(game_id_u32)
        length = int(games["ply_count"][game_id])
        anchor = _choose_anchor(
            config=config,
            game_id=game_id,
            selected_rank=rank,
            game_length=length,
            game_mass=float(game_mass[game_id]),
            invfreq=invfreq,
            unique_states=unique_states,
            band_totals=band_totals,
            early_freq=early_freq,
            total_plies=total_plies,
        )
        start, stop = _window_for_anchor(length, anchor, config.max_window_plies, config.anchor_policy)
        selected[rank] = (
            game_id,
            int(games["pgn_start"][game_id]),
            int(games["pgn_end"][game_id]),
            length,
            start,
            stop,
            anchor,
            0,
            float(game_mass[game_id]),
        )
        meter.maybe_log(rank + 1)
    meter.maybe_log(len(selected_game_ids), force=True)
    return selected


# ---------------------------------------------------------------------------
# Per-ply loss selection
# ---------------------------------------------------------------------------


def _position_weight(
    *,
    config: PlannerConfig,
    ply: int,
    frequency: np.ndarray | float | int | None,
    total_plies: int,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
) -> np.ndarray | float:
    band = band_index(ply, config.bin_starts)
    active_bands = int(np.count_nonzero(band_totals))
    natural = (1.0 - config.lambda_balance) / float(total_plies)
    if ply < config.state_balance_until:
        if frequency is None:
            raise ValueError("early position weight requires state frequency")
        u = int(unique_states[band])
        return natural + config.lambda_balance / active_bands / u / np.asarray(frequency)
    return natural + config.lambda_balance / active_bands / float(band_totals[band])


def _candidate_class_histogram(
    config: PlannerConfig,
    selected: np.ndarray,
    early_freq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    total_plies: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], float]]:
    """Histogram candidate occurrences by (band, early_frequency).

    ``frequency=0`` identifies later bands, whose q weight is constant within band.
    The returned counts INCLUDE forced anchors; callers may subtract them before solving
    the free-target PPS probabilities.
    """

    counts: dict[tuple[int, int], int] = Counter()
    starts = selected["window_start"].astype(np.int64)
    stops = selected["window_stop"].astype(np.int64)

    # Early part: only at most state_balance_until vectorized sweeps.
    for ply in range(config.state_balance_until):
        active = (starts <= ply) & (stops > ply)
        if not np.any(active):
            continue
        freqs = np.asarray(early_freq[active, ply], dtype=np.uint32)
        if np.any(freqs == 0):
            raise PlanError(f"missing state frequency at early ply {ply}")
        unique_f, n = np.unique(freqs, return_counts=True)
        band = band_index(ply, config.bin_starts)
        for f, c in zip(unique_f, n):
            counts[(band, int(f))] += int(c)

    # Later bands: interval overlap is enough; no state lookup is needed.
    for band in range(len(config.bin_starts)):
        band_start, band_stop = _band_bounds(config.bin_starts, band)
        if band_start < config.state_balance_until:
            continue
        c = int(_interval_overlap_counts(starts, stops, band_start, band_stop).sum(dtype=np.int64))
        if c:
            counts[(band, 0)] += c

    weights: dict[tuple[int, int], float] = {}
    for (band, freq), count in counts.items():
        representative_ply = int(config.bin_starts[band])
        if representative_ply < config.state_balance_until:
            w = float(
                _position_weight(
                    config=config,
                    ply=representative_ply,
                    frequency=freq,
                    total_plies=total_plies,
                    unique_states=unique_states,
                    band_totals=band_totals,
                )
            )
        else:
            w = float(
                _position_weight(
                    config=config,
                    ply=representative_ply,
                    frequency=None,
                    total_plies=total_plies,
                    unique_states=unique_states,
                    band_totals=band_totals,
                )
            )
        if not (w > 0 and math.isfinite(w)):
            raise PlanError(f"invalid q weight {w} for class {(band, freq)}")
        weights[(band, freq)] = w
    return dict(counts), weights


def _anchor_class(
    config: PlannerConfig,
    selected: np.ndarray,
    early_freq: np.ndarray,
) -> Counter[tuple[int, int]]:
    result: Counter[tuple[int, int]] = Counter()
    for rank, row in enumerate(selected):
        ply = int(row["anchor_ply"])
        band = band_index(ply, config.bin_starts)
        if ply < config.state_balance_until:
            f = int(early_freq[rank, ply])
            if f <= 0:
                raise PlanError("forced anchor is missing its early-state frequency")
            result[(band, f)] += 1
        else:
            result[(band, 0)] += 1
    return result


def solve_pps_scale(
    counts: Mapping[tuple[int, int], int],
    weights: Mapping[tuple[int, int], float],
    target: int,
) -> tuple[float, dict[tuple[int, int], float]]:
    """Solve tau in sum_c n_c min(1, tau w_c) = target.

    This is the inclusion-probability calibration for fixed expected sample size.  It is
    exact at the class level; a later deterministic boundary correction makes the realized
    number of selected plies exact as well.
    """

    total = int(sum(counts.values()))
    if target < 0 or target > total:
        raise ValueError(f"target={target} must lie in [0, {total}]")
    if target == 0:
        return 0.0, {key: 0.0 for key in counts}
    if target == total:
        return math.inf, {key: 1.0 for key in counts}

    classes = [(key, int(counts[key]), float(weights[key])) for key in counts if counts[key] > 0]
    saturated: set[tuple[int, int]] = set()
    saturated_count = 0
    while True:
        denominator = sum(n * w for key, n, w in classes if key not in saturated)
        remaining = target - saturated_count
        if denominator <= 0 or remaining < 0:
            raise PlanError("could not calibrate PPS inclusion probabilities")
        tau = remaining / denominator
        newly = {
            key
            for key, _n, w in classes
            if key not in saturated and tau * w >= 1.0
        }
        if not newly:
            probs = {
                key: (1.0 if key in saturated else min(1.0, tau * w))
                for key, _n, w in classes
            }
            expected = sum(counts[key] * probs[key] for key in counts)
            if abs(expected - target) > max(1e-6, 1e-10 * max(1, target)):
                raise PlanError(
                    f"PPS calibration failed: expected {expected}, target {target}"
                )
            return tau, probs
        for key, n, _w in classes:
            if key in newly:
                saturated_count += n
        saturated |= newly


def _pack_mask(mask_u8: np.memmap, output_path: Path, chunk_bits: int = 8_000_000) -> None:
    # Keep chunks byte-aligned so concatenated np.packbits outputs are valid.
    chunk_bits = max(8, (chunk_bits // 8) * 8)
    with output_path.open("wb") as out:
        n = len(mask_u8)
        for start in range(0, n, chunk_bits):
            stop = min(start + chunk_bits, n)
            chunk = np.asarray(mask_u8[start:stop], dtype=np.uint8)
            out.write(np.packbits(chunk, bitorder=MASK_BITORDER).tobytes())
    expected = (len(mask_u8) + 7) // 8
    actual = output_path.stat().st_size
    if actual != expected:
        raise PlanError(f"packed mask size mismatch: expected {expected}, got {actual}")


class _SmallestMargins:
    """Keep the k smallest (margin, global_index) pairs with bounded memory."""

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.margins = np.empty(0, dtype=np.float64)
        self.indices = np.empty(0, dtype=np.uint64)

    def add(self, margins: np.ndarray, indices: np.ndarray) -> None:
        if self.k <= 0 or len(margins) == 0:
            return
        margins = np.asarray(margins, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.uint64)
        if len(margins) > self.k:
            loc = np.argpartition(margins, self.k - 1)[: self.k]
            margins = margins[loc]
            indices = indices[loc]
        if len(self.margins) == 0:
            self.margins = margins.copy()
            self.indices = indices.copy()
        else:
            all_m = np.concatenate([self.margins, margins])
            all_i = np.concatenate([self.indices, indices])
            if len(all_m) > self.k:
                loc = np.argpartition(all_m, self.k - 1)[: self.k]
                all_m = all_m[loc]
                all_i = all_i[loc]
            self.margins, self.indices = all_m, all_i

    def result(self) -> np.ndarray:
        if len(self.indices) != self.k:
            raise PlanError(
                f"boundary correction needed {self.k:,} candidates but found {len(self.indices):,}"
            )
        order = np.argsort(self.margins, kind="stable")
        return self.indices[order]


def _candidate_vectors_at_ply(
    *,
    config: PlannerConfig,
    selected: np.ndarray,
    early_freq: np.ndarray,
    ply: int,
    total_plies: int,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    tau: float,
    free_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = selected["window_start"]
    stops = selected["window_stop"]
    active = (starts <= ply) & (stops > ply)
    if free_only:
        active &= selected["anchor_ply"] != ply
    ranks = np.flatnonzero(active)
    if len(ranks) == 0:
        empty_u = np.empty(0, dtype=np.uint64)
        empty_f = np.empty(0, dtype=np.float64)
        return ranks, empty_u, empty_f, empty_f
    game_ids = selected["game_id"][ranks].astype(np.uint64, copy=False)
    if ply < config.state_balance_until:
        freqs = np.asarray(early_freq[ranks, ply], dtype=np.float64)
        weights = np.asarray(
            _position_weight(
                config=config,
                ply=ply,
                frequency=freqs,
                total_plies=total_plies,
                unique_states=unique_states,
                band_totals=band_totals,
            ),
            dtype=np.float64,
        )
    else:
        w = float(
            _position_weight(
                config=config,
                ply=ply,
                frequency=None,
                total_plies=total_plies,
                unique_states=unique_states,
                band_totals=band_totals,
            )
        )
        weights = np.full(len(ranks), w, dtype=np.float64)
    probabilities = np.ones(len(ranks), dtype=np.float64) if math.isinf(tau) else np.minimum(1.0, tau * weights)
    uniforms = _uniform_for_game_ply(game_ids, ply, config.seed, 31)
    return ranks, game_ids, probabilities, uniforms


def _select_loss_mask(
    config: PlannerConfig,
    selected: np.ndarray,
    game_ply_offsets: np.ndarray,
    early_freq: np.ndarray,
    unique_states: np.ndarray,
    band_totals: np.ndarray,
    target_plies: int,
    intermediate_dir: Path,
    output_dir: Path,
) -> tuple[np.memmap, float, int, int]:
    total_plies = int(game_ply_offsets[-1])
    candidate_count = int(
        (selected["window_stop"].astype(np.int64) - selected["window_start"].astype(np.int64)).sum()
    )
    n_items = len(selected)
    if target_plies > candidate_count:
        raise PlanError(
            f"target_plies={target_plies:,} exceeds the {candidate_count:,} plies available "
            "inside the selected context windows. Increase --max-window-plies, increase "
            "--target-games, or provide a smaller --target-plies budget."
        )
    if target_plies < n_items:
        raise PlanError(
            f"target_plies={target_plies:,} is smaller than target_games={n_items:,}; "
            "the plan guarantees at least one supervised anchor per training item"
        )

    class_counts, class_weights = _candidate_class_histogram(
        config, selected, early_freq, unique_states, band_totals, total_plies
    )
    forced = _anchor_class(config, selected, early_freq)
    free_counts = dict(class_counts)
    for key, n in forced.items():
        free_counts[key] = free_counts.get(key, 0) - n
        if free_counts[key] < 0:
            raise AssertionError("anchor count exceeds candidate class count")
    target_free = target_plies - n_items
    tau, _ = solve_pps_scale(free_counts, class_weights, target_free)

    mask_path = intermediate_dir / "loss_mask_u8.dat"
    mask = np.memmap(mask_path, mode="w+", dtype=np.uint8, shape=(total_plies,))
    mask[:] = 0

    # Forced q-distributed anchor: every selected item has at least one gradient-bearing target.
    anchor_flat = (
        game_ply_offsets[selected["game_id"].astype(np.int64)]
        + selected["anchor_ply"].astype(np.uint64)
    ).astype(np.uint64, copy=False)
    mask[anchor_flat] = 1
    realized_free = 0
    max_stop = int(selected["window_stop"].max(initial=0))
    meter = ProgressMeter("Sample supervised plies", max_stop, config.progress_seconds)

    for ply in range(max_stop):
        ranks, game_ids, probs, uniforms = _candidate_vectors_at_ply(
            config=config,
            selected=selected,
            early_freq=early_freq,
            ply=ply,
            total_plies=total_plies,
            unique_states=unique_states,
            band_totals=band_totals,
            tau=tau,
            free_only=True,
        )
        if len(ranks):
            choose = uniforms < probs
            if np.any(choose):
                chosen_ranks = ranks[choose]
                flat = (
                    game_ply_offsets[selected["game_id"][chosen_ranks].astype(np.int64)] + ply
                ).astype(np.uint64, copy=False)
                mask[flat] = 1
                realized_free += len(flat)
        meter.maybe_log(ply + 1, extra=f"selected={n_items + realized_free:,}/{target_plies:,}")
    meter.maybe_log(max_stop, extra=f"selected={n_items + realized_free:,}/{target_plies:,}", force=True)

    delta = target_free - realized_free
    if delta:
        LOGGER.info(
            "Deterministic boundary correction: %+d free targets to reach exact budget", delta
        )
        collector = _SmallestMargins(abs(delta))
        correction_meter = ProgressMeter(
            "Boundary correction", max_stop, config.progress_seconds
        )
        for ply in range(max_stop):
            ranks, _game_ids, probs, uniforms = _candidate_vectors_at_ply(
                config=config,
                selected=selected,
                early_freq=early_freq,
                ply=ply,
                total_plies=total_plies,
                unique_states=unique_states,
                band_totals=band_totals,
                tau=tau,
                free_only=True,
            )
            if len(ranks):
                selected_now = uniforms < probs
                if delta > 0:
                    eligible = ~selected_now
                    margins = uniforms[eligible] - probs[eligible]
                else:
                    eligible = selected_now
                    margins = probs[eligible] - uniforms[eligible]
                if np.any(eligible):
                    eranks = ranks[eligible]
                    flat = (
                        game_ply_offsets[selected["game_id"][eranks].astype(np.int64)] + ply
                    ).astype(np.uint64, copy=False)
                    collector.add(margins, flat)
            correction_meter.maybe_log(ply + 1)
        correction_meter.maybe_log(max_stop, force=True)
        flip = collector.result()
        mask[flip] = np.uint8(1 if delta > 0 else 0)

    realized = int(mask.sum(dtype=np.uint64))
    if realized != target_plies:
        raise PlanError(f"loss-mask budget mismatch: realized {realized:,}, target {target_plies:,}")
    mask.flush()
    _pack_mask(mask, output_dir / "loss_mask.bin")
    return mask, tau, candidate_count, delta


def _count_active_targets_per_game(
    selected: np.ndarray,
    game_ply_offsets: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(len(selected), dtype=np.uint32)
    max_stop = int(selected["window_stop"].max(initial=0))
    for ply in range(max_stop):
        active = (selected["window_start"] <= ply) & (selected["window_stop"] > ply)
        ranks = np.flatnonzero(active)
        if len(ranks):
            flat = game_ply_offsets[selected["game_id"][ranks].astype(np.int64)] + ply
            counts[ranks] += np.asarray(mask[flat], dtype=np.uint32)
    if np.any(counts == 0):
        raise PlanError("internal error: a selected training item has zero supervised targets")
    return counts


def _selected_counts_by_band(
    config: PlannerConfig,
    selected: np.ndarray,
    game_ply_offsets: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(len(config.bin_starts), dtype=np.int64)
    max_stop = int(selected["window_stop"].max(initial=0))
    for ply in range(max_stop):
        active = (selected["window_start"] <= ply) & (selected["window_stop"] > ply)
        ranks = np.flatnonzero(active)
        if not len(ranks):
            continue
        flat = game_ply_offsets[selected["game_id"][ranks].astype(np.int64)] + ply
        c = int(np.asarray(mask[flat], dtype=np.uint8).sum(dtype=np.int64))
        if c:
            counts[band_index(ply, config.bin_starts)] += c
    return counts


def _audit_selected_early_states(
    config: PlannerConfig,
    intermediate_dir: Path,
    game_ply_offsets: np.ndarray,
    mask: np.ndarray,
) -> StateAudit:
    selected_occ = [0 for _ in config.bin_starts]
    selected_unique = [0 for _ in config.bin_starts]
    early_bands = set(_early_band_ids(config))
    partitions_dir = intermediate_dir / "early_partitions"
    meter = ProgressMeter("Audit selected early states", config.partitions, config.progress_seconds)

    for partition in range(config.partitions):
        part = partitions_dir / f"part_{partition:04d}.bin"
        if not part.exists() or part.stat().st_size == 0:
            meter.maybe_log(partition + 1)
            continue
        records = np.fromfile(part, dtype=EARLY_RECORD_DTYPE)
        bands = records["band"]
        for band in early_bands:
            left = int(np.searchsorted(bands, band, side="left"))
            right = int(np.searchsorted(bands, band, side="right"))
            if left == right:
                continue
            sub = records[left:right]
            starts, _counts = _group_band_records(sub)
            flat = game_ply_offsets[sub["game"].astype(np.int64)] + sub["ply"].astype(np.uint64)
            chosen = np.asarray(mask[flat], dtype=np.int8)
            group_selected = np.add.reduceat(chosen, starts)
            selected_occ[band] += int(group_selected.sum(dtype=np.int64))
            selected_unique[band] += int(np.count_nonzero(group_selected))
        del records
        meter.maybe_log(partition + 1)
    meter.maybe_log(config.partitions, force=True)
    return StateAudit(selected_occ, selected_unique)


# ---------------------------------------------------------------------------
# Reports / contract
# ---------------------------------------------------------------------------


def _step_report(config: PlannerConfig, target_games: int) -> dict[str, Any] | None:
    if config.batch_size is None:
        return None
    batches = (
        target_games // config.batch_size
        if config.drop_last
        else math.ceil(target_games / config.batch_size)
    )
    optimizer_steps = math.ceil(batches / config.gradient_accumulation_steps)
    total = optimizer_steps * (config.epochs or 1)
    return {
        "batch_size": config.batch_size,
        "drop_last": config.drop_last,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
        "loader_batches_per_epoch": batches,
        "optimizer_steps_per_epoch": optimizer_steps,
        "optimizer_steps_total": total if config.epochs is not None else None,
    }


def _bin_rows(
    config: PlannerConfig,
    band_totals: np.ndarray,
    selected_counts: np.ndarray,
    unique_states: np.ndarray,
    state_audit: StateAudit,
    total_plies: int,
    target_plies: int,
) -> list[dict[str, Any]]:
    active_bands = int(np.count_nonzero(band_totals))
    rows: list[dict[str, Any]] = []
    for band in range(len(config.bin_starts)):
        start, stop = _band_bounds(config.bin_starts, band)
        n = int(band_totals[band])
        target_mass = (
            (1.0 - config.lambda_balance) * _safe_div(n, total_plies)
            + (config.lambda_balance / active_bands if n else 0.0)
        )
        row = {
            "band": band,
            "start_ply": start,
            "stop_ply_exclusive": stop,
            "reservoir_occurrences": n,
            "reservoir_fraction": _safe_div(n, total_plies),
            "ideal_q_mass": target_mass,
            "selected_occurrences": int(selected_counts[band]),
            "selected_fraction": _safe_div(int(selected_counts[band]), target_plies),
            "state_balanced": start < config.state_balance_until,
            "reservoir_unique_states": (
                int(unique_states[band]) if start < config.state_balance_until else None
            ),
            "selected_unique_states": (
                int(state_audit.selected_unique_states[band])
                if start < config.state_balance_until
                else None
            ),
        }
        if start < config.state_balance_until:
            so = int(state_audit.selected_occurrences[band])
            su = int(state_audit.selected_unique_states[band])
            row["selected_duplicate_fraction"] = _safe_div(max(so - su, 0), so)
        else:
            row["selected_duplicate_fraction"] = None
        rows.append(row)
    return rows


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Balanced chess training plan report",
        "",
        f"- Format: `{report['format']}`",
        f"- Profile: **{report['config']['profile']}**",
        f"- $\\lambda$: **{report['config']['lambda_balance']}**",
        f"- Selected training items: **{report['selection']['selected_games']:,}**",
        f"- Supervised plies: **{report['selection']['selected_plies']:,}**",
        f"- Candidate plies inside selected windows: **{report['selection']['candidate_plies']:,}**",
        f"- Reservoir valid plies: **{report['reservoir']['valid_plies']:,}**",
        f"- Reservoir valid games: **{report['reservoir']['valid_games']:,}**",
        "",
        "## Absolute-ply distribution",
        "",
        "| Band | Reservoir | Ideal mixture | Selected | State correction | Selected unique states | Duplicate fraction |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in report["bands"]:
        stop = "∞" if row["stop_ply_exclusive"] is None else str(row["stop_ply_exclusive"])
        label = f"[{row['start_ply']},{stop})"
        uniq = "—" if row["selected_unique_states"] is None else f"{row['selected_unique_states']:,}"
        dup = (
            "—"
            if row["selected_duplicate_fraction"] is None
            else f"{100 * row['selected_duplicate_fraction']:.3f}%"
        )
        lines.append(
            f"| {label} | {100*row['reservoir_fraction']:.3f}% | "
            f"{100*row['ideal_q_mass']:.3f}% | {100*row['selected_fraction']:.3f}% | "
            f"{'yes' if row['state_balanced'] else 'no'} | {uniq} | {dup} |"
        )

    lines.extend(["", "## Training contract", ""])
    contract = report["contract"]
    lines.extend(
        [
            f"- `selected_games.npy`: {contract['selected_games']}",
            f"- `game_ply_offsets.npy`: {contract['game_ply_offsets']}",
            f"- `loss_mask.bin`: {contract['loss_mask']}",
            f"- Decoder rule: **{contract['decoder_loss_rule']}**",
            "",
        ]
    )
    if report.get("steps"):
        steps = report["steps"]
        lines.extend(
            [
                "## Step budget",
                "",
                f"- Loader batches/epoch: **{steps['loader_batches_per_epoch']:,}**",
                f"- Optimizer steps/epoch: **{steps['optimizer_steps_per_epoch']:,}**",
                (
                    f"- Total optimizer steps: **{steps['optimizer_steps_total']:,}**"
                    if steps["optimizer_steps_total"] is not None
                    else "- Total optimizer steps: not computed (epochs not supplied)",
                ),
                "",
            ]
        )

    lines.extend(["## Warnings / interpretation", ""])
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _write_frequency_csv(
    path: Path,
    config: PlannerConfig,
    states_by_frequency: Sequence[Counter[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("band,start_ply,stop_ply_exclusive,state_frequency,num_distinct_states,num_occurrences\n")
        for band, counter in enumerate(states_by_frequency):
            if not counter:
                continue
            start, stop = _band_bounds(config.bin_starts, band)
            stop_text = "" if stop is None else str(stop)
            for freq in sorted(counter):
                states = int(counter[freq])
                fh.write(f"{band},{start},{stop_text},{freq},{states},{freq*states}\n")


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------


def build_plan(config: PlannerConfig) -> dict[str, Any]:
    """Build a plan and return the JSON-serializable audit report."""

    return _build_plan_impl(config)


def _build_plan_impl(config: PlannerConfig) -> dict[str, Any]:
    config.validate()
    reservoir = config.reservoir_pgn.resolve()
    baseline = config.baseline_pgn.resolve() if config.baseline_pgn else None
    excludes = tuple(path.resolve() for path in config.exclude_pgns)
    if not reservoir.exists():
        raise FileNotFoundError(reservoir)
    if baseline is not None and not baseline.exists():
        raise FileNotFoundError(baseline)
    for path in excludes:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        if not config.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory {output_dir} is not empty; pass --overwrite to replace it"
            )
        if config.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    LOGGER.info("Building %s", FORMAT_VERSION)
    LOGGER.info("Reservoir: %s", reservoir)
    LOGGER.info("Output: %s", output_dir)

    # Rebind normalized paths without mutating the frozen config.
    cfg = PlannerConfig(
        reservoir_pgn=reservoir,
        output_dir=output_dir,
        profile=config.profile,
        lambda_balance=config.lambda_balance,
        bin_starts=config.bin_starts,
        state_balance_until=config.state_balance_until,
        seed=config.seed,
        max_window_plies=config.max_window_plies,
        anchor_policy=config.anchor_policy,
        partitions=config.partitions,
        target_games=config.target_games,
        target_plies=config.target_plies,
        baseline_pgn=baseline,
        baseline_train_fraction=config.baseline_train_fraction,
        exclude_pgns=excludes,
        batch_size=config.batch_size,
        epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        drop_last=config.drop_last,
        overwrite=config.overwrite,
        keep_intermediate=config.keep_intermediate,
        progress_seconds=config.progress_seconds,
    )

    budget, baseline_stats = _resolve_budget(cfg)
    LOGGER.info(
        "Budget: %s training items and %s supervised plies",
        f"{budget.target_games:,}",
        f"{budget.target_plies:,}",
    )

    intermediate_dir = output_dir / "_intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    exclusion_set = _build_exclusion_set(excludes, cfg.progress_seconds)
    if excludes:
        LOGGER.info("Loaded %d unique exclusion-game fingerprints", len(exclusion_set))

    games_raw, reservoir_stats, early_records = _scan_reservoir(
        cfg, intermediate_dir, exclusion_set
    )
    if reservoir_stats.valid_games == 0 or reservoir_stats.raw_plies == 0:
        raise PlanError("reservoir contains no valid non-empty games")
    if budget.target_games > reservoir_stats.valid_games:
        raise PlanError(
            f"target game budget {budget.target_games:,} exceeds usable reservoir games "
            f"{reservoir_stats.valid_games:,}"
        )

    games = _load_game_table(games_raw, output_dir)
    game_ply_offsets = _write_game_ply_offsets(games, output_dir)
    band_totals = _band_counts_from_lengths(games["ply_count"], cfg.bin_starts)
    total_plies = int(game_ply_offsets[-1])
    if total_plies != reservoir_stats.raw_plies:
        raise AssertionError("ply prefix sum does not match scan total")

    invfreq, unique_states, states_by_frequency = _process_early_partitions_first_pass(
        cfg, intermediate_dir, len(games)
    )
    _write_frequency_csv(output_dir / "early_state_frequency_histogram.csv", cfg, states_by_frequency)

    game_mass = _build_game_masses(
        cfg, games, invfreq, unique_states, band_totals, intermediate_dir
    )
    selected_ids = _weighted_sample_games_without_replacement(
        game_mass, budget.target_games, cfg.seed
    )
    early_freq = _fill_selected_early_frequencies(
        cfg, intermediate_dir, selected_ids, len(games)
    )
    selected = _build_selected_games(
        cfg,
        games,
        game_mass,
        selected_ids,
        invfreq,
        unique_states,
        band_totals,
        early_freq,
    )

    mask, tau, candidate_count, correction_delta = _select_loss_mask(
        cfg,
        selected,
        game_ply_offsets,
        early_freq,
        unique_states,
        band_totals,
        budget.target_plies,
        intermediate_dir,
        output_dir,
    )
    selected["active_targets"] = _count_active_targets_per_game(
        selected, game_ply_offsets, mask
    )
    np.save(output_dir / "selected_games.npy", selected, allow_pickle=False)

    selected_counts = _selected_counts_by_band(
        cfg, selected, game_ply_offsets, mask
    )
    if int(selected_counts.sum()) != budget.target_plies:
        raise AssertionError("selected band counts do not sum to target plies")
    state_audit = _audit_selected_early_states(
        cfg, intermediate_dir, game_ply_offsets, mask
    )

    warnings: list[str] = []
    if not excludes:
        warnings.append(
            "No --exclude-pgn was supplied. The planner cannot prove that the expanded reservoir "
            "excludes the original validation/test games. Keep the TG1 evaluation split fixed and "
            "supply those PGNs (or otherwise filter the reservoir) before the final experiment."
        )
    if cfg.baseline_pgn is not None and cfg.baseline_train_fraction != 1.0 and (
        cfg.target_games is None or cfg.target_plies is None
    ):
        warnings.append(
            "At least one budget was obtained by multiplying aggregate baseline counts by "
            f"baseline_train_fraction={cfg.baseline_train_fraction}. This preserves the budget "
            "approximately but does not reproduce a historical random split exactly. Pass explicit "
            "--target-games/--target-plies from the TG1 run for exact parity."
        )
    if cfg.max_window_plies is None:
        warnings.append(
            "No --max-window-plies was supplied. Windows span full games; set the actual TG1 "
            "context limit when building the final encoder/decoder plan so the manifest matches "
            "what the model can consume."
        )
    if cfg.profile == "decoder":
        warnings.append(
            "For decoder integration, zero loss bits are label masks only. Never delete those move "
            "tokens from input_ids/attention; they remain causal context. window_start may be >0 "
            "only when the unchanged model context limit forces left truncation."
        )

    finished = datetime.now(timezone.utc)
    bins = _bin_rows(
        cfg,
        band_totals,
        selected_counts,
        unique_states,
        state_audit,
        total_plies,
        budget.target_plies,
    )
    report: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "config": {
            "profile": cfg.profile,
            "lambda_balance": cfg.lambda_balance,
            "bin_starts": list(cfg.bin_starts),
            "state_balance_until": cfg.state_balance_until,
            "seed": cfg.seed,
            "max_window_plies": cfg.max_window_plies,
            "anchor_policy": cfg.anchor_policy,
            "partitions": cfg.partitions,
        },
        "sources": {
            "reservoir": asdict(FileIdentity.from_path(reservoir)),
            "baseline": asdict(FileIdentity.from_path(baseline)) if baseline else None,
            "exclusions": [asdict(FileIdentity.from_path(path)) for path in excludes],
        },
        "budget": asdict(budget),
        "baseline_scan": asdict(baseline_stats) if baseline_stats else None,
        "reservoir": {
            "valid_games": reservoir_stats.valid_games,
            "valid_plies": reservoir_stats.raw_plies,
            "effective_plies_at_window_limit": reservoir_stats.effective_plies,
            "parse_errors_skipped": reservoir_stats.parse_errors,
            "zero_ply_games_skipped": reservoir_stats.zero_ply_games,
            "excluded_games": reservoir_stats.excluded_games,
            "early_state_records": early_records,
        },
        "selection": {
            "selected_games": len(selected),
            "selected_plies": budget.target_plies,
            "candidate_plies": candidate_count,
            "candidate_utilization": _safe_div(budget.target_plies, candidate_count),
            "pps_tau": None if math.isinf(tau) else tau,
            "boundary_correction_delta_before_fix": correction_delta,
            "min_active_targets_per_item": int(selected["active_targets"].min()),
            "max_active_targets_per_item": int(selected["active_targets"].max()),
            "mean_active_targets_per_item": float(selected["active_targets"].mean()),
        },
        "bands": bins,
        "steps": _step_report(cfg, len(selected)),
        "contract": {
            "selected_games": (
                "Structured NumPy array. Dataset item j must read reservoir bytes [pgn_start,pgn_end), "
                "return exactly [window_start, window_stop), and align local target k with absolute "
                "ply window_start+k. Dataset length is the number of rows."
            ),
            "game_ply_offsets": (
                "uint64 prefix-sum array for every valid reservoir game. The global bit index for "
                "(game_id, ply) is game_ply_offsets[game_id] + ply."
            ),
            "loss_mask": (
                f"One bit per valid reservoir ply, bitorder={MASK_BITORDER}. A 1 means that target "
                "contributes supervised loss; 0 means no supervised loss."
            ),
            "decoder_loss_rule": (
                "A zero loss bit sets the corresponding next-move/piece labels to ignore_index=-100; "
                "the move token itself remains visible in input_ids and attention as causal context."
            ),
            "encoder_loss_rule": (
                "AND the plan loss mask with the encoder's existing padding/valid-move/legal-target "
                "mask before cross entropy; do not overload the attention/padding mask."
            ),
        },
        "state_identity": {
            "fingerprint": "BLAKE2b-128",
            "fields": [
                "piece placement",
                "side to move",
                "clean castling rights",
                "legal en-passant square",
            ],
            "scope": f"exact frequency correction only for absolute ply < {cfg.state_balance_until}",
        },
        "warnings": warnings,
    }

    _atomic_write_text(output_dir / "report.json", json.dumps(report, indent=2) + "\n")
    _atomic_write_text(output_dir / "report.md", _report_markdown(report))

    plan = {
        "format": FORMAT_VERSION,
        "created_at_utc": finished.isoformat(),
        "reservoir": {
            "path": str(reservoir),
            "valid_games": reservoir_stats.valid_games,
            "valid_plies": reservoir_stats.raw_plies,
        },
        "profile": cfg.profile,
        "lambda_balance": cfg.lambda_balance,
        "bin_starts": list(cfg.bin_starts),
        "state_balance_until": cfg.state_balance_until,
        "max_window_plies": cfg.max_window_plies,
        "anchor_policy": cfg.anchor_policy,
        "seed": cfg.seed,
        "selected_games_file": "selected_games.npy",
        "reservoir_games_file": "reservoir_games.npy",
        "game_ply_offsets_file": "game_ply_offsets.npy",
        "loss_mask_file": "loss_mask.bin",
        "loss_mask_bitorder": MASK_BITORDER,
        "report_file": "report.json",
        "selected_game_dtype": SELECTED_GAME_DTYPE.descr,
        "reservoir_game_dtype": GAME_RECORD_DTYPE.descr,
        "budget": {
            "training_items": len(selected),
            "supervised_plies": budget.target_plies,
        },
    }
    _atomic_write_text(output_dir / "plan.json", json.dumps(plan, indent=2) + "\n")

    # Final self-check through the public reader contract.
    reader = BalancedPlanReader(output_dir)
    if len(reader) != budget.target_games:
        raise AssertionError("reader sees wrong number of training items")
    check_rows = min(64, len(reader))
    for i in range(check_rows):
        local = reader.local_loss_mask(i)
        expected = int(reader.selected_games[i]["active_targets"])
        if int(local.sum()) != expected:
            raise AssertionError("public reader contract does not reproduce active target count")

    if not cfg.keep_intermediate:
        del mask
        del early_freq
        del invfreq
        shutil.rmtree(intermediate_dir, ignore_errors=True)

    LOGGER.info("Plan complete: %s", output_dir)
    LOGGER.info("Selected items: %d; supervised plies: %d", len(selected), budget.target_plies)
    LOGGER.info("Report: %s", output_dir / "report.md")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_anchor_policy(profile: str) -> str:
    return "right" if profile == "decoder" else "center"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-compute, phase/state-balanced loss-mask plan from an expanded chess PGN."
        )
    )
    parser.add_argument("--reservoir-pgn", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["encoder", "decoder"], required=True)
    parser.add_argument(
        "--baseline-pgn",
        type=Path,
        default=None,
        help=(
            "Original unbalanced TG1 training corpus. Used only to derive the training-item and "
            "effective-ply budgets when explicit budgets are not supplied."
        ),
    )
    parser.add_argument(
        "--baseline-train-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of aggregate baseline counts to use when deriving budgets (default 1.0). "
            "For exact historical split parity, prefer explicit --target-games/--target-plies."
        ),
    )
    parser.add_argument("--target-games", type=int, default=None)
    parser.add_argument("--target-plies", type=int, default=None)
    parser.add_argument("--lambda-balance", type=float, default=0.5)
    parser.add_argument(
        "--bins",
        type=parse_bin_starts,
        default=DEFAULT_BIN_STARTS,
        help="Absolute-ply band starts, comma-separated (default: 0,20,40,60,80,100).",
    )
    parser.add_argument(
        "--state-balance-until",
        type=int,
        default=DEFAULT_STATE_BALANCE_UNTIL,
        help="Apply exact inverse state-frequency correction below this ply (default: 40).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-window-plies",
        type=int,
        default=None,
        help=(
            "Maximum target plies returned by one dataset item under the unchanged model context "
            "limit. Strongly recommended for final plans."
        ),
    )
    parser.add_argument(
        "--anchor-policy",
        choices=["auto", "center", "right"],
        default="auto",
        help="Window placement around the q-distributed anchor; auto=center encoder/right decoder.",
    )
    parser.add_argument(
        "--exclude-pgn",
        type=Path,
        action="append",
        default=[],
        help="PGN whose games must not enter the training reservoir (repeat for val/test PGNs).",
    )
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-seconds", type=float, default=DEFAULT_PROGRESS_SECONDS)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    anchor_policy = (
        _default_anchor_policy(args.profile) if args.anchor_policy == "auto" else args.anchor_policy
    )
    config = PlannerConfig(
        reservoir_pgn=args.reservoir_pgn,
        output_dir=args.output_dir,
        profile=args.profile,
        lambda_balance=args.lambda_balance,
        bin_starts=tuple(args.bins),
        state_balance_until=args.state_balance_until,
        seed=args.seed,
        max_window_plies=args.max_window_plies,
        anchor_policy=anchor_policy,
        partitions=args.partitions,
        target_games=args.target_games,
        target_plies=args.target_plies,
        baseline_pgn=args.baseline_pgn,
        baseline_train_fraction=args.baseline_train_fraction,
        exclude_pgns=tuple(args.exclude_pgn),
        batch_size=args.batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        drop_last=args.drop_last,
        overwrite=args.overwrite,
        keep_intermediate=args.keep_intermediate,
        progress_seconds=args.progress_seconds,
    )
    _build_plan_impl(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
