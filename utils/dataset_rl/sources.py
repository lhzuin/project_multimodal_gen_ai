"""Bounded two-pass Parquet sampling and streaming opening-suite exclusions."""

from collections import Counter
import heapq
import json
from pathlib import Path
from typing import Any, Iterator

import chess

from .config import BUCKETS, SHARES, BuildConfig
from .positions import digest, position_key


METADATA_COLUMNS = (
    "PuzzleId",
    "GameId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "WhiteElo",
    "BlackElo",
    "variant",
)
FULL_COLUMNS = METADATA_COLUMNS + ("moves", "analysis")


def eligible_buckets(row: dict[str, Any], config: BuildConfig) -> list[str]:
    if (
        row.get("variant") != "standard"
        or not row.get("GameId")
        or min(row.get("WhiteElo") or 0, row.get("BlackElo") or 0)
        < config.min_player_elo
    ):
        return []
    buckets = ["quiet"]
    themes = set(row.get("Themes") or [])
    line = str(row.get("Moves") or "").split()
    if (
        not config.min_puzzle_rating
        <= (row.get("Rating") or 0)
        <= config.max_puzzle_rating
        or (row.get("RatingDeviation") or 999) > config.max_rating_deviation
        or (row.get("Popularity") or 0) < config.min_popularity
        or (row.get("NbPlays") or 0) < config.min_plays
        or not 4 <= len(line) <= config.max_solution_plies
        or "mateIn1" in themes
    ):
        return buckets
    endgame = any("endgame" in theme.lower() for theme in themes)
    if endgame:
        buckets.append("endgame")
    elif "middlegame" in themes:
        buckets.append("tactics")
    return buckets


def candidate_rows(
    paths: list[Path], config: BuildConfig, counts: Counter
) -> dict[tuple[int, int], set[str]]:
    """Stable hash reservoirs avoid choosing just the start of a sorted shard."""
    import pyarrow.parquet as pq

    capacities = {
        bucket: int(sum(config.split_sizes.values()) * SHARES[bucket])
        * config.candidate_multiplier
        for bucket in BUCKETS
    }
    heaps: dict[str, list[tuple[int, int, int]]] = {bucket: [] for bucket in BUCKETS}
    for source_index, path in enumerate(paths):
        with pq.ParquetFile(path) as parquet:
            missing = set(FULL_COLUMNS) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Parquet source missing columns: {sorted(missing)}")
            row_index = 0
            for batch in parquet.iter_batches(
                batch_size=1024, columns=list(METADATA_COLUMNS), use_threads=False
            ):
                for row in batch.to_pylist():
                    counts["source_rows_scanned"] += 1
                    for bucket in eligible_buckets(row, config):
                        counts[f"metadata_eligible_{bucket}"] += 1
                        priority = int(
                            digest(f"{config.seed}|{bucket}|{row['PuzzleId']}"), 16
                        )
                        entry = (-priority, source_index, row_index)
                        heap = heaps[bucket]
                        if len(heap) < capacities[bucket]:
                            heapq.heappush(heap, entry)
                        elif entry > heap[0]:
                            heapq.heapreplace(heap, entry)
                    row_index += 1
    selected: dict[tuple[int, int], set[str]] = {}
    for bucket, heap in heaps.items():
        counts[f"reservoir_{bucket}"] = len(heap)
        for _, source, row in heap:
            selected.setdefault((source, row), set()).add(bucket)
    return selected


def iter_selected_rows(
    paths: list[Path],
    selected: dict[tuple[int, int], set[str]],
    columns: tuple[str, ...] = FULL_COLUMNS,
) -> Iterator[tuple[int, int, dict[str, Any], set[str]]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for source_index, path in enumerate(paths):
        wanted = {row for (source, row) in selected if source == source_index}
        with pq.ParquetFile(path) as parquet:
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=64, columns=list(columns), use_threads=False
            ):
                indices = [i for i in range(batch.num_rows) if offset + i in wanted]
                if indices:
                    rows = batch.take(pa.array(indices, type=pa.int32())).to_pylist()
                    for index, row in zip(indices, rows, strict=True):
                        yield source_index, offset + index, row, selected[
                            (source_index, offset + index)
                        ]
                offset += batch.num_rows


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Read a large opening array without loading the whole suite into memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError("exclusion file must contain a JSON array")
                    started = True
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError("opening array items must be objects")
                yield value
                buffer = buffer[end:]
            if not chunk:
                raise ValueError("truncated exclusion JSON array")


def exclusion_positions(paths: tuple[str, ...]) -> set[str]:
    output = set()
    for path in paths:
        for row in iter_json_array(Path(path)):
            fen = next(
                (
                    row[k]
                    for k in ("fen", "fen_after_prefix", "start_fen", "fen_after")
                    if row.get(k)
                ),
                None,
            )
            if fen is None:
                raise ValueError(f"opening without a FEN in {path}")
            output.add(position_key(chess.Board(fen)))
    return output
