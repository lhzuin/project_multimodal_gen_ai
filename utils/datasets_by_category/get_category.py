"""Split PGN games into Blitz, Rapid, and Classic event datasets.

The bulk helper deliberately reads one game at a time, making it suitable for
large merged PGN files as well as already-open streams.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import io
from pathlib import Path
from typing import BinaryIO, Literal, Mapping, TextIO, TypeAlias

import chess.pgn

from dataset.offsets import ensure_offsets


GameCategory: TypeAlias = Literal["Blitz", "Rapid", "Classic"]
PGNSource: TypeAlias = str | Path | TextIO | BinaryIO

_CATEGORY_FILENAMES: Mapping[GameCategory, str] = {
    "Blitz": "blitz",
    "Rapid": "rapid",
    "Classic": "classic",
}


@dataclass(frozen=True)
class SkippedGame:
    """A source game that could not safely be included in an output corpus."""

    game_number: int
    source_offset: int | None
    reason: str


@dataclass(frozen=True)
class SplitPGNResult:
    """Artifacts and counts produced by :func:`split_pgn_by_time_control`."""

    run_id: str
    event_catalog_path: Path
    pgn_paths: Mapping[GameCategory, Path]
    index_paths: Mapping[GameCategory, Path]
    category_counts: Mapping[GameCategory, int]
    parsed_games: int
    skipped_games: tuple[SkippedGame, ...]


class _TextLineAdapter:
    """Expose a binary or text line stream as the text interface python-chess needs."""

    def __init__(self, stream: TextIO | BinaryIO) -> None:
        self._stream = stream

    def readline(self) -> str:
        line = self._stream.readline()
        if isinstance(line, bytes):
            return line.decode("utf-8", errors="replace")
        return line

    def tell(self) -> int:
        return self._stream.tell()


def _normalise_event_name(event: str | None) -> str:
    """Trim and collapse whitespace without changing the Event's display spelling."""
    normalised = " ".join((event or "").split())
    # python-chess supplies "?" for absent mandatory roster tags. It was not an
    # Event value found in the source and should not pollute the event catalog.
    return "" if normalised == "?" else normalised


def classify_event(event: str | None) -> GameCategory:
    """Return the time-control label encoded in a PGN Event value.

    ``Blitz`` wins when both words appear. Any event that does not mention
    Blitz or Rapid, including an absent Event tag, is considered Classic.
    """
    normalised = _normalise_event_name(event).casefold()
    if "blitz" in normalised:
        return "Blitz"
    if "rapid" in normalised:
        return "Rapid"
    return "Classic"


def classify_game(game_or_pgn: chess.pgn.Game | str) -> GameCategory:
    """Classify a parsed game or a string containing one raw PGN game.

    Raises:
        TypeError: If the value is neither a ``chess.pgn.Game`` nor PGN text.
        ValueError: If raw PGN text does not contain a parseable game.
    """
    if isinstance(game_or_pgn, chess.pgn.Game):
        game = game_or_pgn
    elif isinstance(game_or_pgn, str):
        game = chess.pgn.read_game(io.StringIO(game_or_pgn))
        if game is None or game.errors:
            detail = "; ".join(str(error) for error in (game.errors if game else []))
            raise ValueError(f"Raw PGN text does not contain a valid game{': ' + detail if detail else ''}")
    else:
        raise TypeError("game_or_pgn must be a chess.pgn.Game or raw PGN text")

    return classify_event(game.headers.get("Event"))


def _stream_offset(stream: _TextLineAdapter) -> int | None:
    try:
        return stream.tell()
    except (AttributeError, OSError, ValueError):
        return None


def _open_source(source: PGNSource) -> tuple[_TextLineAdapter, TextIO | None]:
    """Return an adapter and, for path input, the owned stream to close."""
    if isinstance(source, (str, Path)):
        owned_stream = Path(source).open("r", encoding="utf-8", errors="replace")
        return _TextLineAdapter(owned_stream), owned_stream

    if not hasattr(source, "readline"):
        raise TypeError("source must be a PGN path or an open stream with readline()")
    return _TextLineAdapter(source), None


def _write_event_catalog(event_counts: Counter[str], catalog_path: Path) -> None:
    with catalog_path.open("w", encoding="utf-8", newline="\n") as catalog:
        for event_name in sorted(event_counts, key=lambda name: (name.casefold(), name)):
            catalog.write(f"{event_name}\t{event_counts[event_name]}\n")


def split_pgn_by_time_control(source: PGNSource, output_dir: str | Path) -> SplitPGNResult:
    """Create timestamped Blitz, Rapid, and Classic PGN datasets from *source*.

    ``source`` can be an uncompressed PGN path or an open text/binary stream.
    Caller-owned streams are read from their current position and are never
    closed. Invalid games are skipped and described in the returned result.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    pgn_paths = {
        category: destination / f"{stem}_games_{run_id}.pgn"
        for category, stem in _CATEGORY_FILENAMES.items()
    }
    index_paths = {
        category: destination / f"{stem}_games_{run_id}_index.json"
        for category, stem in _CATEGORY_FILENAMES.items()
    }
    catalog_path = destination / f"event_names_{run_id}.txt"
    counts: dict[GameCategory, int] = {category: 0 for category in _CATEGORY_FILENAMES}
    event_counts: Counter[str] = Counter()
    skipped: list[SkippedGame] = []
    parsed_games = 0

    reader, owned_source = _open_source(source)
    try:
        with (
            pgn_paths["Blitz"].open("w", encoding="utf-8", newline="\n") as blitz_out,
            pgn_paths["Rapid"].open("w", encoding="utf-8", newline="\n") as rapid_out,
            pgn_paths["Classic"].open("w", encoding="utf-8", newline="\n") as classic_out,
        ):
            outputs: Mapping[GameCategory, TextIO] = {
                "Blitz": blitz_out,
                "Rapid": rapid_out,
                "Classic": classic_out,
            }
            game_number = 0
            while True:
                offset = _stream_offset(reader)
                try:
                    game = chess.pgn.read_game(reader)
                except Exception as exc:
                    game_number += 1
                    skipped.append(SkippedGame(game_number, offset, f"PGN parse error: {exc}"))
                    # Continue only when the parser consumed input. Retrying an
                    # exception at the same position would otherwise loop forever.
                    next_offset = _stream_offset(reader)
                    if offset is None or next_offset is None or next_offset <= offset:
                        break
                    continue

                if game is None:
                    break

                game_number += 1
                if game.errors:
                    reason = "; ".join(str(error) for error in game.errors)
                    skipped.append(SkippedGame(game_number, offset, f"PGN parse error: {reason}"))
                    continue

                category = classify_event(game.headers.get("Event"))
                event_name = _normalise_event_name(game.headers.get("Event"))
                if event_name:
                    event_counts[event_name] += 1

                print(game, file=outputs[category], end="\n\n")
                counts[category] += 1
                parsed_games += 1
    finally:
        if owned_source is not None:
            owned_source.close()

    _write_event_catalog(event_counts, catalog_path)
    for category in _CATEGORY_FILENAMES:
        ensure_offsets(str(pgn_paths[category]), str(index_paths[category]))

    return SplitPGNResult(
        run_id=run_id,
        event_catalog_path=catalog_path,
        pgn_paths=pgn_paths,
        index_paths=index_paths,
        category_counts=counts,
        parsed_games=parsed_games,
        skipped_games=tuple(skipped),
    )