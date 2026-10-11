"""Filter local PGNs into training-ready time-control category datasets.

The input may be one PGN file or a directory. Directory inputs are searched
recursively for ``*.pgn`` files and processed in deterministic path order as a
single corpus. Complete games are appended to one output PGN per selected
category until every category reaches its requested ply target or the input is
exhausted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO
import uuid

import chess.pgn

from dataset.offsets import ensure_offsets, load_pgn_offsets
from .categories import Category, VALID_CATEGORIES, get_event_category


DEFAULT_OUTPUT_DIR = Path("processed_games") / "by_category"
DEFAULT_PREFIX = "pgn"


def _parse_date(value: date | str | None, *, field_name: str = "date") -> date | None:
    """Parse a complete PGN/ISO date, returning ``None`` only for no value."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        normalized = value.strip().replace("-", ".")
        return datetime.strptime(normalized, "%Y.%m.%d").date()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a complete date in YYYY.MM.DD or YYYY-MM-DD format"
        ) from exc


def _parse_game_date(value: str | None) -> date | None:
    """Return a complete, valid PGN Date header or ``None`` for unknown dates."""

    if value is None or "?" in value:
        return None
    try:
        return _parse_date(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for one local-PGN category-dataset build.

    ``date_after`` and ``date_before`` are strict bounds. A game on either
    boundary is excluded. Strings may use PGN (``YYYY.MM.DD``) or ISO
    (``YYYY-MM-DD``) notation.
    """

    source_path: Path
    target_plies: int
    categories: tuple[Category, ...]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prefix: str = DEFAULT_PREFIX
    min_elo: int | None = None
    include_variant: bool = False
    date_after: date | str | None = None
    date_before: date | str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(category.casefold() for category in self.categories))
        invalid = sorted(set(normalized) - set(VALID_CATEGORIES))
        if invalid:
            raise ValueError(f"invalid categories: {', '.join(invalid)}")
        if not normalized:
            raise ValueError("at least one category is required")
        if self.target_plies <= 0:
            raise ValueError("target_plies must be positive")
        if self.min_elo is not None and self.min_elo < 0:
            raise ValueError("min_elo cannot be negative")
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise ValueError("prefix must be a non-empty filename prefix, not a path")

        after = _parse_date(self.date_after, field_name="date_after")
        before = _parse_date(self.date_before, field_name="date_before")
        if after is not None and before is not None and after >= before:
            raise ValueError("date_after must be earlier than date_before")

        object.__setattr__(self, "source_path", Path(self.source_path).expanduser())
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "categories", normalized)
        object.__setattr__(self, "date_after", after)
        object.__setattr__(self, "date_before", before)


@dataclass
class CategoryStats:
    games: int = 0
    plies: int = 0
    source_files: set[str] = field(default_factory=set)


@dataclass
class SourceStats:
    path: str
    status: str = "pending"
    games_seen: int = 0
    games_written: dict[str, int] = field(default_factory=dict)
    plies_written: dict[str, int] = field(default_factory=dict)
    skipped: Counter[str] = field(default_factory=Counter)
    error: str | None = None


@dataclass(frozen=True)
class BuildResult:
    pgn_paths: Mapping[Category, Path]
    index_paths: Mapping[Category, Path]
    report_json_path: Path
    report_markdown_path: Path
    target_reached: Mapping[Category, bool]
    plies: Mapping[Category, int]
    games: Mapping[Category, int]
    source_files: tuple[Path, ...]
    failed_sources: tuple[Path, ...]
    warnings: tuple[str, ...]


def _parse_elo(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def _passes_min_elo(game: chess.pgn.Game, minimum: int | None) -> bool:
    """Require both players to have a numeric rating at least ``minimum``."""

    if minimum is None:
        return True
    white = _parse_elo(game.headers.get("WhiteElo"))
    black = _parse_elo(game.headers.get("BlackElo"))
    return white is not None and black is not None and white >= minimum and black >= minimum


def _has_variant_header(game: chess.pgn.Game) -> bool:
    value = game.headers.get("Variant")
    return value is not None and bool(value.strip())


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PGNCategoryDatasetBuilder:
    """Build category datasets from one local PGN file or directory of PGNs."""

    def __init__(self, config: BuildConfig) -> None:
        self.config = config
        self.stats = {category: CategoryStats() for category in config.categories}
        self.source_stats: list[SourceStats] = []
        self.started_at = datetime.now(timezone.utc)

    def _paths(self) -> tuple[dict[Category, Path], dict[Category, Path], Path, Path]:
        output = self.config.output_dir.resolve()
        pgns = {
            category: output / f"{self.config.prefix}_{category}.pgn"
            for category in self.config.categories
        }
        indexes = {
            category: output / f"{self.config.prefix}_{category}_index.json"
            for category in self.config.categories
        }
        report_json = output / f"{self.config.prefix}_report.json"
        report_md = output / f"{self.config.prefix}_report.md"
        return pgns, indexes, report_json, report_md

    def _source_files(self, excluded: Iterable[Path] = ()) -> tuple[Path, ...]:
        source = self.config.source_path.expanduser().resolve()
        excluded_paths = {path.resolve() for path in excluded}
        if not source.exists():
            raise FileNotFoundError(f"PGN source does not exist: {source}")
        if source.is_file():
            if source.suffix.casefold() != ".pgn":
                raise ValueError(f"source file must have a .pgn extension: {source}")
            candidates = (source,)
        elif source.is_dir():
            candidates = tuple(
                sorted(
                    (
                        path.resolve()
                        for path in source.rglob("*")
                        if path.is_file() and path.suffix.casefold() == ".pgn"
                    ),
                    key=lambda path: path.as_posix().casefold(),
                )
            )
        else:
            raise ValueError(f"PGN source is neither a regular file nor a directory: {source}")

        candidates = tuple(path for path in candidates if path not in excluded_paths)
        if not candidates:
            raise ValueError(f"no PGN files found in source: {source}")
        return candidates

    def _all_targets_reached(self) -> bool:
        return all(stat.plies >= self.config.target_plies for stat in self.stats.values())

    def _validate_destinations(self, destinations: Iterable[Path]) -> None:
        existing = [path for path in destinations if path.exists()]
        if existing and not self.config.overwrite:
            preview = "\n".join(f"- {path}" for path in existing)
            raise FileExistsError(
                f"output files already exist; pass --overwrite to replace them:\n{preview}"
            )

    def _passes_date_filter(self, game: chess.pgn.Game, source: SourceStats) -> bool:
        after = self.config.date_after
        before = self.config.date_before
        if after is None and before is None:
            return True
        game_date = _parse_game_date(game.headers.get("Date"))
        if game_date is None:
            source.skipped["date_missing_or_invalid"] += 1
            return False
        if after is not None and game_date <= after:
            source.skipped["date_not_after"] += 1
            return False
        if before is not None and game_date >= before:
            source.skipped["date_not_before"] += 1
            return False
        return True

    def _process_stream(
        self,
        stream: TextIO,
        outputs: Mapping[Category, TextIO],
        source: SourceStats,
    ) -> bool:
        """Process a PGN stream, returning false if parsing cannot safely continue."""

        while not self._all_targets_reached():
            try:
                before = stream.tell()
            except (AttributeError, OSError):
                before = None
            try:
                game = chess.pgn.read_game(stream)
            except Exception as exc:
                source.skipped["parser_exception"] += 1
                source.error = f"PGN parser exception: {exc}"
                try:
                    after = stream.tell()
                except (AttributeError, OSError):
                    after = None
                if before is None or after is None or after <= before:
                    return False
                continue

            if game is None:
                return True
            source.games_seen += 1
            if game.errors:
                source.skipped["parse_error"] += 1
                continue
            if not self.config.include_variant and _has_variant_header(game):
                source.skipped["variant"] += 1
                continue
            if not _passes_min_elo(game, self.config.min_elo):
                source.skipped["elo"] += 1
                continue
            if not self._passes_date_filter(game, source):
                continue

            category = get_event_category(game.headers.get("Event"))
            if category not in self.stats:
                source.skipped["unselected_category"] += 1
                continue
            category_stats = self.stats[category]
            if category_stats.plies >= self.config.target_plies:
                source.skipped["category_target_already_reached"] += 1
                continue

            plies = sum(1 for _ in game.mainline_moves())
            if plies == 0:
                source.skipped["zero_plies"] += 1
                continue

            exporter = chess.pgn.StringExporter(
                headers=True, variations=False, comments=False, columns=None
            )
            outputs[category].write(game.accept(exporter))
            outputs[category].write("\n\n")
            category_stats.games += 1
            category_stats.plies += plies
            category_stats.source_files.add(source.path)
            source.games_written[category] = source.games_written.get(category, 0) + 1
            source.plies_written[category] = source.plies_written.get(category, 0) + plies
        return True

    def _make_report(
        self,
        pgn_paths: Mapping[Category, Path],
        index_paths: Mapping[Category, Path],
        source_files: Sequence[Path],
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        categories: dict[str, Any] = {}
        for category, stat in self.stats.items():
            index = load_pgn_offsets(str(index_paths[category]))
            categories[category] = {
                "target_plies": self.config.target_plies,
                "target_reached": stat.plies >= self.config.target_plies,
                "plies": stat.plies,
                "overshoot_plies": max(0, stat.plies - self.config.target_plies),
                "games": stat.games,
                "indexed_games": index["num_games"],
                "source_files": sorted(stat.source_files),
                "pgn_path": str(pgn_paths[category]),
                "pgn_size_bytes": pgn_paths[category].stat().st_size,
                "index_path": str(index_paths[category]),
            }

        config = asdict(self.config)
        config["source_path"] = str(self.config.source_path.resolve())
        config["output_dir"] = str(self.config.output_dir.resolve())
        config["date_after"] = (
            self.config.date_after.isoformat() if self.config.date_after is not None else None
        )
        config["date_before"] = (
            self.config.date_before.isoformat() if self.config.date_before is not None else None
        )
        sources = []
        for stat in self.source_stats:
            row = asdict(stat)
            row["skipped"] = dict(stat.skipped)
            sources.append(row)

        return {
            "format": "pgn_category_dataset_report_v1",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - self.started_at).total_seconds(),
            "config": config,
            "all_targets_reached": self._all_targets_reached(),
            "categories": categories,
            "source_files": [str(path) for path in source_files],
            "sources": sources,
            "failed_sources": [
                stat.path for stat in self.source_stats if stat.status == "source_failed"
            ],
            "warnings": list(warnings),
        }

    @staticmethod
    def _report_markdown(report: Mapping[str, Any]) -> str:
        lines = [
            "# Local PGN category dataset report",
            "",
            f"- All targets reached: **{report['all_targets_reached']}**",
            f"- Started (UTC): `{report['started_at_utc']}`",
            f"- Finished (UTC): `{report['finished_at_utc']}`",
            f"- Input PGN files: **{len(report['source_files'])}**",
            f"- Failed sources: `{report['failed_sources']}`",
            "",
            "## Datasets",
            "",
            "| Category | Games | Plies | Target | Reached | PGN | Index |",
            "|---|---:|---:|---:|:---:|---|---|",
        ]
        for category, stat in report["categories"].items():
            lines.append(
                f"| {category} | {stat['games']} | {stat['plies']} | "
                f"{stat['target_plies']} | {stat['target_reached']} | "
                f"`{stat['pgn_path']}` | `{stat['index_path']}` |"
            )
        lines.extend(["", "## Warnings", ""])
        warnings = report["warnings"]
        lines.extend(f"- {warning}" for warning in warnings)
        if not warnings:
            lines.append("None.")
        lines.extend(
            [
                "",
                "The JSON report beside this file contains per-source filtering and write counts.",
                "",
            ]
        )
        return "\n".join(lines)

    def build(self) -> BuildResult:
        pgn_paths, index_paths, report_json_path, report_markdown_path = self._paths()
        destinations = [
            *pgn_paths.values(),
            *index_paths.values(),
            report_json_path,
            report_markdown_path,
        ]
        source_files = self._source_files(excluded=pgn_paths.values())
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_destinations(destinations)

        token = uuid.uuid4().hex
        temporary_pgns = {
            category: path.with_name(f".{path.name}.{token}.tmp")
            for category, path in pgn_paths.items()
        }
        outputs: dict[Category, TextIO] = {}
        try:
            outputs = {
                category: path.open("w", encoding="utf-8", newline="\n")
                for category, path in temporary_pgns.items()
            }
            for source_path in source_files:
                if self._all_targets_reached():
                    break
                source = SourceStats(path=str(source_path))
                self.source_stats.append(source)
                print(f"[PGN] processing {source_path}...")
                try:
                    with source_path.open(
                        "r", encoding="utf-8", errors="replace", newline=None
                    ) as stream:
                        completed = self._process_stream(stream, outputs, source)
                    source.status = "processed" if completed else "parser_abandoned"
                    if not completed:
                        source.skipped["abandoned_after_parser_exception"] += 1
                    progress = ", ".join(
                        f"{category}={stat.plies}/{self.config.target_plies}"
                        for category, stat in self.stats.items()
                    )
                    print(f"[PGN] processed {source_path} ({progress})")
                except OSError as exc:
                    source.status = "source_failed"
                    source.error = str(exc)
                    print(f"[PGN] source failed {source_path}: {exc}")

            for output in outputs.values():
                output.flush()
                output.close()
            outputs.clear()

            for category, temporary in temporary_pgns.items():
                os.replace(temporary, pgn_paths[category])
                # Always regenerate an existing index when --overwrite is used,
                # even in the unlikely case of identical size and timestamp.
                index_paths[category].unlink(missing_ok=True)
                ensure_offsets(str(pgn_paths[category]), str(index_paths[category]))

            warnings: list[str] = []
            for category, stat in self.stats.items():
                if stat.plies < self.config.target_plies:
                    warnings.append(
                        f"{category} reached {stat.plies:,}/{self.config.target_plies:,} plies "
                        "before the local PGN input was exhausted."
                    )
            failed = [
                Path(stat.path) for stat in self.source_stats if stat.status == "source_failed"
            ]
            if failed:
                warnings.append(f"Failed to read {len(failed)} source file(s): {failed}")

            report = self._make_report(pgn_paths, index_paths, source_files, warnings)
            _atomic_write_text(report_json_path, json.dumps(report, indent=2) + "\n")
            _atomic_write_text(report_markdown_path, self._report_markdown(report))

            return BuildResult(
                pgn_paths=pgn_paths,
                index_paths=index_paths,
                report_json_path=report_json_path,
                report_markdown_path=report_markdown_path,
                target_reached={
                    category: stat.plies >= self.config.target_plies
                    for category, stat in self.stats.items()
                },
                plies={category: stat.plies for category, stat in self.stats.items()},
                games={category: stat.games for category, stat in self.stats.items()},
                source_files=source_files,
                failed_sources=tuple(failed),
                warnings=tuple(warnings),
            )
        finally:
            for output in outputs.values():
                output.close()
            for temporary in temporary_pgns.values():
                temporary.unlink(missing_ok=True)


def _category(value: str) -> Category:
    normalized = value.casefold()
    if normalized not in VALID_CATEGORIES:
        raise argparse.ArgumentTypeError(f"choose from: {', '.join(VALID_CATEGORIES)}")
    return normalized  # type: ignore[return-value]


def _date_bound(value: str) -> date:
    try:
        parsed = _parse_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert parsed is not None
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-ready Blitz/Rapid/Classic corpora from local PGNs."
    )
    parser.add_argument("source_path", type=Path, help="A .pgn file or directory of PGN files.")
    parser.add_argument(
        "target_plies", type=int, help="Minimum plies to collect per selected category."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        type=_category,
        required=True,
        help="One or more of: blitz rapid classic.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Shared output filename prefix.")
    parser.add_argument(
        "--min-elo",
        type=int,
        default=None,
        help="Require both player Elos to meet this value.",
    )
    parser.add_argument(
        "--include-variant", action="store_true", help="Keep games carrying a Variant header."
    )
    parser.add_argument(
        "--date-after",
        "--date-greater-than",
        dest="date_after",
        type=_date_bound,
        default=None,
        metavar="DATE",
        help="Keep only games strictly after DATE (YYYY.MM.DD or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-before",
        "--date-less-than",
        dest="date_before",
        type=_date_bound,
        default=None,
        metavar="DATE",
        help="Keep only games strictly before DATE (YYYY.MM.DD or YYYY-MM-DD).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs with the same prefix.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BuildConfig(
        source_path=args.source_path,
        target_plies=args.target_plies,
        categories=tuple(args.categories),
        output_dir=args.output_dir,
        prefix=args.prefix,
        min_elo=args.min_elo,
        include_variant=args.include_variant,
        date_after=args.date_after,
        date_before=args.date_before,
        overwrite=args.overwrite,
    )
    result = PGNCategoryDatasetBuilder(config).build()

    print("Build complete.")
    for category in config.categories:
        print(
            f"- {category}: {result.games[category]} games, {result.plies[category]} plies, "
            f"target_reached={result.target_reached[category]}"
        )
        print(f"  PGN: {result.pgn_paths[category]}")
        print(f"  index: {result.index_paths[category]}")
    print(f"Report: {result.report_markdown_path}")
    return 0 if all(result.target_reached.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
