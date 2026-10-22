"""Download, filter, and preprocess TWIC games one weekly issue at a time.

Only one temporary ZIP is retained at any point. Games are parsed directly
from PGN members inside that ZIP and appended to one output corpus per selected
time-control category. The output PGNs and their byte-offset indexes are ready
for ``ChessGameSequenceDataset`` and ``PGNMoveDataset``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence, TextIO
import urllib.error
import urllib.request
import uuid
import zipfile

import chess.pgn

from dataset.offsets import ensure_offsets, load_pgn_offsets
from .categories import Category, VALID_CATEGORIES, get_event_category


DEFAULT_START_WEEK = 1661
DEFAULT_LOWER_WEEK = 920
DEFAULT_OUTPUT_DIR = Path("processed_games") / "by_category"
DEFAULT_PREFIX = "twic"
DEFAULT_URL_TEMPLATES = (
    # Kept in sync with utils/twic_download.py without importing that module:
    # its unconditional third-party ``requests`` import is not available in all
    # of the project's training environments.
    "https://theweekinchess.com/zips/twic{n}g.zip",
    "https://theweekinchess.com/assets/files/twic{n}g.zip",
)


class DownloadError(RuntimeError):
    """Raised after all URLs and retry attempts for one issue fail."""


@dataclass(frozen=True)
class DownloadReceipt:
    url: str
    attempts: int
    size_bytes: int


class IssueDownloader(Protocol):
    """Injectable downloader interface, allowing deterministic offline tests."""

    def download(self, issue: int, destination: Path) -> DownloadReceipt:
        ...


class HttpIssueDownloader:
    """Streaming TWIC downloader with browser-like headers and bounded retries."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        *,
        url_templates: Sequence[str] = DEFAULT_URL_TEMPLATES,
        retries: int = 4,
        timeout: float = 120.0,
        backoff_seconds: float = 1.0,
        chunk_size: int = 1 << 20,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if not url_templates:
            raise ValueError("at least one URL template is required")
        for template in url_templates:
            try:
                template.format(n=1, issue=1)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(f"invalid URL template {template!r}: {exc}") from exc
        self.url_templates = tuple(url_templates)
        self.retries = retries
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.timeout = timeout
        self.backoff_seconds = backoff_seconds
        self.chunk_size = chunk_size

    def close(self) -> None:
        """Retained for symmetry with injectable downloaders; no resources persist."""

        return None

    def download(self, issue: int, destination: Path) -> DownloadReceipt:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        errors: list[str] = []
        total_attempts = 0

        for template in self.url_templates:
            url = template.format(n=issue, issue=issue)
            for attempt in range(1, self.retries + 1):
                total_attempts += 1
                try:
                    request = urllib.request.Request(url, headers=self.HEADERS)
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        with partial.open("wb") as output:
                            while chunk := response.read(self.chunk_size):
                                output.write(chunk)

                    if partial.stat().st_size == 0:
                        raise DownloadError("server returned an empty file")
                    # Detect HTML error pages and corrupt/truncated archives before
                    # exposing the file to the processing stage.
                    with zipfile.ZipFile(partial) as archive:
                        if archive.testzip() is not None:
                            raise DownloadError("ZIP CRC validation failed")
                        if not any(
                            name.lower().endswith(".pgn") and not name.endswith("/")
                            for name in archive.namelist()
                        ):
                            raise DownloadError("archive contains no PGN member")

                    os.replace(partial, destination)
                    return DownloadReceipt(url, total_attempts, destination.stat().st_size)
                except (OSError, urllib.error.URLError, zipfile.BadZipFile, DownloadError) as exc:
                    partial.unlink(missing_ok=True)
                    errors.append(f"{url} attempt {attempt}/{self.retries}: {exc}")
                    # Retrying a permanently missing/forbidden URL only delays
                    # trying the alternate TWIC endpoint. Rate limits and server
                    # errors still receive exponential-backoff retries.
                    if isinstance(exc, urllib.error.HTTPError) and exc.code in {
                        400, 401, 403, 404, 410
                    }:
                        break
                    if attempt < self.retries:
                        time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise DownloadError("; ".join(errors))


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for one category-dataset build."""

    target_plies: int | None
    categories: tuple[Category, ...]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    prefix: str = DEFAULT_PREFIX
    min_elo: int | None = None
    include_variant: bool = False
    start_week: int = DEFAULT_START_WEEK
    lower_week: int = DEFAULT_LOWER_WEEK
    overwrite: bool = False
    all_games_in_range: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(category.casefold() for category in self.categories))
        invalid = sorted(set(normalized) - set(VALID_CATEGORIES))
        if invalid:
            raise ValueError(f"invalid categories: {', '.join(invalid)}")
        if not normalized:
            raise ValueError("at least one category is required")
        if self.all_games_in_range:
            if self.target_plies is not None:
                raise ValueError("omit target_plies with --all-games-in-range")
        elif self.target_plies is None or self.target_plies <= 0:
            raise ValueError("target_plies must be positive unless all_games_in_range is set")
        if self.min_elo is not None and self.min_elo < 0:
            raise ValueError("min_elo cannot be negative")
        if self.start_week < self.lower_week:
            raise ValueError("start_week must be greater than or equal to lower_week")
        if not self.prefix or Path(self.prefix).name != self.prefix:
            raise ValueError("prefix must be a non-empty filename prefix, not a path")
        object.__setattr__(self, "categories", normalized)
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass
class CategoryStats:
    games: int = 0
    plies: int = 0
    source_weeks: set[int] = field(default_factory=set)


@dataclass
class WeekStats:
    issue: int
    status: str = "pending"
    source_url: str | None = None
    download_attempts: int = 0
    download_size_bytes: int = 0
    download_sha256: str | None = None
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
    failed_weeks: tuple[int, ...]
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


class TWICCategoryDatasetBuilder:
    """Orchestrate incremental downloads and per-category PGN construction."""

    def __init__(self, config: BuildConfig, downloader: IssueDownloader | None = None) -> None:
        self.config = config
        self.downloader = downloader or HttpIssueDownloader()
        self._owns_downloader = downloader is None
        self.stats = {category: CategoryStats() for category in config.categories}
        self.week_stats: list[WeekStats] = []
        self.started_at = datetime.now(timezone.utc)
        self._seen_games: set[bytes] = set()

    def _paths(self) -> tuple[dict[Category, Path], dict[Category, Path], Path, Path]:
        output = self.config.output_dir.resolve()
        pgns = {category: output / f"{self.config.prefix}_{category}.pgn" for category in self.config.categories}
        indexes = {
            category: output / f"{self.config.prefix}_{category}_index.json"
            for category in self.config.categories
        }
        report_json = output / f"{self.config.prefix}_report.json"
        report_md = output / f"{self.config.prefix}_report.md"
        return pgns, indexes, report_json, report_md

    def _all_targets_reached(self) -> bool:
        if self.config.all_games_in_range:
            return False
        return all(stat.plies >= self.config.target_plies for stat in self.stats.values())

    def _range_complete(self) -> bool:
        return self.config.all_games_in_range and all(
            stat.status == "processed" for stat in self.week_stats
        ) and len(self.week_stats) == self.config.start_week - self.config.lower_week + 1

    @staticmethod
    def _game_digest(game: chess.pgn.Game) -> bytes:
        digest = hashlib.blake2b(digest_size=16, person=b"twic-eval-game")
        digest.update(game.board().fen().encode("ascii"))
        digest.update(b"\0")
        for move in game.mainline_moves():
            digest.update(move.uci().encode("ascii"))
            digest.update(b" ")
        return digest.digest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_destinations(self, destinations: Iterable[Path]) -> None:
        existing = [path for path in destinations if path.exists()]
        if existing and not self.config.overwrite:
            preview = "\n".join(f"- {path}" for path in existing)
            raise FileExistsError(f"output files already exist; pass --overwrite to replace them:\n{preview}")

    def _process_stream(self, source: TextIO, outputs: Mapping[Category, TextIO], week: WeekStats) -> bool:
        """Process one PGN member. Return false if parsing cannot safely continue."""

        while not self._all_targets_reached():
            try:
                before = source.tell()
            except (AttributeError, OSError):
                before = None
            try:
                game = chess.pgn.read_game(source)
            except Exception as exc:  # malformed input must not abort other weeks
                week.skipped["parser_exception"] += 1
                week.error = f"PGN parser exception: {exc}"
                try:
                    after = source.tell()
                except (AttributeError, OSError):
                    after = None
                if before is None or after is None or after <= before:
                    return False
                continue

            if game is None:
                return True
            week.games_seen += 1
            if game.errors:
                week.skipped["parse_error"] += 1
                continue
            if not self.config.include_variant and _has_variant_header(game):
                week.skipped["variant"] += 1
                continue
            if not _passes_min_elo(game, self.config.min_elo):
                week.skipped["elo"] += 1
                continue

            category = get_event_category(game.headers.get("Event"))
            if category not in self.stats:
                week.skipped["unselected_category"] += 1
                continue
            category_stats = self.stats[category]
            if not self.config.all_games_in_range and category_stats.plies >= self.config.target_plies:
                week.skipped["category_target_already_reached"] += 1
                continue

            plies = sum(1 for _ in game.mainline_moves())
            if plies == 0:
                week.skipped["zero_plies"] += 1
                continue
            if self.config.all_games_in_range:
                if any(move == chess.Move.null() for move in game.mainline_moves()):
                    week.skipped["null_move"] += 1
                    continue
                digest = self._game_digest(game)
                if digest in self._seen_games:
                    week.skipped["duplicate_game"] += 1
                    continue
                self._seen_games.add(digest)
                game.headers["TWICIssue"] = str(week.issue)

            # A python-chess exporter is stateful and must not be reused across
            # games; reuse would prepend earlier games to later outputs.
            exporter = chess.pgn.StringExporter(
                headers=True, variations=False, comments=False, columns=None
            )
            outputs[category].write(game.accept(exporter))
            outputs[category].write("\n\n")
            category_stats.games += 1
            category_stats.plies += plies
            category_stats.source_weeks.add(week.issue)
            week.games_written[category] = week.games_written.get(category, 0) + 1
            week.plies_written[category] = week.plies_written.get(category, 0) + plies
        return True

    def _process_archive(self, archive_path: Path, outputs: Mapping[Category, TextIO], week: WeekStats) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".pgn") and not name.endswith("/")
            )
            if not members:
                raise DownloadError("archive contains no PGN member")
            for member in members:
                if self._all_targets_reached():
                    break
                with archive.open(member, "r") as binary_source:
                    with io.TextIOWrapper(binary_source, encoding="utf-8", errors="replace", newline=None) as source:
                        if not self._process_stream(source, outputs, week):
                            week.skipped["abandoned_member_after_parser_exception"] += 1
                            if self.config.all_games_in_range:
                                raise DownloadError(
                                    f"cannot certify a complete range: abandoned PGN member {member}"
                                )

    def _make_report(
        self,
        pgn_paths: Mapping[Category, Path],
        index_paths: Mapping[Category, Path],
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        categories: dict[str, Any] = {}
        for category, stat in self.stats.items():
            index = load_pgn_offsets(str(index_paths[category]))
            categories[category] = {
                "target_plies": self.config.target_plies,
                "target_reached": self._range_complete() if self.config.all_games_in_range else stat.plies >= self.config.target_plies,
                "plies": stat.plies,
                "overshoot_plies": 0 if self.config.all_games_in_range else max(0, stat.plies - self.config.target_plies),
                "games": stat.games,
                "indexed_games": index["num_games"],
                "source_weeks": sorted(stat.source_weeks, reverse=True),
                "pgn_path": str(pgn_paths[category]),
                "pgn_size_bytes": pgn_paths[category].stat().st_size,
                "index_path": str(index_paths[category]),
            }

        config = asdict(self.config)
        config["output_dir"] = str(self.config.output_dir.resolve())
        weeks = []
        for stat in self.week_stats:
            row = asdict(stat)
            row["skipped"] = dict(stat.skipped)
            weeks.append(row)

        return {
            "format": "twic_category_dataset_report_v1",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - self.started_at).total_seconds(),
            "config": config,
            "all_targets_reached": self._range_complete() if self.config.all_games_in_range else self._all_targets_reached(),
            "range_complete": self._range_complete() if self.config.all_games_in_range else None,
            "categories": categories,
            "weeks": weeks,
            "failed_weeks": [stat.issue for stat in self.week_stats if stat.status == "download_failed"],
            "archive_failed_weeks": [
                stat.issue for stat in self.week_stats if stat.status == "archive_failed"
            ],
            "warnings": list(warnings),
        }

    @staticmethod
    def _report_markdown(report: Mapping[str, Any]) -> str:
        lines = [
            "# TWIC category dataset report",
            "",
            f"- All targets reached: **{report['all_targets_reached']}**",
            f"- Started (UTC): `{report['started_at_utc']}`",
            f"- Finished (UTC): `{report['finished_at_utc']}`",
            f"- Failed weeks: `{report['failed_weeks']}`",
            f"- Archive-processing failures: `{report['archive_failed_weeks']}`",
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
        lines.extend([
            "",
            "The JSON report beside this file contains per-week download, filtering, and write counts.",
            "",
        ])
        return "\n".join(lines)

    def build(self) -> BuildResult:
        pgn_paths, index_paths, report_json_path, report_markdown_path = self._paths()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_destinations([
            *pgn_paths.values(), *index_paths.values(), report_json_path, report_markdown_path
        ])

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
            archive_path = self.config.output_dir / f".twic_current_week_{token}.zip"
            try:
                for issue in range(self.config.start_week, self.config.lower_week - 1, -1):
                    if self._all_targets_reached():
                        break
                    week = WeekStats(issue=issue)
                    self.week_stats.append(week)
                    print(f"[TWIC {issue}] downloading...")
                    try:
                        receipt = self.downloader.download(issue, archive_path)
                        week.source_url = receipt.url
                        week.download_attempts = receipt.attempts
                        week.download_size_bytes = receipt.size_bytes
                        week.download_sha256 = self._file_sha256(archive_path)
                    except Exception as exc:
                        week.status = "download_failed"
                        week.error = str(exc)
                        print(f"[TWIC {issue}] download failed: {exc}")
                        continue

                    try:
                        self._process_archive(archive_path, outputs, week)
                        week.status = "processed"
                        progress = ", ".join(
                            f"{category}={stat.plies}" + (
                                "" if self.config.all_games_in_range else f"/{self.config.target_plies}"
                            ) for category, stat in self.stats.items()
                        )
                        print(f"[TWIC {issue}] processed ({progress})")
                    except (OSError, zipfile.BadZipFile, DownloadError) as exc:
                        week.status = "archive_failed"
                        week.error = str(exc)
                        print(f"[TWIC {issue}] archive processing failed: {exc}")
                    finally:
                        archive_path.unlink(missing_ok=True)
            finally:
                archive_path.unlink(missing_ok=True)

            for output in outputs.values():
                output.flush()
                output.close()
            outputs.clear()

            for category, temporary in temporary_pgns.items():
                os.replace(temporary, pgn_paths[category])
                ensure_offsets(str(pgn_paths[category]), str(index_paths[category]))

            warnings: list[str] = []
            for category, stat in self.stats.items():
                if not self.config.all_games_in_range and stat.plies < self.config.target_plies:
                    warnings.append(
                        f"{category} reached {stat.plies:,}/{self.config.target_plies:,} plies "
                        f"before the inclusive lower week {self.config.lower_week}."
                    )
            failed = [stat.issue for stat in self.week_stats if stat.status == "download_failed"]
            if failed:
                warnings.append(f"Failed to download {len(failed)} week(s): {failed}")
            archive_failed = [
                stat.issue for stat in self.week_stats if stat.status == "archive_failed"
            ]
            if archive_failed:
                warnings.append(
                    f"Failed to fully process {len(archive_failed)} archive(s): {archive_failed}"
                )

            report = self._make_report(pgn_paths, index_paths, warnings)
            _atomic_write_text(report_json_path, json.dumps(report, indent=2) + "\n")
            _atomic_write_text(report_markdown_path, self._report_markdown(report))

            return BuildResult(
                pgn_paths=pgn_paths,
                index_paths=index_paths,
                report_json_path=report_json_path,
                report_markdown_path=report_markdown_path,
                target_reached={
                    category: self._range_complete() if self.config.all_games_in_range else stat.plies >= self.config.target_plies
                    for category, stat in self.stats.items()
                },
                plies={category: stat.plies for category, stat in self.stats.items()},
                games={category: stat.games for category, stat in self.stats.items()},
                failed_weeks=tuple(failed),
                warnings=tuple(warnings),
            )
        finally:
            for output in outputs.values():
                output.close()
            for temporary in temporary_pgns.values():
                temporary.unlink(missing_ok=True)
            if self._owns_downloader and hasattr(self.downloader, "close"):
                self.downloader.close()  # type: ignore[attr-defined]


def _category(value: str) -> Category:
    normalized = value.casefold()
    if normalized not in VALID_CATEGORIES:
        raise argparse.ArgumentTypeError(f"choose from: {', '.join(VALID_CATEGORIES)}")
    return normalized  # type: ignore[return-value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-ready Blitz/Rapid/Classic corpora from TWIC issues."
    )
    parser.add_argument("target_plies", type=int, nargs="?", help="Minimum plies to collect per selected category.")
    parser.add_argument("--all-games-in-range", action="store_true", help="Include all qualifying games from every issue in the inclusive range.")
    parser.add_argument(
        "--categories", nargs="+", type=_category, required=True,
        help="One or more of: blitz rapid classic.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Shared output filename prefix.")
    parser.add_argument("--min-elo", type=int, default=None, help="Require both player Elos to meet this value.")
    parser.add_argument("--include-variant", action="store_true", help="Keep games carrying a Variant header.")
    parser.add_argument("--start-week", type=int, default=DEFAULT_START_WEEK)
    parser.add_argument("--lower-week", type=int, default=DEFAULT_LOWER_WEEK)
    parser.add_argument("--retries", type=int, default=4, help="Attempts per download URL (default: 4).")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--url-template", action="append", default=None,
        help="Override download URL template; use {n} for the week. Repeat for fallbacks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs with the same prefix.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BuildConfig(
        target_plies=args.target_plies,
        categories=tuple(args.categories),
        output_dir=args.output_dir,
        prefix=args.prefix,
        min_elo=args.min_elo,
        include_variant=args.include_variant,
        start_week=args.start_week,
        lower_week=args.lower_week,
        overwrite=args.overwrite,
        all_games_in_range=args.all_games_in_range,
    )
    downloader = HttpIssueDownloader(
        url_templates=tuple(args.url_template) if args.url_template else DEFAULT_URL_TEMPLATES,
        retries=args.retries,
        timeout=args.timeout,
        backoff_seconds=args.backoff_seconds,
    )
    try:
        result = TWICCategoryDatasetBuilder(config, downloader=downloader).build()
    finally:
        downloader.close()

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
