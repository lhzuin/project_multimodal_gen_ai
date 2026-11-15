from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile

import chess.pgn

from utils.dataset_by_category.categories import get_event_category
from utils.dataset_by_category.twic_category_builder import (
    BuildConfig,
    DownloadError,
    DownloadReceipt,
    HttpIssueDownloader,
    TWICCategoryDatasetBuilder,
)


@contextmanager
def workspace_temp_dir():
    root = Path(__file__).parent / f"_twic_test_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield str(root)
    finally:
        shutil.rmtree(root)


def _game(event: str, moves: str, white_elo: str = "2500", black_elo: str = "2500", variant: str | None = None) -> str:
    headers = [
        f'[Event "{event}"]',
        '[Site "Test"]',
        '[Date "2026.01.01"]',
        '[Round "1"]',
        '[White "White"]',
        '[Black "Black"]',
        '[Result "*"]',
        f'[WhiteElo "{white_elo}"]',
        f'[BlackElo "{black_elo}"]',
    ]
    if variant is not None:
        headers.append(f'[Variant "{variant}"]')
    return "\n".join(headers) + f"\n\n{moves} *\n\n"


def _make_zip(path: Path, pgn: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("games.pgn", pgn)


class FakeDownloader:
    def __init__(self, sources: dict[int, Path], failures: set[int] | None = None) -> None:
        self.sources = sources
        self.failures = failures or set()

    def download(self, issue: int, destination: Path) -> DownloadReceipt:
        if issue in self.failures:
            raise DownloadError("simulated network failure")
        shutil.copyfile(self.sources[issue], destination)
        return DownloadReceipt(f"fake://twic/{issue}", 1, destination.stat().st_size)


class CategoryTests(unittest.TestCase):
    def test_event_classification_is_case_insensitive(self) -> None:
        self.assertEqual(get_event_category(" Superbet POL BLITZ 2025 "), "blitz")
        self.assertEqual(get_event_category("World Rapid Championship"), "rapid")
        self.assertEqual(get_event_category("Candidates Tournament"), "classic")
        self.assertEqual(get_event_category(None), "classic")


class BuilderTests(unittest.TestCase):
    def test_full_range_keeps_all_games_and_deduplicates_across_issues(self) -> None:
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            newer = root / "newer.zip"
            older = root / "older.zip"
            duplicate = _game("Open Championship", "1. e4 e5 2. Nf3 Nc6")
            _make_zip(newer, duplicate + _game("Second Open", "1. d4 d5 2. c4"))
            _make_zip(older, duplicate + _game("Third Open", "1. c4 e5 2. Nc3"))
            result = TWICCategoryDatasetBuilder(
                BuildConfig(None, ("classic",), root / "out", prefix="test",
                            min_elo=1600, start_week=1667, lower_week=1666,
                            all_games_in_range=True),
                downloader=FakeDownloader({1667: newer, 1666: older}),
            ).build()
            self.assertEqual(result.games["classic"], 3)
            self.assertTrue(result.target_reached["classic"])
            report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
            self.assertTrue(report["range_complete"])
            self.assertEqual(report["weeks"][1]["skipped"]["duplicate_game"], 1)
            self.assertIsNotNone(report["weeks"][0]["download_sha256"])
            with result.pgn_paths["classic"].open(encoding="utf-8") as stream:
                issues = []
                while game := chess.pgn.read_game(stream):
                    issues.append(game.headers["TWICIssue"])
            self.assertEqual(issues, ["1667", "1667", "1666"])

    def test_builds_filtered_training_artifacts_and_reports_failure(self) -> None:
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            week3 = root / "week3.zip"
            week2 = root / "week2.zip"
            week1 = root / "week1.zip"
            _make_zip(
                week3,
                _game("City Blitz", "1. e4 e5 2. Nf3")
                + _game("City Rapid", "1. d4 d5")
                + _game("Variant Classic", "1. e4 e5 2. Nf3 Nc6", variant="Chess960")
                + _game("Low-rated Classic", "1. c4 e5 2. Nc3", black_elo="1999"),
            )
            # Week 2 deliberately fails, then week 1 completes Rapid and Classic.
            _make_zip(
                week1,
                _game("Open Rapid", "1. c4 e5")
                + _game("Open Championship", "1. e4 e5 2. Nf3 Nc6"),
            )
            _make_zip(week2, _game("Unused", "1. e4"))

            output = root / "out"
            config = BuildConfig(
                target_plies=3,
                categories=("blitz", "rapid", "classic"),
                output_dir=output,
                prefix="training",
                min_elo=2000,
                start_week=3,
                lower_week=1,
            )
            result = TWICCategoryDatasetBuilder(
                config,
                downloader=FakeDownloader({3: week3, 2: week2, 1: week1}, failures={2}),
            ).build()

            self.assertEqual(result.plies, {"blitz": 3, "rapid": 4, "classic": 4})
            self.assertEqual(result.failed_weeks, (2,))
            self.assertTrue(all(result.target_reached.values()))
            self.assertTrue(all(path.exists() for path in result.index_paths.values()))

            report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
            self.assertTrue(report["all_targets_reached"])
            self.assertEqual(report["failed_weeks"], [2])
            self.assertEqual(report["weeks"][0]["skipped"]["variant"], 1)
            self.assertEqual(report["weeks"][0]["skipped"]["elo"], 1)
            self.assertEqual(report["categories"]["classic"]["indexed_games"], 1)
            for category in config.categories:
                self.assertEqual(
                    report["categories"][category]["indexed_games"],
                    result.games[category],
                )

            with result.pgn_paths["classic"].open(encoding="utf-8") as stream:
                game = chess.pgn.read_game(stream)
            self.assertEqual(game.headers["Event"], "Open Championship")

    def test_warns_when_lower_bound_is_exhausted(self) -> None:
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            week = root / "week.zip"
            _make_zip(week, _game("Only Blitz", "1. e4"))
            result = TWICCategoryDatasetBuilder(
                BuildConfig(5, ("blitz", "rapid"), root / "out", start_week=1, lower_week=1),
                downloader=FakeDownloader({1: week}),
            ).build()
            self.assertFalse(result.target_reached["blitz"])
            self.assertFalse(result.target_reached["rapid"])
            self.assertEqual(len(result.warnings), 2)

    def test_refuses_to_replace_existing_outputs_by_default(self) -> None:
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            output = root / "out"
            output.mkdir()
            (output / "twic_blitz.pgn").write_text("existing", encoding="utf-8")
            builder = TWICCategoryDatasetBuilder(
                BuildConfig(1, ("blitz",), output, start_week=1, lower_week=1),
                downloader=FakeDownloader({}),
            )
            with self.assertRaises(FileExistsError):
                builder.build()


class DownloaderTests(unittest.TestCase):
    def test_streaming_downloader_uses_fallback_and_validates_zip(self) -> None:
        with workspace_temp_dir() as raw_tmp:
            root = Path(raw_tmp)
            fallback = root / "fallback"
            fallback.mkdir()
            source = fallback / "twic7g.zip"
            _make_zip(source, _game("Test Blitz", "1. e4 e5"))
            destination = root / "download" / "week.zip"

            downloader = HttpIssueDownloader(
                url_templates=(
                    f"{(root / 'missing').as_uri()}/twic{{n}}g.zip",
                    f"{fallback.as_uri()}/twic{{n}}g.zip",
                ),
                retries=1,
            )
            receipt = downloader.download(7, destination)

            self.assertEqual(receipt.attempts, 2)
            self.assertEqual(receipt.url, source.as_uri())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(destination.with_name("week.zip.part").exists())


if __name__ == "__main__":
    unittest.main()
