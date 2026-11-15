from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import chess.pgn

from utils.dataset_by_category.pgn_category_builder import (
    BuildConfig,
    PGNCategoryDatasetBuilder,
    parse_args,
)


def _game(
    event: str,
    moves: str,
    *,
    game_date: str = "2020.01.01",
    white_elo: str = "2500",
    black_elo: str = "2500",
    variant: str | None = None,
) -> str:
    headers = [
        f'[Event "{event}"]',
        '[Site "Test"]',
        f'[Date "{game_date}"]',
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


def _events(path: Path) -> list[str]:
    events: list[str] = []
    with path.open(encoding="utf-8") as stream:
        while game := chess.pgn.read_game(stream):
            events.append(game.headers["Event"])
    return events


class PGNCategoryBuilderTests(unittest.TestCase):
    def test_directory_is_merged_and_all_filters_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            inputs = root / "inputs"
            nested = inputs / "nested"
            nested.mkdir(parents=True)
            (inputs / "a.pgn").write_text(
                _game("First Blitz", "1. e4 e5", game_date="2006.09.14")
                + _game("Boundary Rapid", "1. d4 d5", game_date="2006.09.13")
                + _game("Old Classic", "1. c4 e5", game_date="2000.01.01"),
                encoding="utf-8",
            )
            (nested / "b.PGN").write_text(
                _game("Second Blitz", "1. Nf3 d5", game_date="2006.09.15")
                + _game("Good Rapid", "1. d4 d5 2. c4", game_date="2007.01.01")
                + _game(
                    "Variant Classic",
                    "1. e4 e5",
                    game_date="2007.01.01",
                    variant="Chess960",
                )
                + _game(
                    "Low Classic",
                    "1. e4 e5",
                    game_date="2007.01.01",
                    black_elo="1999",
                )
                + _game("Good Classic", "1. e4 e5 2. Nf3", game_date="2007.01.02"),
                encoding="utf-8",
            )
            (inputs / "ignored.txt").write_text(
                _game("Ignored Rapid", "1. e4"), encoding="utf-8"
            )

            result = PGNCategoryDatasetBuilder(
                BuildConfig(
                    source_path=inputs,
                    target_plies=3,
                    categories=("blitz", "rapid", "classic"),
                    output_dir=root / "out",
                    prefix="local",
                    min_elo=2000,
                    date_after="2006.09.13",
                    date_before="2008-01-01",
                )
            ).build()

            self.assertEqual(result.plies, {"blitz": 4, "rapid": 3, "classic": 3})
            self.assertEqual(_events(result.pgn_paths["blitz"]), ["First Blitz", "Second Blitz"])
            self.assertEqual(_events(result.pgn_paths["rapid"]), ["Good Rapid"])
            self.assertEqual(_events(result.pgn_paths["classic"]), ["Good Classic"])
            self.assertEqual(len(result.source_files), 2)
            self.assertTrue(all(result.target_reached.values()))

            report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["format"], "pgn_category_dataset_report_v1")
            self.assertEqual(report["config"]["date_after"], "2006-09-13")
            self.assertEqual(report["config"]["date_before"], "2008-01-01")
            self.assertEqual(report["sources"][0]["skipped"]["date_not_after"], 2)
            self.assertEqual(report["sources"][1]["skipped"]["variant"], 1)
            self.assertEqual(report["sources"][1]["skipped"]["elo"], 1)
            for category in ("blitz", "rapid", "classic"):
                self.assertEqual(
                    report["categories"][category]["indexed_games"],
                    result.games[category],
                )

    def test_single_file_input_and_unknown_date_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            source = root / "games.pgn"
            source.write_text(
                _game("Unknown Blitz", "1. e4", game_date="????.??.??")
                + _game("Known Blitz", "1. d4", game_date="2020.01.02"),
                encoding="utf-8",
            )
            result = PGNCategoryDatasetBuilder(
                BuildConfig(
                    source_path=source,
                    target_plies=1,
                    categories=("blitz",),
                    output_dir=root / "out",
                    date_after="2020.01.01",
                )
            ).build()

            self.assertEqual(_events(result.pgn_paths["blitz"]), ["Known Blitz"])
            report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["sources"][0]["skipped"]["date_missing_or_invalid"], 1)

    def test_cli_supports_greater_than_and_less_than_aliases(self) -> None:
        args = parse_args(
            [
                "games.pgn",
                "100",
                "--categories",
                "blitz",
                "rapid",
                "classic",
                "--date-greater-than",
                "2006.09.13",
                "--date-less-than",
                "2007-09-13",
            ]
        )
        self.assertEqual(args.date_after.isoformat(), "2006-09-13")
        self.assertEqual(args.date_before.isoformat(), "2007-09-13")

    def test_refuses_existing_output_and_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            source = root / "source.pgn"
            source.write_text(_game("Blitz", "1. e4"), encoding="utf-8")
            output = root / "out"
            output.mkdir()
            (output / "pgn_blitz.pgn").write_text("existing", encoding="utf-8")
            builder = PGNCategoryDatasetBuilder(
                BuildConfig(source, 1, ("blitz",), output_dir=output)
            )
            with self.assertRaises(FileExistsError):
                builder.build()

            empty = root / "empty"
            empty.mkdir()
            empty_builder = PGNCategoryDatasetBuilder(
                BuildConfig(empty, 1, ("blitz",), output_dir=root / "other")
            )
            with self.assertRaises(ValueError):
                empty_builder.build()


if __name__ == "__main__":
    unittest.main()
