#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Make sure the project root is importable even when the script lives in utils/
PROJECT_ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_FILE))

from dataset.offsets import ensure_offsets
from dataset.dataset import ChessGameSampleDataset

DEFAULT_START = 1484
DEFAULT_END = 1633
DEFAULT_RESOLUTION = 256
DEFAULT_SAMPLE_RATIO = 0.10
DEFAULT_MAX_POSITIONS_PER_GAME = 32
DEFAULT_SEED = 42
DEFAULT_CACHE_SIZE = 64
TWIC_BASE_URL = "https://theweekinchess.com/assets/files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the main training artifacts for the chess project: "
            "merged PGN, PGN byte-offset index, and sampled positions JSON."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_FROM_FILE,
        help="Project root. Defaults to the parent of utils/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the PGN and index will be written. "
            "Defaults to <project-root>/processed_games."
        ),
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the samples JSON will be written. "
            "Defaults to <project-root>/dataset to match dataset/new_test.py."
        ),
    )
    parser.add_argument(
        "--twic-start",
        type=int,
        default=DEFAULT_START,
        help=f"First TWIC issue to use (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--twic-end",
        type=int,
        default=DEFAULT_END,
        help=f"Last TWIC issue to use (default: {DEFAULT_END}).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing TWIC zips. Reuse an existing PGN or existing zips only.",
    )
    parser.add_argument(
        "--pgn-path",
        type=Path,
        default=None,
        help="Existing PGN to reuse directly. When provided, no TWIC merge is done.",
    )
    parser.add_argument(
        "--zips-dir",
        type=Path,
        default=None,
        help="Directory where TWIC zip files are stored. Defaults to <project-root>/twic_zips.",
    )
    parser.add_argument(
        "--pgn-name",
        type=str,
        default="games.pgn",
        help="Output PGN filename (default: games.pgn).",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="games_index.json",
        help="Output offsets filename (default: games_index.json).",
    )
    parser.add_argument(
        "--samples-name",
        type=str,
        default=None,
        help=(
            "Output sampled positions filename. If omitted, a name is generated from "
            "seed/sample_ratio/max_positions_per_game."
        ),
    )
    parser.add_argument(
        "--sprites-dir",
        type=Path,
        default=None,
        help="Sprites directory used by ChessGameSampleDataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Rendered board resolution (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=DEFAULT_SAMPLE_RATIO,
        help=f"Per-game sampling ratio (default: {DEFAULT_SAMPLE_RATIO}).",
    )
    parser.add_argument(
        "--max-positions-per-game",
        type=int,
        default=DEFAULT_MAX_POSITIONS_PER_GAME,
        help=f"Maximum sampled positions per game (default: {DEFAULT_MAX_POSITIONS_PER_GAME}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the samples file (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f"Dataset cache size used during samples creation (default: {DEFAULT_CACHE_SIZE}).",
    )
    parser.add_argument(
        "--apply-augmentations",
        action="store_true",
        help="Instantiate the dataset with augmentations enabled.",
    )
    parser.add_argument(
        "--force-rebuild-pgn",
        action="store_true",
        help="Rebuild the merged PGN even if it already exists.",
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Delete the existing index before rebuilding it.",
    )
    parser.add_argument(
        "--force-rebuild-samples",
        action="store_true",
        help="Delete the existing samples JSON before rebuilding it.",
    )
    return parser.parse_args()



def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or (project_root / "processed_games")).resolve()
    samples_dir = (args.samples_dir or (project_root / "dataset")).resolve()
    zips_dir = (args.zips_dir or (project_root / "twic_zips")).resolve()
    sprites_dir = (args.sprites_dir or (project_root / "dataset" / "sprites")).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)

    pgn_path = output_dir / args.pgn_name
    index_path = output_dir / args.index_name

    if args.samples_name is None:
        ratio_str = str(args.sample_ratio)
        samples_name = f"games_samples_s{args.seed}_r{ratio_str}_m{args.max_positions_per_game}.json"
    else:
        samples_name = args.samples_name
    samples_path = samples_dir / samples_name

    return {
        "project_root": project_root,
        "output_dir": output_dir,
        "samples_dir": samples_dir,
        "zips_dir": zips_dir,
        "sprites_dir": sprites_dir,
        "pgn_path": pgn_path,
        "index_path": index_path,
        "samples_path": samples_path,
    }



def twic_numbers(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError(f"Invalid TWIC range: start={start}, end={end}")
    return list(range(start, end + 1))



def twic_zip_name(issue: int) -> str:
    return f"twic{issue}g.zip"



def twic_zip_path(zips_dir: Path, issue: int) -> Path:
    return zips_dir / twic_zip_name(issue)



def twic_zip_url(issue: int) -> str:
    return f"{TWIC_BASE_URL}/{twic_zip_name(issue)}"



def ensure_twic_zips(start: int, end: int, zips_dir: Path, skip_download: bool) -> list[Path]:
    zip_paths: list[Path] = []
    missing: list[int] = []

    for issue in twic_numbers(start, end):
        path = twic_zip_path(zips_dir, issue)
        zip_paths.append(path)
        if not path.exists():
            missing.append(issue)

    if missing and skip_download:
        missing_preview = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Missing TWIC zip files while --skip-download is enabled. "
            f"First missing issues: {missing_preview}{suffix}\n"
            f"Expected location: {zips_dir}"
        )

    for issue in missing:
        url = twic_zip_url(issue)
        out_path = twic_zip_path(zips_dir, issue)
        print(f"Downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)

    return zip_paths



def merge_pgns_from_zips(zip_paths: list[Path], out_pgn_path: Path) -> None:
    out_pgn_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pgn_path.open("wb") as fout:
        for zip_path in zip_paths:
            if not zip_path.exists():
                raise FileNotFoundError(f"TWIC zip not found: {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [
                    name for name in zf.namelist()
                    if name.lower().endswith(".pgn") and not name.endswith("/")
                ]
                if not members:
                    print(f"Warning: no PGN found inside {zip_path.name}")
                    continue

                members.sort()
                for member in members:
                    print(f"Appending {zip_path.name}:{member}")
                    with zf.open(member, "r") as src:
                        shutil.copyfileobj(src, fout)
                    fout.write(b"\n\n")



def build_or_reuse_pgn(args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    pgn_path = paths["pgn_path"]

    if args.pgn_path is not None:
        source_pgn = args.pgn_path.resolve()
        if not source_pgn.exists():
            raise FileNotFoundError(f"PGN not found: {source_pgn}")
        if source_pgn != pgn_path:
            shutil.copy2(source_pgn, pgn_path)
            print(f"Copied existing PGN to: {pgn_path}")
        else:
            print(f"Reusing existing PGN at: {pgn_path}")
        return pgn_path

    if pgn_path.exists() and not args.force_rebuild_pgn:
        print(f"Merged PGN already exists, reusing: {pgn_path}")
        return pgn_path

    zip_paths = ensure_twic_zips(
        start=args.twic_start,
        end=args.twic_end,
        zips_dir=paths["zips_dir"],
        skip_download=args.skip_download,
    )
    print(f"Merging {len(zip_paths)} TWIC zip files into {pgn_path}...")
    merge_pgns_from_zips(zip_paths, pgn_path)
    print(f"Merged PGN written to: {pgn_path}")
    return pgn_path



def build_index(index_path: Path, pgn_path: Path, force_rebuild: bool) -> None:
    if force_rebuild and index_path.exists():
        index_path.unlink()
    print("Ensuring PGN byte-offset index...")
    ensure_offsets(str(pgn_path), str(index_path))
    print(f"Index ready at: {index_path}")



def build_samples(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.force_rebuild_samples and paths["samples_path"].exists():
        paths["samples_path"].unlink()

    print("Instantiating ChessGameSampleDataset to force/create the samples file...")
    ds = ChessGameSampleDataset(
        pgn_path=str(paths["pgn_path"]),
        index_path=str(paths["index_path"]),
        sprites_dir=str(paths["sprites_dir"]),
        resolution=args.resolution,
        sample_ratio=args.sample_ratio,
        max_positions_per_game=args.max_positions_per_game,
        seed=args.seed,
        apply_augmentations=args.apply_augmentations,
        return_fen=False,
        samples_path=str(paths["samples_path"]),
        cache_size=args.cache_size,
    )

    dataset_len = len(ds)
    print(f"Samples file ready at: {paths['samples_path']}")
    print(f"Dataset length after sampling: {dataset_len}")



def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    print("=== Preparing training files ===")
    print(f"Project root : {paths['project_root']}")
    print(f"PGN/index dir: {paths['output_dir']}")
    print(f"Samples dir  : {paths['samples_dir']}")
    print(f"TWIC zips dir: {paths['zips_dir']}")
    print(f"Sprites dir  : {paths['sprites_dir']}")

    if not paths["sprites_dir"].exists():
        raise FileNotFoundError(
            f"Sprites directory not found: {paths['sprites_dir']}\n"
            "Pass --sprites-dir explicitly if your project layout is different."
        )

    pgn_path = build_or_reuse_pgn(args, paths)
    build_index(paths["index_path"], pgn_path, force_rebuild=args.force_rebuild_index)
    build_samples(args, paths)

    print("\nDone. Generated files:")
    print(f"- PGN          : {paths['pgn_path']}")
    print(f"- games_index  : {paths['index_path']}")
    print(f"- games_samples: {paths['samples_path']}")


if __name__ == "__main__":
    main()
