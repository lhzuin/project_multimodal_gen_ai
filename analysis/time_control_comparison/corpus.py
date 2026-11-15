"""Validate candidate corpora and freeze a seeded, game-disjoint test sample."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import chess
import chess.pgn

from dataset.offsets import build_pgn_offsets, load_pgn_offsets
from utils.dataset_by_category.categories import get_event_category
from .common import CATEGORIES, Config, atomic_json, digest, sha256


def game_identity(game: chess.pgn.Game) -> str:
    """Metadata-independent identity: normalized initial FEN and complete mainline."""
    return digest([game.board().fen(), [move.uci() for move in game.mainline_moves()]])


def validate_game(game: chess.pgn.Game, config: Config, category: str) -> str | None:
    if game.errors:
        return "parse_error"
    if game.headers.get("Variant", "").strip() or game.board().chess960:
        return "variant"
    if get_event_category(game.headers.get("Event")) != category:
        return "category_mismatch"
    try:
        ratings = [int(game.headers[key]) for key in ("WhiteElo", "BlackElo")]
        issue = int(game.headers["TWICIssue"])
    except (ValueError, KeyError):
        return "missing_or_invalid_metadata"
    if min(ratings) < config.min_elo:
        return "elo"
    if not config.lower_week <= issue <= config.start_week:
        return "issue_outside_range"
    board = game.board()
    if not board.is_valid():
        return "invalid_initial_board"
    moves = list(game.mainline_moves())
    if not moves:
        return "empty_game"
    for move in moves:
        if move == chess.Move.null() or move not in board.legal_moves:
            return "illegal_or_null_move"
        board.push(move)
    return None


def read_game(path: Path, row: dict) -> chess.pgn.Game:
    with path.open("rb") as source:
        source.seek(row["start"])
        raw = source.read(row["end"] - row["start"])
    game = chess.pgn.read_game(io.StringIO(raw.decode("utf-8")))
    if game is None or game.errors:
        raise ValueError(f"invalid indexed game at {path}:{row['start']}")
    return game


def _excluded_identities(paths: tuple[Path, ...]) -> set[str]:
    identities = set()
    for path in paths:
        with path.open(encoding="utf-8-sig") as source:
            while (game := chess.pgn.read_game(source)) is not None:
                if game.errors:
                    raise ValueError(f"cannot reliably audit overlap against malformed PGN: {path}")
                identities.add(game_identity(game))
        print(f"Overlap reference: {path}; {len(identities):,} unique games indexed", flush=True)
    return identities


def prepare(config: Config, *, resume: bool) -> dict:
    """Freeze all selection decisions before examining policy outcomes.

    Full-range builder reports are required, so missing weeks cannot silently
    change the sampling frame. Source indexes are rebuilt by the existing builder.
    """
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report_path = config.corpus_dir / f"{config.prefix}_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings = report["config"]
    if not (report.get("range_complete") and settings.get("all_games_in_range")
            and settings["start_week"] == config.start_week and settings["lower_week"] == config.lower_week
            and settings.get("min_elo") == config.min_elo and not settings.get("include_variant")
            and set(settings["categories"]) == set(CATEGORIES)):
        raise ValueError("builder report must certify the complete requested range, all three categories, min Elo and variant exclusion")
    sources = {"builder_report": sha256(report_path)}
    for category in CATEGORIES:
        for suffix in (f"_{category}.pgn", f"_{category}_index.json"):
            path = config.corpus_dir / (config.prefix + suffix)
            sources[category + suffix] = sha256(path)
    sources.update({f"excluded_pgn_{i}": sha256(p) for i, p in enumerate(config.exclude_pgns)})
    signature = digest({"sources": sources, "selection": config.selection_settings(),
                        "code": {p.name: sha256(p) for p in (
                            Path(__file__), Path(__file__).with_name("common.py"),
                            Path(__file__).resolve().parents[2] / "utils/dataset_by_category/categories.py")}})
    manifest_path = output / "selection_manifest.json"
    snapshot = output / "selected_games.pgn"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"selection exists: {manifest_path}; use --resume")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["signature"] != signature or sha256(snapshot) != manifest["selected_pgn_sha256"]
                or digest(manifest["games"]) != manifest["games_sha256"]):
            raise ValueError("selection inputs/settings or frozen PGN changed; use a new output directory")
        return manifest
    forbidden = _excluded_identities(config.exclude_pgns)
    counts = {c: Counter() for c in CATEGORIES}
    candidates: dict[str, dict] = {}
    conflicts = set()
    for category in CATEGORIES:
        path = config.corpus_dir / f"{config.prefix}_{category}.pgn"
        index = load_pgn_offsets(str(path.with_name(path.stem + "_index.json")))
        if index["pgn_size"] != path.stat().st_size or index["num_games"] != report["categories"][category]["games"]:
            raise ValueError(f"source index/report mismatch: {path}")
        for row in index["games"]:
            game = read_game(path, row)
            counts[category]["seen"] += 1
            if reason := validate_game(game, config, category):
                counts[category][reason] += 1
                continue
            identity = game_identity(game)
            if identity in forbidden:
                counts[category]["excluded_reference_overlap"] += 1
                continue
            if identity in candidates:
                counts[category]["duplicate"] += 1
                if candidates[identity]["category"] != category:
                    conflicts.add(identity)
                continue
            candidates[identity] = {
                "game_id": identity, "category": category, "source_path": str(path),
                "source_start": row["start"], "source_end": row["end"],
                "rank": hashlib.sha256(f"{config.seed}:{identity}".encode()).hexdigest(),
                "game_plies": len(list(game.mainline_moves())), "initial_fen": game.board().fen(),
                "headers": dict(game.headers),
            }
        print(f"Audited {category}: {counts[category]['seen']:,} source games", flush=True)
    pools = {c: sorted((r for gid, r in candidates.items() if gid not in conflicts and r["category"] == c),
                       key=lambda row: (row["rank"], row["game_id"])) for c in CATEGORIES}
    audit = {
        "sources": sources, "settings": config.selection_settings(),
        "filters": {c: dict(v) for c, v in counts.items()},
        "cross_category_duplicate_conflicts_excluded": len(conflicts),
        "eligible_unique_games": {c: len(v) for c, v in pools.items()},
        "shortfalls": {c: config.games_per_category - len(v) for c, v in pools.items() if len(v) < config.games_per_category},
        "overlap_basis": "canonical identity exclusion" if config.exclude_pgns else "temporally separated TWIC source issues; training identities not scanned",
    }
    atomic_json(output / "corpus_audit.json", audit)
    if any(not v for v in pools.values()) or (audit["shortfalls"] and not config.allow_shortfall):
        raise ValueError(f"insufficient distinct eligible games: {audit['eligible_unique_games']}; see corpus_audit.json. No earlier issues or duplicate games were substituted.")
    selected = [row for c in CATEGORIES for row in pools[c][:config.games_per_category]]
    temporary = snapshot.with_suffix(".pgn.tmp")
    with temporary.open("wb") as destination:
        for row in selected:
            game = read_game(Path(row["source_path"]), {"start": row["source_start"], "end": row["source_end"]})
            row["start"] = destination.tell()
            text = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False, columns=None))
            destination.write((text + "\n\n").encode("utf-8"))
            row["end"] = destination.tell()
    temporary.replace(snapshot)
    build_pgn_offsets(str(snapshot), str(output / "selected_games_index.json"))
    manifest = {"signature": signature, "selected_pgn_sha256": sha256(snapshot), "games_sha256": digest(selected),
                "settings": config.selection_settings(), "sources": sources, "audit": audit,
                "games": selected,
                "selection": "seeded SHA-256 rank over canonical game identity; without replacement within category"}
    atomic_json(manifest_path, manifest)
    return manifest
