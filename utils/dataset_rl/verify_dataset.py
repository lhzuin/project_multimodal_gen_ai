"""Replay, hash-check, and optionally smoke-train the final RL dataset."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Any

import chess

from train.rl.contracts import PositionPrompt
from train.rl.config_io import load_config
from .builder import file_hash
from .config import BuildConfig
from .positions import position_key, puzzle_candidate, san_tokens
from .sources import METADATA_COLUMNS, exclusion_positions, iter_selected_rows


def training_settings(directory: Path) -> dict[str, Any]:
    """Resolve the manifest's training recipe, including relocated configs."""
    manifest = json.loads((directory / "manifest.json").read_text())
    reference = manifest["training_config"]
    path = Path(reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError("training configuration hash mismatch")
    return load_config(path)


def verify(directory: Path, *, replay_source: bool = True) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = BuildConfig(**manifest["config"])
    for name, expected in manifest["artifact_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    audits = [
        json.loads(line)
        for line in (directory / "audit.jsonl").read_text().splitlines()
    ]
    audit_by_id = {row["entry_id"]: row for row in audits}
    if len(audit_by_id) != len(audits):
        raise ValueError("duplicate audit entry IDs")
    positions, game_hashes, game_ids, entry_ids = set(), set(), set(), set()
    prompts = {}
    actual = Counter()
    selected_rows: dict[tuple[int, int], set[str]] = {}
    excluded = exclusion_positions(config.exclusion_openings)
    for split in config.split_sizes:
        rows = json.loads((directory / f"{split}.json").read_text())
        if len(rows) != config.split_sizes[split]:
            raise ValueError(f"wrong split size: {split}")
        for row in rows:
            if set(row) != {"entry_id", "start_fen", "moves_uci", "metadata"}:
                raise ValueError("unexpected runtime prompt fields")
            if set(row["metadata"]) != {
                "source",
                "bucket",
                "game_id",
                "puzzle_id",
                "selected_ply",
            }:
                raise ValueError(
                    "unexpected metadata; do not pass answers/continuations to training"
                )
            prompt = PositionPrompt.from_mapping(row)
            prompt.validate("encoder")
            prompt.validate("decoder")
            board = prompt.board()
            audit = audit_by_id[prompt.entry_id]
            key = position_key(board)
            if (
                key in positions
                or key in excluded
                or audit["game_fingerprint"] in game_hashes
                or audit["source_game_id"] in game_ids
                or prompt.entry_id in entry_ids
            ):
                raise ValueError("duplicate/excluded position or shared source game")
            if audit["split"] != split or audit["selected_fen"] != board.fen():
                raise ValueError("audit and runtime position disagree")
            if audit["max_probability"] > config.confidence_max_probability:
                raise ValueError("overconfident position escaped selection")
            if audit["bucket"] != "quiet":
                before = chess.Board(audit["pre_trigger_fen"])
                before.push_uci(audit["trigger_uci"])
                if (
                    board.fen() != before.fen()
                    or len(prompt.moves_uci) != before.ply()
                    or prompt.moves_uci[-1] != audit["trigger_uci"]
                ):
                    raise ValueError("trigger alignment or solution-boundary violation")
            color = "white" if board.turn else "black"
            actual[(split, audit["bucket"], color)] += 1
            positions.add(key)
            game_hashes.add(audit["game_fingerprint"])
            game_ids.add(audit["source_game_id"])
            entry_ids.add(prompt.entry_id)
            prompts[prompt.entry_id] = prompt
            selected_rows.setdefault(
                (audit["source_index"], audit["source_row"]), set()
            ).add(audit["bucket"])
    if dict(actual) != config.quotas() or len(prompts) != len(audits):
        raise ValueError("bucket/color quotas or audit coverage disagree")
    replayed = 0
    if replay_source:
        paths = [Path(source["path"]) for source in manifest["source_files"]]
        for path, source in zip(paths, manifest["source_files"], strict=True):
            if file_hash(path) != source["sha256"]:
                raise ValueError("upstream source content changed")
        by_row = {(a["source_index"], a["source_row"]): a for a in audits}
        for source, row_index, row, _ in iter_selected_rows(
            paths, selected_rows, METADATA_COLUMNS + ("moves",)
        ):
            audit = by_row[(source, row_index)]
            prompt = prompts[audit["entry_id"]]
            board = chess.Board()
            prefix = []
            for token in san_tokens(row)[: len(prompt.moves_uci)]:
                move = board.parse_san(token)
                prefix.append(move.uci())
                board.push(move)
            if tuple(prefix) != prompt.moves_uci:
                raise ValueError(
                    "runtime history is not the exact original game prefix"
                )
            if audit["bucket"] != "quiet":
                expected = puzzle_candidate(row, audit["bucket"])
                if expected.prompt.moves_uci != prompt.moves_uci:
                    raise ValueError("puzzle answer was appended or trigger omitted")
            replayed += 1
        if replayed != len(prompts):
            raise ValueError(
                "some accepted source rows were not independently replayed"
            )
    settings = training_settings(directory)
    if settings["rloo"]["randomize_policy_color"] is not False:
        raise ValueError("targeted puzzles require the policy to move first")
    if (
        Path(settings["database"]["manifest"]).resolve()
        != (directory / "train.json").resolve()
    ):
        raise ValueError(
            "training configuration is not pointing at this training split"
        )
    result = {
        "status": "passed",
        "validated_positions": len(prompts),
        "source_rows_independently_replayed": replayed,
        "duplicate_anchor_positions": 0,
        "shared_games_across_splits": 0,
        "solution_continuation_in_inputs": False,
        "policy_moves_first": True,
        "contract": "PositionPrompt for encoder and decoder",
    }
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def smoke_train(directory: Path) -> None:
    """One temporary optimizer update; never overwrite experiment checkpoints."""
    from train.rloo.launch import run

    settings = training_settings(directory)
    settings["max_updates"] = 1
    settings["rloo"]["device"] = "cpu"
    settings["rloo"]["shuffle"] = False
    with tempfile.TemporaryDirectory(prefix="rl-dataset-smoke-") as temporary:
        settings["output_checkpoint"] = str(Path(temporary) / "latest.pt")
        settings["export_weights"] = str(Path(temporary) / "weights.pt")
        settings["logging"] = {"jsonl_path": str(Path(temporary) / "metrics.jsonl")}
        await run(settings)
    result = json.loads((directory / "verification.json").read_text())
    result["real_stockfish_rloo_optimizer_smoke_test"] = (
        "passed: one update on training split; temporary outputs"
    )
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("processed_games/rl/lichess_mixed_v1")
    )
    parser.add_argument("--skip-source-replay", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.dataset, replay_source=not args.skip_source_replay), indent=2
        )
    )
    if args.smoke_train:
        asyncio.run(smoke_train(args.dataset))


if __name__ == "__main__":
    main()
