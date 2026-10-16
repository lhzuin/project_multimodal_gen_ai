"""Streaming mixed-position builder, frozen-policy filtering, and provenance."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import torch

from train.rl.policies import load_policy, select_device
from train.rl.config_io import dump_config
from .config import BUCKETS, BuildConfig, resolve_sources
from .positions import (
    Candidate,
    digest,
    game_fingerprint,
    position_key,
    puzzle_candidate,
    quiet_candidate,
)
from .sources import candidate_rows, exclusion_positions, iter_selected_rows


LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def split_for_game(game_hash: str, config: BuildConfig) -> str:
    value = int(digest(f"{config.seed}|split|{game_hash}")[:16], 16) / 2**64
    total = sum(config.split_sizes.values())
    boundary = 0.0
    for split, size in config.split_sizes.items():
        boundary += size / total
        if value < boundary:
            return split
    return "test"


def training_config(config: BuildConfig) -> dict[str, Any]:
    """A runnable conservative first experiment, using the established contract."""
    output = Path(config.output_dir)
    return {
        "database": {"manifest": str(output / "train.json")},
        "checkpoint_path": config.confidence_checkpoint,
        "model_type": "encoder",
        "model_kwargs": {
            "version": 2,
            "img_size": 256,
            "encoder_dim": 256,
            "n_heads": 4,
            "n_layers": 8,
            "ep_embed_dim": 16,
            "piece_embed_dim": 32,
        },
        "server": {
            "mode": "local",
            "opponent": {
                "executable": config.stockfish,
                "workers": 3,
                "threads_per_engine": 1,
                "hash_mb_per_engine": 64,
                "limit_strength": True,
                "uci_elo": 2100,
                "default_limit": {"movetime_ms": 50},
            },
            "evaluator": {
                "executable": config.stockfish,
                "workers": 1,
                "threads_per_engine": 1,
                "hash_mb_per_engine": 128,
                "default_limit": {"nodes": 100000},
            },
        },
        "rloo": {
            "training_batch_size": 8,
            "rollouts_per_entry": 4,
            "max_concurrent_games": 8,
            "sampling_batch_size": 2,
            "micro_batch_size": 1,
            "ply_depth": 10,
            "randomize_policy_color": False,
            "temperature": 0.8,
            "learning_rate": 5e-6,
            "max_grad_norm": 1.0,
            "kl_beta": 0.04,
            "reward_mode": "tanh",
            "reward_scale_cp": 400.0,
            "device": "auto",
            "seed": config.seed,
        },
        "epochs": 1,
        "save_every_batches": 10,
        "logging": {
            "jsonl_path": str(output / "training.metrics.jsonl"),
            "wandb": {"enabled": False, "project": "chess_vision"},
        },
        "output_checkpoint": "checkpoints/rloo_encoder_balanced_v1_lichess_mixed.latest.pt",
        "export_weights": "checkpoints/rloo_encoder_balanced_v1_lichess_mixed.pt",
    }


class ConfidenceFilter:
    """Use precisely the encoder adapter's full legal action distribution."""

    def __init__(self, config: BuildConfig) -> None:
        self.config = config
        torch.set_num_threads(config.torch_threads)
        self.policy = load_policy(
            config.confidence_checkpoint,
            "encoder",
            {
                "version": 2,
                "img_size": 256,
                "encoder_dim": 256,
                "n_heads": 4,
                "n_layers": 8,
                "ep_embed_dim": 16,
                "piece_embed_dim": 32,
            },
        )
        self.policy.model.to(select_device(config.device)).eval().requires_grad_(False)

    def score(self, candidates: list[Candidate]) -> list[dict[str, float]]:
        observations = [
            {
                "fen": c.board.fen(),
                "legal_moves_uci": [move.uci() for move in c.board.legal_moves],
            }
            for c in candidates
        ]
        with torch.no_grad():
            distributions = self.policy.log_distributions(
                observations, self.config.confidence_temperature
            )
            return [
                {
                    "max_probability": float(log.exp().max().item()),
                    "entropy_nats": float(-(log.exp() * log).sum().item()),
                }
                for log in distributions
            ]


class DatasetBuilder:
    def __init__(self, config: BuildConfig, confidence_filter: Any = None) -> None:
        self.config = config
        self.filter = confidence_filter
        self.counts: Counter = Counter()

    def build(self) -> dict[str, Any]:
        config = self.config
        output = Path(config.output_dir)
        if output.exists():
            raise FileExistsError(f"refusing to replace existing dataset: {output}")
        sources = resolve_sources(config)
        started = datetime.now(timezone.utc).isoformat()
        begin = time.monotonic()
        LOGGER.info("Reading exclusion suites")
        excluded = exclusion_positions(config.exclusion_openings)
        LOGGER.info("Scanning metadata for reproducible candidate reservoirs")
        selected = candidate_rows(sources, config, self.counts)
        LOGGER.info("Candidate rows: %d; loading frozen encoder", len(selected))
        confidence = self.filter or ConfidenceFilter(config)
        quotas = config.quotas()
        filled: Counter = Counter()
        accepted: dict[str, list[dict[str, Any]]] = {
            split: [] for split in config.split_sizes
        }
        audits = []
        used_positions: set[str] = set()
        used_games: set[str] = set()
        used_ids: set[str] = set()
        pending: list[Candidate] = []

        def accept_batch() -> None:
            if not pending:
                return
            scores = confidence.score(pending)
            for candidate, score in zip(pending, scores, strict=True):
                self.counts["confidence_evaluated"] += 1
                if score["max_probability"] > config.confidence_max_probability:
                    self.counts["overconfident_rejected"] += 1
                    continue
                split = split_for_game(candidate.game_hash, config)
                color = "white" if candidate.board.turn else "black"
                key = (split, candidate.bucket, color)
                pos_key = position_key(candidate.board)
                if filled[key] >= quotas[key]:
                    continue
                if (
                    candidate.game_hash in used_games
                    or candidate.source_game_id in used_ids
                ):
                    self.counts["duplicate_game_rejected"] += 1
                    continue
                if pos_key in used_positions or pos_key in excluded:
                    self.counts["duplicate_or_excluded_position_rejected"] += 1
                    continue
                try:
                    candidate.prompt.validate(
                        "decoder"
                    )  # same manifest supports both model types
                except ValueError:
                    self.counts["invalid_or_claimable_draw_prompt_rejected"] += 1
                    continue
                prompt = asdict(candidate.prompt)
                accepted[split].append(prompt)
                audits.append(
                    {**candidate.audit, **score, "split": split, "policy_color": color}
                )
                filled[key] += 1
                used_games.add(candidate.game_hash)
                used_ids.add(candidate.source_game_id)
                used_positions.add(pos_key)
            pending.clear()

        for source_index, row_index, row, buckets in iter_selected_rows(
            sources, selected
        ):
            self.counts["candidate_source_rows_visited"] += 1
            identity = game_fingerprint(row)
            if identity in used_games:
                continue
            split = split_for_game(identity, config)
            for bucket in BUCKETS:
                if bucket not in buckets or all(
                    filled[(split, bucket, color)] >= quotas[(split, bucket, color)]
                    for color in ("white", "black")
                ):
                    continue
                try:
                    candidate = (
                        quiet_candidate(row, config)
                        if bucket == "quiet"
                        else puzzle_candidate(row, bucket)
                    )
                    if (
                        candidate.board.is_game_over(claim_draw=True)
                        or candidate.board.legal_moves.count() < 2
                    ):
                        self.counts["terminal_or_forced_rejected"] += 1
                        continue
                except (ValueError, KeyError, TypeError) as exc:
                    self.counts[f"{bucket}_rejected:{str(exc)[:100]}"] += 1
                    continue
                candidate.audit.update(
                    {"source_index": source_index, "source_row": row_index}
                )
                color = "white" if candidate.board.turn else "black"
                if filled[(split, bucket, color)] >= quotas[(split, bucket, color)]:
                    continue
                if position_key(candidate.board) in excluded:
                    self.counts["opening_suite_position_rejected"] += 1
                    continue
                pending.append(candidate)
                if len(pending) >= config.confidence_batch_size:
                    accept_batch()
            if self.counts["candidate_source_rows_visited"] % 1000 == 0:
                LOGGER.info(
                    "Visited %d candidate rows; accepted %s",
                    self.counts["candidate_source_rows_visited"],
                    {split: len(rows) for split, rows in accepted.items()},
                )
            if all(filled[key] >= target for key, target in quotas.items()):
                break
        accept_batch()
        missing = {
            "/".join(key): target - filled[key]
            for key, target in quotas.items()
            if filled[key] < target
        }
        if missing:
            raise ValueError(
                f"candidate supply exhausted; do not silently change quotas. Missing: {missing}; counts: {dict(self.counts)}"
            )
        for split, rows in accepted.items():
            random.Random(f"{config.seed}|final|{split}").shuffle(rows)
        report = {
            "format": "chess_rl_positions_v1",
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - begin,
            "config": asdict(config),
            "source_files": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
                for path in sources
            ],
            "confidence_checkpoint_sha256": file_hash(
                Path(config.confidence_checkpoint)
            ),
            "exclusion_files": [
                {"path": path, "sha256": file_hash(Path(path))}
                for path in config.exclusion_openings
            ],
            "excluded_position_keys": len(excluded),
            "counts": dict(self.counts),
            "quotas": {"/".join(key): value for key, value in quotas.items()},
            "actual": {"/".join(key): filled[key] for key in quotas},
            "overlap_policy": {
                "rl_game_overlap": 0,
                "rl_anchor_position_overlap": 0,
                "pretraining_overlap": "allowed; not audited or claimed absent",
                "tournament_exclusion": "exact anchor positions from listed opening suites; not full source-game exclusion",
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}-", dir=output.parent
        ) as temporary:
            directory = Path(temporary)

            def write_json(name: str, value: Any) -> None:
                with (directory / name).open("w", encoding="utf-8") as handle:
                    json.dump(
                        value, handle, indent=2, ensure_ascii=False, allow_nan=False
                    )
                    handle.write("\n")

            for split, rows in accepted.items():
                write_json(f"{split}.json", rows)
            with (directory / "audit.jsonl").open("w", encoding="utf-8") as handle:
                for audit in audits:
                    handle.write(
                        json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n"
                    )
            write_json("build_config.json", asdict(config))
            (directory / "training_config.yaml").write_text(
                dump_config(training_config(config)), encoding="utf-8"
            )
            report["training_config"] = {
                "path": str(output / "training_config.yaml"),
                "sha256": file_hash(directory / "training_config.yaml"),
            }
            report["artifact_sha256"] = {
                path.name: file_hash(path) for path in directory.iterdir()
            }
            write_json("manifest.json", report)
            (directory / "REPORT.md").write_text(
                render_report(report), encoding="utf-8"
            )
            directory.rename(output)
        LOGGER.info("Published verified-size dataset at %s", output)
        return report


def render_report(report: dict[str, Any]) -> str:
    config = report["config"]
    counts = report["counts"]
    quota_lines = "\n".join(
        f"| {key.replace('/', ' | ')} | {value} |"
        for key, value in report["actual"].items()
    )
    return f"""# Mixed Lichess RL dataset: reproducible TG2 methods

Built from `{config['dataset_id']}` revision `{config['revision']}` using the
listed Parquet shards. This is a subset of the upstream dataset, not all 14 shards.
Seed: {config['seed']}. Wall time: {report['duration_seconds']:.1f} seconds.

## Selection and rationale

- Exact mix in every split: 50% tactical middlegame puzzles, 30% quiet original-game
  middlegames, 20% endgame puzzles. White/Black to move are equally represented in each bucket.
- Puzzle rating {config['min_puzzle_rating']}–{config['max_puzzle_rating']}, deviation ≤
  {config['max_rating_deviation']}, popularity ≥ {config['min_popularity']}, attempts ≥ {config['min_plays']}.
  Both original players have Elo ≥ {config['min_player_elo']}. Puzzle rating is not playing Elo.
- Puzzle lines have 4–{config['max_solution_plies']} plies including the opponent trigger;
  mate-in-one tasks are excluded. Endgames use upstream endgame themes.
- Quiet candidates use earlier game positions, from ply {config['quiet_min_ply']} to
  min({config['quiet_max_ply']}, 75% of original game length), at least 11 plies from the puzzle anchor.
  The preceding move is not a capture/promotion, the side to move is not in check,
  ≥14 pieces and ≥12 legal moves remain, the inherited evaluation has |CP| ≤
  {config['quiet_max_abs_cp']} and changes by ≤{config['quiet_max_delta_cp']} CP from the preceding annotation.
  No last-move judgment is present. These heuristics do not prove tactical quietness.
- Drawn games are retained. The proposed 80% final-quarter bias is not used because
  this bucket is intended to preserve middlegame coverage. Original game results
  are never included in trainer prompts.
- Candidate selection uses seeded hash reservoirs over all {counts['source_rows_scanned']:,} scanned rows.
  Frozen balanced-v1 encoder confidence is measured at temperature {config['confidence_temperature']}.
  Positions with max legal action probability >{config['confidence_max_probability']} are rejected.
  Evaluated {counts.get('confidence_evaluated', 0):,} candidates; rejected
  {counts.get('overconfident_rejected', 0):,} as overconfident. No new Stockfish searches are used in curation.

## Leakage and overlap rules

Lichess FEN is BEFORE the opponent trigger (`Moves[0]`). We reconstruct the actual
original SAN game prefix, verify that it reaches that FEN, apply and verify the
trigger, and STOP. `Moves[1:]` are checked for legality only, never emitted as
prompt history, metadata, or audit labels. Quiet prompts also end exactly at their
selected position. Full original games and engine annotations are used only for
offline curation and grouping; they are not trainer inputs.

At most one starting position per original game is retained. Game fingerprints
are hashes of normalized original SAN move streams. Split assignment is a seeded
hash of that fingerprint; normalized Lichess IDs provide an additional duplicate
guard. Starting boards are globally deduplicated, including turn/castling/en-passant.
Histories and draw counters are preserved in inputs. This does not claim that
all opening prefixes or future online rollout states are disjoint across splits.

Pretraining overlap is allowed and NOT measured here; results cannot be described
as evaluation on games unseen during supervised training. The listed tournament
opening-suite anchor positions are excluded ({report['excluded_position_keys']:,} distinct keys),
but full tournament source-game exclusion is not claimed. Holdout splits are
never supplied to the training configuration. Quiet states come from puzzle-linked
Lichess games, a selection bias that should be acknowledged in TG2.

## Actual counts

| Split | Bucket | Policy color | Count |
|---|---|---|---:|
{quota_lines}

## Runtime contract and experiment

`train.json`, `validation.json`, and `test.json` contain only PositionPrompt fields:
entry_id, standard start_fen, actual UCI history through the selected moment, metadata.
The model always moves first (`randomize_policy_color=false`). Training uses
8 entries/update, 4 samples/entry, 10 new plies, temperature 0.8, LR 5e-6,
micro-batch 1, one epoch, tanh(CP/400), KL coefficient 0.04.
Opponent: strength-limited Stockfish 2100, 50 ms/move, 3 one-thread workers;
evaluator: unrestricted, 100000 nodes, one worker. The ELO setting is an engine
control, not an independently calibrated opponent rating for these positions.

Inherited annotations have unspecified search budgets and serve only selection.
Online Stockfish supplies fresh RL rewards. Store full-game tournament results
separately; improved training CP alone is not evidence of improved Elo.

`manifest.json` records source/checkpoint hashes, settings, rejection counts and
artifact hashes. `audit.jsonl` records provenance and frozen-model statistics;
it is not part of the training data. See `verification.json` for integrity checks.

Sources: [Lichess puzzle specification](https://database.lichess.org/#puzzles),
[joined game dataset](https://huggingface.co/datasets/Lichess/chess-puzzles-with-games).
"""
