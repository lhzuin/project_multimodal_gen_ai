"""Evaluate four frozen chess checkpoints on disjoint TWIC 1662–1667 games.

Run after the category builder's --all-games-in-range command. The output is
one row per model and human move, paired summaries, a provenance manifest,
and SVG figures. No validation split or training loss mask is used here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

from analysis.ply_comparison.aggregate import (
    ComparisonAccumulator, DELTA_FIELDS, RELATIVE_BUCKETS, SUMMARY_FIELDS,
    bucket_for_ply, relative_bucket_for_ply, write_summary_csv,
)
from analysis.ply_comparison.plot_results import MODELS, main as plot_main
from dataset.offsets import load_pgn_offsets
from dataset.renderer import board_to_grid_ids
from models.chess_decoder_player import ChessDecoderPlayer
from models.chess_encoder_player import ChessEncoderPlayerV2


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "encoder_baseline": ROOT / "chess_encoder_seq_v1_07_03_FINAL.pt",
    "encoder_balanced": ROOT / "chess_encoder_seq_balanced_v1.best.pt",
    "decoder_baseline": ROOT / "chess_llm_decoder_v12.pt",
    "decoder_balanced": ROOT / "chess_llm_decoder_balanced_v1_final.pt",
}
PER_MOVE_FIELDS = (
    "game_id", "ply", "game_plies", "bucket", "relative_bucket", "model_id", "twic_issue", "date",
    "white_elo", "black_elo", "fen_before", "target_uci", "predicted_uci",
    "top1_correct", "top5_correct", "nll_legal", "p_target_legal",
    "n_legal", "context_len", "context_truncated", "pair_correct",
)


@dataclass
class Example:
    game_id: int
    ply: int
    game_plies: int
    issue: int
    date: str
    white_elo: int
    black_elo: int
    fen: str
    target: chess.Move
    legal_moves: list[chess.Move]
    labels64: object
    turn: int
    castling: tuple[int, int, int, int]
    ep_square: int
    move_ids: list[int]
    piece_ids: list[int]
    start_turn: int
    context_truncated: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_weights(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_models(device: torch.device, tokenizer_path: Path):
    result = {}
    for model_id in MODELS:
        if model_id.startswith("encoder"):
            model = ChessEncoderPlayerV2(
                img_size=256, piece_embed_dim=32, ep_embed_dim=16,
                encoder_dim=256, n_heads=4, n_layers=8,
                encoder_dropout=0.1, vit_path=None, freeze_vit=True,
            )
        else:
            model = ChessDecoderPlayer(
                tokenizer_path=str(tokenizer_path), model_dim=256,
                mlp_ratio=4.0, n_heads=4, n_layers=12, dropout=0.1,
                max_seq_len=256, tie_weights=True,
            )
        weights = load_weights(CHECKPOINTS[model_id])
        model.load_state_dict(weights, strict=True)
        result[model_id] = model.eval().to(device)
        print(f"Loaded {model_id}: {len(weights)} tensors", flush=True)
    return result


def game_examples(game: chess.pgn.Game, game_id: int, tokenizer, *, max_context: int = 256):
    issue = int(game.headers["TWICIssue"])
    if not 1662 <= issue <= 1667:
        raise ValueError(f"game {game_id}: unexpected TWIC issue {issue}")
    white_elo = int(game.headers["WhiteElo"])
    black_elo = int(game.headers["BlackElo"])
    if white_elo < 1600 or black_elo < 1600:
        raise ValueError(f"game {game_id}: Elo filter mismatch")
    board = game.board()
    initial_turn = 0 if board.turn == chess.WHITE else 1
    move_ids = [tokenizer.bos_id]
    piece_ids = [tokenizer.p_bos_id]
    examples = []
    moves = list(game.mainline_moves())
    if not moves or game.errors:
        raise ValueError(f"game {game_id}: empty or invalid PGN")
    for ply, move in enumerate(moves):
        if move == chess.Move.null() or move not in board.legal_moves:
            raise ValueError(f"game {game_id}, ply {ply}: illegal or null move")
        target_uci = move.uci()
        if target_uci not in tokenizer.move2id:
            raise ValueError(f"game {game_id}, ply {ply}: UCI absent from tokenizer: {target_uci}")
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError(f"game {game_id}, ply {ply}: no moved piece")
        start = max(0, len(move_ids) - max_context)
        ply_before = max(0, start - 1)
        castling = (
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        )
        examples.append(Example(
            game_id=game_id, ply=ply, game_plies=len(moves), issue=issue,
            date=game.headers.get("Date", ""), white_elo=white_elo,
            black_elo=black_elo, fen=board.fen(), target=move,
            legal_moves=list(board.legal_moves),
            labels64=board_to_grid_ids(board).reshape(-1),
            turn=int(board.turn == chess.WHITE), castling=castling,
            ep_square=board.ep_square if board.ep_square is not None else -1,
            move_ids=move_ids[start:].copy(), piece_ids=piece_ids[start:].copy(),
            start_turn=(initial_turn + ply_before) % 2,
            context_truncated=start > 0,
        ))
        move_ids.append(tokenizer.move2id[target_uci])
        piece_ids.append(tokenizer.piece2id[piece.symbol().upper()])
        board.push(move)
    return examples


def _log_softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    total = sum(math.exp(value - maximum) for value in values)
    normalizer = maximum + math.log(total)
    return [value - normalizer for value in values]


def encoder_result(example: Example, policy, promo) -> tuple[str, list[str], float, bool]:
    # One probability per legal (from,to) pair. Promotions from the same pair
    # share its mass, conditional on the promotion head's legal classes.
    pairs: dict[tuple[int, int], list[chess.Move]] = {}
    for move in example.legal_moves:
        pairs.setdefault((move.from_square, move.to_square), []).append(move)
    keys = list(pairs)
    pair_logp = _log_softmax([float(policy[frm, to]) for frm, to in keys])
    move_logp: dict[str, float] = {}
    for pair, base_logp in zip(keys, pair_logp):
        candidates = pairs[pair]
        if len(candidates) == 1:
            move_logp[candidates[0].uci()] = base_logp
        else:
            promo_classes = {chess.KNIGHT: 1, chess.BISHOP: 2,
                             chess.ROOK: 3, chess.QUEEN: 4}
            conditional = _log_softmax([
                float(promo[promo_classes[move.promotion]]) for move in candidates
            ])
            for move, conditional_logp in zip(candidates, conditional):
                move_logp[move.uci()] = base_logp + conditional_logp

    # Match ChessEncoderPlayerV2.sample_moves(greedy=True), including its
    # first-legal-candidate fallback when class 0 is selected at a promotion.
    best_pair = keys[max(range(len(keys)), key=lambda i: pair_logp[i])]
    candidates = pairs[best_pair]
    if len(candidates) == 1:
        greedy = candidates[0]
    else:
        preferred = max(range(5), key=lambda i: float(promo[i]))
        promo_piece = {1: chess.KNIGHT, 2: chess.BISHOP,
                       3: chess.ROOK, 4: chess.QUEEN}.get(preferred)
        greedy = next((move for move in candidates if move.promotion == promo_piece),
                      candidates[0])
    greedy_uci = greedy.uci()
    ranking = [greedy_uci] + [
        uci for uci in sorted(move_logp, key=move_logp.get, reverse=True)
        if uci != greedy_uci
    ]
    target_pair = (example.target.from_square, example.target.to_square)
    return greedy_uci, ranking[:5], -move_logp[example.target.uci()], best_pair == target_pair


def decoder_result(example: Example, logits, tokenizer) -> tuple[str, list[str], float]:
    uci = [move.uci() for move in example.legal_moves]
    scores = [float(logits[tokenizer.move2id[move]]) for move in uci]
    logp = _log_softmax(scores)
    best_index = max(range(len(uci)), key=lambda i: scores[i])
    greedy = uci[best_index]
    ranking = [greedy] + [
        uci[i] for i in sorted(range(len(uci)), key=lambda i: logp[i], reverse=True)
        if i != best_index
    ]
    target_index = uci.index(example.target.uci())
    return greedy, ranking[:5], -logp[target_index]


@torch.inference_mode()
def cached_decoder_logits(examples: list[Example], models, device):
    """One causal pass scores all pre-truncation prefixes in a game exactly."""
    last_ply = min(len(examples), 256) - 1
    prefix = examples[last_ply]
    ids = torch.tensor([prefix.move_ids], dtype=torch.long, device=device)
    pieces = torch.tensor([prefix.piece_ids], dtype=torch.long, device=device)
    turn = torch.tensor([prefix.start_turn], dtype=torch.long, device=device)
    result = {}
    for model_id in ("decoder_baseline", "decoder_balanced"):
        output = models[model_id](input_ids=ids, piece_input_ids=pieces,
                                  start_turn=turn)
        result[model_id] = output["logits"][0].detach().float().cpu()
    return result


@torch.inference_mode()
def evaluate_batch(examples: list[Example], models, device, writer, accumulator,
                   relative_accumulator,
                   decoder_cache):
    labels = torch.as_tensor(np.stack([item.labels64 for item in examples]),
                             dtype=torch.long, device=device)
    turn = torch.tensor([item.turn for item in examples], dtype=torch.long, device=device)
    castling = torch.tensor([item.castling for item in examples], dtype=torch.long, device=device)
    ep_square = torch.tensor([item.ep_square for item in examples], dtype=torch.long, device=device)

    for model_id in MODELS:
        model = models[model_id]
        if model_id.startswith("encoder"):
            output = model(labels64=labels, turn=turn, castling=castling, ep_square=ep_square)
            policy = output["policy_logits"].detach().float().cpu()
            promo = output["promo_logits"].detach().float().cpu()
        else:
            # Most games fit wholly within 256 tokens. For them, a single
            # teacher-forced causal pass per game yields every next-move logit.
            logits = [decoder_cache[model_id][item.ply] if item.ply < 256 else None
                      for item in examples]
            late = [i for i, item in enumerate(examples) if item.ply >= 256]
            if late:
                length = max(len(examples[i].move_ids) for i in late)
                ids = torch.zeros((len(late), length), dtype=torch.long, device=device)
                pieces = torch.zeros_like(ids)
                attention = torch.zeros_like(ids, dtype=torch.bool)
                starts = torch.tensor([examples[i].start_turn for i in late],
                                      dtype=torch.long, device=device)
                last = []
                for j, i in enumerate(late):
                    item = examples[i]
                    n = len(item.move_ids)
                    ids[j, :n] = torch.tensor(item.move_ids, dtype=torch.long, device=device)
                    pieces[j, :n] = torch.tensor(item.piece_ids, dtype=torch.long, device=device)
                    attention[j, :n] = True
                    last.append(n - 1)
                output = model(input_ids=ids, piece_input_ids=pieces,
                               attention_mask=attention, start_turn=starts)
                late_logits = output["logits"][torch.arange(len(late), device=device), last]
                for j, i in enumerate(late):
                    logits[i] = late_logits[j].detach().float().cpu()
            tokenizer = model.decoder.tokenizer
        for i, item in enumerate(examples):
            if model_id.startswith("encoder"):
                greedy, top5, nll, pair_correct = encoder_result(item, policy[i], promo[i])
            else:
                greedy, top5, nll = decoder_result(item, logits[i], tokenizer)
                pair_correct = ""
            target = item.target.uci()
            top1_correct = greedy == target
            top5_correct = target in top5
            accumulator.add(item.game_id, item.ply, model_id,
                            top1=top1_correct, top5=top5_correct, nll=nll)
            relative_accumulator.add_bucket(
                item.game_id, relative_bucket_for_ply(item.ply, item.game_plies),
                model_id, top1=top1_correct, top5=top5_correct, nll=nll,
            )
            writer.writerow({
                "game_id": item.game_id, "ply": item.ply,
                "game_plies": item.game_plies,
                "bucket": bucket_for_ply(item.ply),
                "relative_bucket": relative_bucket_for_ply(item.ply, item.game_plies),
                "model_id": model_id,
                "twic_issue": item.issue, "date": item.date,
                "white_elo": item.white_elo, "black_elo": item.black_elo,
                "fen_before": item.fen, "target_uci": target,
                "predicted_uci": greedy, "top1_correct": int(top1_correct),
                "top5_correct": int(top5_correct), "nll_legal": nll,
                "p_target_legal": math.exp(-nll),
                "n_legal": len(item.legal_moves),
                "context_len": len(item.move_ids) if model_id.startswith("decoder") else "",
                "context_truncated": int(item.context_truncated) if model_id.startswith("decoder") else "",
                "pair_correct": int(pair_correct) if model_id.startswith("encoder") else "",
            })


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_issue_sample(pgn_path: Path, index: dict, games_per_issue: int) -> set[int]:
    """Choose a reproducible, uniform hash-ranked subset within each issue."""
    if games_per_issue < 1:
        raise ValueError("sample-games-per-issue must be positive")
    candidates: dict[int, list[tuple[bytes, int]]] = {issue: [] for issue in range(1662, 1668)}
    with pgn_path.open("rb") as source:
        for row in index["games"]:
            source.seek(int(row["start"]))
            raw = source.read(int(row["end"]) - int(row["start"]))
            headers = chess.pgn.read_headers(io.StringIO(raw.decode("utf-8", errors="replace")))
            if headers is None:
                raise ValueError(f"cannot read headers for indexed game {row['idx']}")
            issue = int(headers["TWICIssue"])
            if issue not in candidates:
                raise ValueError(f"unexpected issue {issue} in the evaluation corpus")
            rank = hashlib.sha256(raw).digest()
            candidates[issue].append((rank, int(row["idx"])))
    if any(not rows for rows in candidates.values()):
        raise ValueError("at least one TWIC issue has no eligible games")
    selected = set()
    for rows in candidates.values():
        selected.update(game_id for _, game_id in sorted(rows)[:games_per_issue])
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_classic.pgn")
    parser.add_argument("--index", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_classic_index.json")
    parser.add_argument("--builder-report", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_report.json")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/chess_uci_vocab.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/twic_1662_1667")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-games", type=int, default=None, help="Smoke test: stop after this many accepted games.")
    parser.add_argument("--sample-games-per-issue", type=int, default=None,
                        help="Exploratory run: select this many hash-ranked games from each issue.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-csv", type=Path, action="append", help="Frozen partial per_move.csv; repeat to merge sources, first source wins whole-game overlaps.")
    parser.add_argument("--resume-source", type=Path, action="append", help="Matching provenance JSON for each resume-csv, in the same order.")
    parser.add_argument("--resume-code-review", type=Path, help="Local reviewed source equivalence receipt for explicitly inspected source differences.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1 or args.bootstrap < 1:
        raise ValueError("batch-size and bootstrap must be positive")
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("max-games must be positive")
    if args.max_games is not None and args.sample_games_per_issue is not None:
        raise ValueError("max-games and sample-games-per-issue cannot be combined")
    if bool(args.resume_csv) != bool(args.resume_source):
        raise ValueError("resume-csv and resume-source must be supplied together")
    if args.resume_csv and len(args.resume_csv) != len(args.resume_source):
        raise ValueError("each resume-csv needs a matching resume-source")
    if args.resume_csv and len({p.resolve() for p in args.resume_csv}) != len(args.resume_csv):
        raise ValueError("duplicate resume CSV paths")
    if args.resume_csv and (args.max_games is not None or args.sample_games_per_issue is not None):
        raise ValueError("recovery requires a full-corpus evaluation, without sampling/game caps")
    if args.resume_csv and args.overwrite:
        raise ValueError("recovery requires a fresh output directory; overwrite could destroy the source CSV")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device_name)
    report = json.loads(args.builder_report.read_text(encoding="utf-8"))
    config = report["config"]
    if (not config.get("all_games_in_range") or config["start_week"] != 1667
            or config["lower_week"] != 1662 or not report.get("range_complete")
            or "classic" not in config["categories"] or config.get("min_elo") != 1600):
        raise ValueError("builder report does not match the complete TWIC 1662–1667 protocol")
    index = load_pgn_offsets(str(args.index))
    if index["num_games"] != report["categories"]["classic"]["games"]:
        raise ValueError("PGN index/game count mismatch")
    selected_game_ids = (select_issue_sample(args.pgn, index, args.sample_games_per_issue)
                         if args.sample_games_per_issue is not None else None)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(device, args.tokenizer)
    tokenizer = models["decoder_baseline"].decoder.tokenizer
    other = models["decoder_balanced"].decoder.tokenizer
    if tokenizer.id2move != other.id2move or tokenizer.id2piece != other.id2piece:
        raise ValueError("decoder tokenizer mismatch")
    accumulator = ComparisonAccumulator()
    relative_accumulator = ComparisonAccumulator()
    exclusions = Counter()
    issue_counts = Counter()
    date_counts = Counter()
    early_state_hashes: set[bytes] = set()
    early_state_occurrences = 0
    accepted_games = 0
    expected_games = len(selected_game_ids) if selected_game_ids is not None else index["num_games"]
    progress_target = min(expected_games, args.max_games) if args.max_games is not None else expected_games
    processed_games = 0
    progress_started = time.monotonic()
    started = datetime.now(timezone.utc)
    per_move = args.output_dir / "per_move.csv"
    recovered_ids = set()
    resume_proof = None
    resume_discarded = []
    resume_segments = []
    if args.resume_csv:
        from analysis.ply_comparison.resume import verify_source
        resume_proof = [verify_source(csv_path, proof_path,
                                     {"pgn": args.pgn, "tokenizer": args.tokenizer, **CHECKPOINTS},
                                     code_review=args.resume_code_review)
                        for csv_path, proof_path in zip(args.resume_csv, args.resume_source)]
        print("All source CSVs, corpus, checkpoints, tokenizer and scoring code match; rebuilding completed-game statistics.", flush=True)
    with args.pgn.open("rb") as source, per_move.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PER_MOVE_FIELDS)
        writer.writeheader()
        if args.resume_csv:
            from analysis.ply_comparison.resume import recover
            for csv_path in args.resume_csv:
                print(f"Validating and recovering {csv_path}", flush=True)
                imported = recover(csv_path, pgn_path=args.pgn, index=index,
                                   fields=PER_MOVE_FIELDS, writer=writer, accumulator=accumulator,
                                   relative_accumulator=relative_accumulator, skip_game_ids=recovered_ids,
                                   progress=lambda count, gid: print(f"Recovery: validated {count} complete games through ID {gid}", flush=True))
                added = len(imported["game_ids"])
                resume_segments.append({"csv": str(csv_path.resolve()), "added_games": added,
                    "duplicate_games_skipped": len(imported["validated_game_ids"]) - added,
                    "discarded_final_games": imported["discarded_final_games"]})
                recovered_ids.update(imported["game_ids"])
                resume_discarded.extend(imported["discarded_final_games"])
                issue_counts.update(imported["issue_counts"])
                date_counts.update(imported["date_counts"])
                early_state_hashes.update(imported["early_hashes"])
                early_state_occurrences += imported["early_occurrences"]
            accepted_games = processed_games = len(recovered_ids)
            progress_started = time.monotonic()
            print(f"Recovered {len(recovered_ids)}/{expected_games} complete games; discarded partial final games {resume_discarded}.", flush=True)
        for game_row in index["games"]:
            game_id = int(game_row["idx"])
            if game_id in recovered_ids:
                continue
            if selected_game_ids is not None and game_id not in selected_game_ids:
                continue
            source.seek(int(game_row["start"]))
            raw = source.read(int(game_row["end"]) - int(game_row["start"]))
            game = chess.pgn.read_game(io.StringIO(raw.decode("utf-8", errors="replace")))
            if game is None:
                exclusions["parse_failure"] += 1
                processed_games += 1
                continue
            try:
                examples = game_examples(game, game_id, tokenizer)
            except (ValueError, KeyError) as exc:
                exclusions[str(exc).split(":", 1)[-1].strip()] += 1
                processed_games += 1
                continue
            issue_counts[examples[0].issue] += 1
            date_counts[examples[0].date] += 1
            for item in examples:
                if item.ply < 40:
                    state = " ".join(item.fen.split()[:4]).encode("ascii")
                    early_state_hashes.add(hashlib.blake2b(state, digest_size=16).digest())
                    early_state_occurrences += 1
            decoder_cache = cached_decoder_logits(examples, models, device)
            for start in range(0, len(examples), args.batch_size):
                evaluate_batch(examples[start:start + args.batch_size], models,
                               device, writer, accumulator, relative_accumulator,
                               decoder_cache)
            accepted_games += 1
            processed_games += 1
            if processed_games % 100 == 0 or accepted_games == progress_target:
                elapsed = time.monotonic() - progress_started
                rate = (processed_games - len(recovered_ids)) / elapsed if elapsed > 0 else 0.0
                remaining = max(0, progress_target - (accepted_games if args.max_games is not None else processed_games))
                eta = remaining / rate if rate > 0 else float("inf")
                eta_text = f"{eta / 60:.1f} min" if math.isfinite(eta) else "unknown"
                work_done = accepted_games if args.max_games is not None else processed_games
                percent = 100.0 * min(work_done, progress_target) / progress_target
                print(
                    f"Progress: {work_done}/{progress_target} games "
                    f"({percent:.2f}%), {processed_games} processed, "
                    f"{sum(exclusions.values())} excluded; ETA {eta_text}",
                    flush=True,
                )
            if args.max_games is not None and accepted_games >= args.max_games:
                break

    complete = args.max_games is None and accepted_games + sum(exclusions.values()) == expected_games
    if complete and accepted_games == 0:
        raise ValueError("no valid evaluation games")
    if complete:
        summary, deltas = accumulator.summarize(
            bootstrap_replicates=args.bootstrap, seed=args.seed,
        )
        summary_path = args.output_dir / "bucket_summary.csv"
        delta_path = args.output_dir / "paired_deltas.csv"
        write_summary_csv(summary_path, summary, SUMMARY_FIELDS)
        write_summary_csv(delta_path, deltas, DELTA_FIELDS)
        write_summary_csv(
            args.output_dir / "relative_summary.csv",
            relative_accumulator.summarize_without_intervals(RELATIVE_BUCKETS),
            SUMMARY_FIELDS,
        )
    manifest = {
        "format": "twic_disjoint_comparison_v1",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_disjointness": "TWIC issues 1662-1667 confirmed disjoint by project owner",
        "complete": complete,
        "full_corpus": complete and selected_game_ids is None,
        "corpus_games": index["num_games"],
        "selected_games": expected_games,
        "sampling": ({"method": "SHA-256 rank of full indexed PGN record within issue",
                      "games_per_issue": args.sample_games_per_issue}
                     if selected_game_ids is not None else None),
        "smoke_max_games": args.max_games,
        "accepted_games": accepted_games,
        "excluded_games": dict(exclusions),
        "games_by_issue": dict(sorted(issue_counts.items())),
        "pgn_date_min": min(date_counts) if date_counts else None,
        "pgn_date_max": max(date_counts) if date_counts else None,
        "early_state_occurrences": early_state_occurrences,
        "early_unique_states": len(early_state_hashes),
        "early_duplicate_fraction": (
            1 - len(early_state_hashes) / early_state_occurrences
            if early_state_occurrences else None
        ),
        "bootstrap_replicates": args.bootstrap if complete else None,
        "bootstrap_seed": args.seed if complete else None,
        "batch_size": args.batch_size,
        "device": str(device),
        "git_revision": _git_revision(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "chess_version": chess.__version__,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "pgn": args.pgn, "index": args.index,
                "builder_report": args.builder_report,
                "tokenizer": args.tokenizer, **CHECKPOINTS,
            }.items()
        },
        "metrics": {
            "primary": "mean exact legal UCI top-1 accuracy by absolute-ply band",
            "secondary": "top-5 with greedy first; legal UCI NLL and perplexity",
            "paired_delta": "balanced minus baseline; game-cluster percentile bootstrap",
        },
    }
    if resume_proof is not None:
        manifest["resume"] = {"source_csv": [str(p.resolve()) for p in args.resume_csv],
                              "source_provenance": resume_proof,
                              "segments": resume_segments,
                              "duplicate_policy": "First source owns the complete game for all four models; no duplicate decisions are counted.",
                              "recovered_games": len(recovered_ids),
                              "discarded_final_games": resume_discarded,
                              "new_inference_device": str(device),
                              "note": "Rows were evaluated on multiple computers; device/software metadata above describe new inference, with source metadata retained here."}
    audit_path = args.pgn.with_name("twic_1662_1667_corpus_audit.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("pgn_sha256") != manifest["files"]["pgn"]["sha256"]
                or audit.get("index_sha256") != manifest["files"]["index"]["sha256"]):
            raise ValueError("corpus audit hashes do not match the evaluation inputs")
        manifest["corpus_audit"] = {
            "path": str(audit_path.resolve()), "sha256": sha256(audit_path),
            "explicit_180_or_300_second_games": audit["explicit_180_or_300_second_games"],
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if complete:
        plot_main(["--summary", str(summary_path), "--deltas", str(delta_path),
                   "--output-dir", str(args.output_dir / "figures")])
    print(f"Done: {accepted_games} games; outputs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
