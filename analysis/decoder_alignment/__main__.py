"""Evaluate a post-trained decoder on the existing shared TWIC human games.

Run from the root: python -m analysis.decoder_alignment --resume
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import groupby
import io
import json
import math
from pathlib import Path
import platform
import time

import chess
import chess.pgn
import numpy as np
import torch

from analysis.ply_comparison.aggregate import bucket_for_ply, relative_bucket_for_ply
from analysis.ply_comparison.evaluate import PER_MOVE_FIELDS, game_examples, sha256
from dataset.offsets import load_pgn_offsets
from models.chess_tokenizer import ChessTokenizer
from .inference import check_baseline_predictions, load_model, predict_game, validate_positions
from .report import write_report
from .storage import BASE_FIELDS, Store, run_lock


CODE_ROOT = Path(__file__).resolve().parents[2]


def log(args, message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (args.output_dir / "evaluation.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stable_hash(path):
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return digest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-id", default="decoder_rloo_v5")
    parser.add_argument("--stage", choices=("all", "prepare", "evaluate", "summarize", "export"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for windows beyond 256 history tokens.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-games", type=int, help="Smoke test only; use a separate output directory.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.bootstrap < 1 or (args.max_games is not None and args.max_games < 1):
        parser.error("batch-size, bootstrap and max-games must be positive")
    if not args.candidate_id or args.candidate_id == "decoder_baseline":
        parser.error("candidate-id must differ from decoder_baseline")
    root = args.project_root.resolve()
    args.project_root = root
    args.baseline_dir = (args.baseline_dir or root / "outputs/twic_1662_1667").resolve()
    args.checkpoint = (args.checkpoint or root / "checkpoints/rloo_decoder_v12_lichess_mixed_v5_horizon8.pt").resolve()
    args.output_dir = (args.output_dir or root / "outputs/twic_1662_1667_decoder_rloo_v5_alignment").resolve()
    if args.output_dir == args.baseline_dir:
        parser.error("output-dir must differ from baseline-dir; saved baseline results are read-only")
    if args.max_games is not None and args.output_dir == root / "outputs/twic_1662_1667_decoder_rloo_v5_alignment":
        parser.error("a smoke test requires its own --output-dir")
    return args


def evaluation_inputs(args):
    """Pin the old corpus/tokenizer/reference identities and all new inputs."""
    root = args.project_root
    paths = {
        "pgn": root / "processed_games/eval/twic_1662_1667_classic.pgn",
        "index": root / "processed_games/eval/twic_1662_1667_classic_index.json",
        "tokenizer": root / "tokenizers/chess_uci_vocab.json",
        "decoder_baseline": root / "checkpoints/chess_llm_decoder_v12.pt",
        "candidate": args.checkpoint,
        "baseline_csv": args.baseline_dir / "per_move.csv",
        "baseline_manifest": args.baseline_dir / "manifest.json",
    }
    manifest = json.loads(paths["baseline_manifest"].read_text())
    if (manifest.get("format") != "twic_disjoint_comparison_v1" or not manifest.get("complete")
            or not manifest.get("full_corpus") or manifest.get("excluded_games")):
        raise ValueError("baseline must be a completed full-corpus TWIC comparison without excluded games")
    log(args, "Verifying corpus, tokenizer, checkpoints and saved baseline CSV...")
    hashes = {name: stable_hash(path) for name, path in paths.items()}
    for key in ("pgn", "index", "tokenizer", "decoder_baseline"):
        if hashes[key] != manifest["files"][key]["sha256"]:
            raise ValueError(f"saved baseline provenance differs from the requested {key}")
    index = load_pgn_offsets(str(paths["index"]))
    if index["num_games"] != manifest["accepted_games"] or index["num_games"] != manifest["corpus_games"]:
        raise ValueError("baseline manifest and corpus game counts differ")
    rows = index["games"][:args.max_games] if args.max_games else index["games"]
    selected = {int(row["idx"]): row for row in rows}
    if not selected or len(selected) != len(rows):
        raise ValueError("empty or duplicate corpus game IDs")
    code_paths = [*Path(__file__).parent.glob("*.py"),
                  CODE_ROOT / "analysis/ply_comparison/evaluate.py",
                  CODE_ROOT / "analysis/ply_comparison/aggregate.py",
                  CODE_ROOT / "analysis/ply_comparison/plot_results.py",
                  CODE_ROOT / "dataset/offsets.py",
                  *(CODE_ROOT / "models" / name for name in ("chess_decoder_player.py", "chess_llm.py", "chess_tokenizer.py"))]
    identity = {"format": "decoder_human_alignment_v1", "input_sha256": hashes,
                "candidate_id": args.candidate_id, "selected_game_ids": sorted(selected),
                "max_context": 256, "greedy": True, "likelihood_temperature": 1.0,
                "code_sha256": {str(path.relative_to(CODE_ROOT)): stable_hash(path) for path in sorted(code_paths)}}
    return paths, selected, identity


def progress(args, store, stage, **extra):
    counts = store.db.execute("SELECT COUNT(*), COALESCE(SUM(candidate_complete),0), COALESCE(SUM(CASE WHEN candidate_complete THEN n_positions ELSE 0 END),0) FROM games").fetchone()
    value = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "baseline_games": counts[0], "completed_games": counts[1], "completed_positions": counts[2],
             **extra}
    atomic_json(args.output_dir / "progress.json", value)


def baseline_row(row):
    converted = {key: row[key] for key in BASE_FIELDS}
    for key in ("ply", "game_plies", "twic_issue", "white_elo", "black_elo", "n_legal",
                "context_len", "context_truncated", "top1_correct", "top5_correct"):
        converted[key] = int(converted[key])
    converted["nll_legal"] = nll = float(converted["nll_legal"])
    p = converted["ply"]
    if (converted["bucket"] != bucket_for_ply(p)
            or converted["relative_bucket"] != relative_bucket_for_ply(p, converted["game_plies"])
            or converted["context_len"] != min(p + 1, 256)
            or converted["context_truncated"] != int(p >= 256)
            or converted["top1_correct"] not in (0, 1) or converted["top5_correct"] not in (0, 1)
            or converted["top5_correct"] < converted["top1_correct"]
            or converted["top1_correct"] != int(converted["predicted_uci"] == converted["target_uci"])
            or not math.isfinite(nll) or nll < 0
            or not math.isclose(float(row["p_target_legal"]), math.exp(-nll), abs_tol=1e-12, rel_tol=1e-6)):
        raise ValueError(f"invalid saved baseline metrics or context: game {row['game_id']}, ply {p}")
    return converted


def prepare(args, paths, selected, store):
    if store.get("baseline_import_complete"):
        if store.imported() != set(selected):
            raise ValueError("stored baseline game IDs differ from the frozen corpus")
        return
    before = paths["baseline_csv"].stat()
    imported = store.imported()
    seen = set()
    started = time.monotonic()
    progress(args, store, "importing_baseline", total_games=len(selected))
    with paths["baseline_csv"].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PER_MOVE_FIELDS):
            raise ValueError("baseline CSV columns differ from the original evaluator")
        for game_id, group in groupby(reader, key=lambda row: int(row["game_id"])):
            if game_id not in selected:
                continue
            if game_id in seen:
                raise ValueError(f"baseline game occurs in multiple CSV groups: {game_id}")
            seen.add(game_id)
            if game_id in imported:
                continue
            rows = [baseline_row(row) for row in group if row["model_id"] == "decoder_baseline"]
            store.save_baseline(game_id, rows)
            imported.add(game_id)
            if len(imported) % 1000 == 0:
                log(args, f"Baseline import: {len(imported):,}/{len(selected):,} games; {(time.monotonic()-started)/60:.1f} min")
                progress(args, store, "importing_baseline", total_games=len(selected))
            if args.max_games and imported == set(selected):
                break
    after = paths["baseline_csv"].stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("baseline CSV changed during import; use an immutable source")
    if imported != set(selected):
        raise ValueError(f"baseline CSV does not cover the selected corpus: {len(imported)}/{len(selected)} games")
    store.set("baseline_import_complete", True)
    log(args, f"Baseline ready: {len(imported):,} games; no baseline model inference rerun.")
    progress(args, store, "baseline_ready", total_games=len(selected))


def read_examples(source, offset, game_id, tokenizer):
    source.seek(int(offset["start"]))
    raw = source.read(int(offset["end"]) - int(offset["start"]))
    game = chess.pgn.read_game(io.StringIO(raw.decode("utf-8", errors="replace")))
    if game is None:
        raise ValueError(f"cannot parse corpus game {game_id}")
    return game_examples(game, game_id, tokenizer, include_board_grid=False)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def evaluate(args, paths, selected, store):
    device = choose_device(args.device)
    store.pin("inference_runtime", {"device": str(device), "python": platform.python_version(),
                                    "torch": torch.__version__, "numpy": np.__version__,
                                    "chess": chess.__version__, "batch_size": args.batch_size})
    tokenizer = ChessTokenizer(str(paths["tokenizer"]))
    with paths["pgn"].open("rb") as source:
        if not store.get("baseline_check"):
            # The full baseline is reused. Six short representative games
            # (one per issue on the full corpus) guard against changed inference.
            representatives = [row[0] for row in store.db.execute("SELECT MIN(game_id) FROM baseline GROUP BY twic_issue ORDER BY twic_issue")]
            baseline = load_model(paths["decoder_baseline"], paths["tokenizer"], device)
            for game_id in representatives:
                examples = read_examples(source, selected[game_id], game_id, tokenizer)
                saved = store.baseline_rows(game_id)
                validate_positions(examples, saved)
                actual = predict_game(examples, baseline, device, args.batch_size)
                check_baseline_predictions(saved, actual)
            del baseline
            store.set("baseline_check", {"status": "passed", "game_ids": representatives,
                                         "method": "one game per represented issue; exact greedy/top5, NLL tolerance 3e-4"})
            log(args, f"Saved baseline compatibility passed on {len(representatives)} games.")
        completed = store.completed()
        remaining = [game_id for game_id in selected if game_id not in completed]
        if not remaining:
            return
        model = load_model(paths["candidate"], paths["tokenizer"], device)
        log(args, f"Evaluating {args.candidate_id} on {device}; resuming at {len(completed):,}/{len(selected):,} games.")
        started = time.monotonic()
        progress(args, store, "evaluating", total_games=len(selected), device=str(device))
        for done, game_id in enumerate(remaining, 1):
            examples = read_examples(source, selected[game_id], game_id, tokenizer)
            validate_positions(examples, store.baseline_rows(game_id))
            predictions = predict_game(examples, model, device, args.batch_size)
            store.save_candidate(game_id, predictions)
            if done % 100 == 0 or done == len(remaining):
                rate = done / max(time.monotonic() - started, 1e-9)
                eta = (len(remaining) - done) / rate
                count = len(completed) + done
                log(args, f"Progress: {count:,}/{len(selected):,} games ({100*count/len(selected):.2f}%); ETA {eta/60:.1f} min")
                progress(args, store, "evaluating", total_games=len(selected), device=str(device), eta_seconds=eta)


def export_csv(args, store):
    temporary = args.output_dir / "per_move.csv.tmp"
    cursor = store.db.execute("""
        SELECT b.*, c.predicted_uci AS candidate_predicted_uci,
               c.top1_correct AS candidate_top1_correct, c.top5_correct AS candidate_top5_correct,
               c.nll_legal AS candidate_nll_legal
        FROM baseline b JOIN candidate c USING(game_id,ply) ORDER BY b.game_id,b.ply
    """)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([col[0] for col in cursor.description])
        writer.writerows(cursor)
    temporary.replace(args.output_dir / "per_move.csv")


def main(argv=None):
    args = parse_args(argv)
    if (args.output_dir.exists() and not (args.output_dir / "results.sqlite").exists()
            and any(p.name not in (".run.lock", "evaluation.log") for p in args.output_dir.iterdir())):
        raise FileExistsError(f"output directory contains unrelated files: {args.output_dir}")
    with run_lock(args.output_dir):
        paths, selected, identity = evaluation_inputs(args)
        store = Store(args.output_dir / "results.sqlite", identity, resume=args.resume)
        try:
            atomic_json(args.output_dir / "manifest.json", {"identity": identity,
                        "input_paths": {name: str(path) for name, path in paths.items()},
                        "full_corpus": args.max_games is None, "selected_games": len(selected),
                        "purpose": "complete legal UCI human-move agreement on recorded histories"})
            if args.stage in ("all", "prepare", "evaluate"):
                prepare(args, paths, selected, store)
            if args.stage in ("all", "evaluate"):
                evaluate(args, paths, selected, store)
                progress(args, store, "evaluated", total_games=len(selected))
            if args.stage in ("all", "summarize", "export"):
                store.validate_complete(set(selected))
                if args.stage == "export":
                    export_csv(args, store)
                else:
                    write_report(args.output_dir, store, replicates=args.bootstrap, seed=args.seed,
                                 candidate_id=args.candidate_id, full_corpus=args.max_games is None)
                progress(args, store, "complete", total_games=len(selected))
            log(args, f"Outputs: {args.output_dir}")
        except KeyboardInterrupt:
            progress(args, store, "interrupted", total_games=len(selected))
            log(args, "Interrupted safely. Repeat the same command with --resume.")
            raise SystemExit(130)
        except Exception as error:
            progress(args, store, "failed", error=str(error), total_games=len(selected))
            log(args, f"Failed: {error}")
            raise
        finally:
            store.close()


if __name__ == "__main__":
    main()
