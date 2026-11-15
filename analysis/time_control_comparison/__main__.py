"""CLI for prepare/evaluate/summarize/plot stages of the shared-game comparison."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import chess
import numpy as np
import torch

from .aggregate import summarize
from .common import CODE_ROOT, PROTOCOL_VERSION, atomic_json, code_hashes, load_config, sha256
from .corpus import game_identity, prepare, read_game
from .inference import evaluate_game, load_models
from .report import render
from .storage import ResultStore, run_lock


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def evaluation_identity(config, selection, device):
    return {
        "protocol": PROTOCOL_VERSION, "selection_signature": selection["signature"],
        "selected_pgn_sha256": selection["selected_pgn_sha256"],
        "selected_games_sha256": selection["games_sha256"],
        "checkpoints": {m: sha256(p) for m, p in config.checkpoints.items()},
        "code_sha256": code_hashes(), "batch_size": config.batch_size,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "chess": chess.__version__, "device": str(device),
                    "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                    "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }


def evaluate(config, selection, *, resume: bool, device, max_new_games=None):
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    identity = evaluation_identity(config, selection, device)
    store = ResultStore(config.output_dir / "results.sqlite", identity, resume=resume)
    try:
        done = store.completed_ids()
        expected = {g["game_id"] for g in selection["games"]}
        if not done <= expected:
            raise ValueError("database contains games outside the frozen sample")
        pending = [g for g in selection["games"] if g["game_id"] not in done]
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        manifest = {"identity": identity, "git_revision": revision, "complete": False,
                    "checkpoint_paths": {m: str(p) for m, p in config.checkpoints.items()},
                    "selected_games": len(expected), "completed_games": len(done),
                    "inference": {
                        "input": "perfect board piece IDs (rank 8 to 1), side, castling rights and en-passant square; no future moves",
                        "architecture": "ChessEncoderPlayerV2: dim 256, 8 layers, 4 attention heads, piece embedding 32, ep embedding 16",
                        "top1": "legal pair-first greedy; flattened-index pair ties; promotion classes 0..4 with deployed first-legal fallback",
                        "nll": "natural log; untempered legal-pair softmax times conditional legal-promotion softmax over classes 1..4",
                        "top5": "five highest joint complete-move probabilities, supplementary",
                        "precision": "float32 model inference, float64 CPU legal log-softmax",
                    },
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
        path = config.output_dir / "run_manifest.json"
        atomic_json(path, manifest)
        models = load_models(config.checkpoints, device) if pending else None
        started = time.monotonic()
        try:
            for i, row in enumerate(pending, 1):
                game = read_game(config.output_dir / "selected_games.pgn", row)
                if game_identity(game) != row["game_id"]:
                    raise ValueError("frozen game identity mismatch")
                positions, predictions = evaluate_game(game, models, device, config.batch_size)
                store.save_game(row, positions, predictions)
                manifest["completed_games"] = len(done) + i
                if i == 1 or i % 10 == 0 or i == len(pending):
                    elapsed = time.monotonic() - started
                    eta = (len(pending) - i) * elapsed / i
                    atomic_json(path, manifest)
                    print(f"Evaluation: {len(done)+i}/{len(expected)} games; {i/max(elapsed, 1e-9):.2f} games/s; ETA {eta/60:.1f} min", flush=True)
                if max_new_games and i >= max_new_games:
                    break
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["completed_games"] = len(store.completed_ids())
            atomic_json(path, manifest)
            raise
        if identity["checkpoints"] != {m: sha256(p) for m, p in config.checkpoints.items()}:
            manifest.update({"status": "failed_checkpoint_changed", "completed_games": len(store.completed_ids())})
            atomic_json(path, manifest)
            raise ValueError("checkpoint changed during evaluation; this run must not be combined with later weights")
        if sha256(config.output_dir / "selected_games.pgn") != identity["selected_pgn_sha256"]:
            manifest.update({"status": "failed_input_changed", "completed_games": len(store.completed_ids())})
            atomic_json(path, manifest)
            raise ValueError("frozen PGN changed during evaluation")
        complete = store.completed_ids() == expected
        manifest.update({"complete": complete, "status": "complete" if complete else "paused_after_game_limit",
                         "completed_games": len(store.completed_ids()), "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        if complete:
            store.validate_complete(selection["games"])
            store.export_csv(config.output_dir)
        atomic_json(path, manifest)
        return complete
    finally:
        store.close()


def summarize_saved(config):
    run = json.loads((config.output_dir / "run_manifest.json").read_text())
    selection = json.loads((config.output_dir / "selection_manifest.json").read_text())
    if not run["complete"]:
        raise ValueError("evaluation is incomplete; resume evaluate before reporting full-sample results")
    store = ResultStore(config.output_dir / "results.sqlite", run["identity"], resume=True)
    try:
        store.validate_complete(selection["games"])
        store.export_csv(config.output_dir)
        summarize(store, config)
    finally:
        store.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "configs/time_control_comparison_twic_1662_1667.yaml")
    parser.add_argument("--project-root", type=Path, help="Resolve configured artifact paths against this project checkout")
    parser.add_argument("--stage", choices=("prepare", "evaluate", "summarize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, help="Override output directory (relative to project root)")
    parser.add_argument("--games-per-category", type=int, help="Override quota for a separate smoke test")
    parser.add_argument("--allow-shortfall", action="store_true", help="Explicitly accept fewer games; never duplicates games to meet quota")
    parser.add_argument("--max-new-games", type=int, help="Pause evaluation after this many newly committed games")
    args = parser.parse_args(argv)
    for name in ("games_per_category", "max_new_games"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config, args.project_root)
    if args.output_dir:
        config = replace(config, output_dir=(config.project_root / args.output_dir).resolve())
    if args.games_per_category:
        config = replace(config, games_per_category=args.games_per_category)
    if args.allow_shortfall:
        config = replace(config, allow_shortfall=True)
    with run_lock(config.output_dir):
        if args.stage in ("prepare", "evaluate", "all"):
            selection = prepare(config, resume=args.resume)
        if args.stage in ("evaluate", "all"):
            complete = evaluate(config, selection, resume=args.resume, device=select_device(args.device), max_new_games=args.max_new_games)
            if not complete:
                print("Saved completed games. Repeat with --resume to finish before generating reports.", flush=True)
                return
        if args.stage in ("summarize", "all"):
            summarize_saved(config)
        if args.stage in ("plot", "summarize", "all"):
            render(config)
        print(f"{args.stage}: artifacts saved in {config.output_dir}", flush=True)


if __name__ == "__main__":
    main()
