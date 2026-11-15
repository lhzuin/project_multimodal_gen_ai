"""Run the frozen decoder-state probe and on-policy drift experiment.

The stages are deliberately restartable. Run ``python -m
analysis.state_probe.pipeline --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from itertools import chain

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from analysis.ply_comparison.aggregate import bucket_for_ply
from analysis.state_probe.core import (
    BloomIndex, CONDITIONS, DECODER_CHECKPOINTS, GENERATED_BUCKETS, MODEL_IDS,
    ROOT, StateProbe, atomic_json, board_target, build_splits, choose_device,
    config_hash, ensemble_metrics, extract_condition_batch, generated_bucket,
    history_tokens, load_config, load_decoders, normalized_position,
    position_digest, probe_loss, progress, read_csv, read_indexed_game,
    resolve_path, seed_everything, sha256, truncate_history, write_csv,
)
from dataset.offsets import load_pgn_offsets


SPLIT_FIELDS = ("game_id", "twic_issue", "split", "sha256", "selected")
METADATA_FIELDS = (
    "game_id", "twic_issue", "split", "ply", "game_plies", "bucket", "fen",
    "board", "turn", "castling", "ep",
)
METRIC_FIELDS = (
    "board_nll", "occupied_nll", "board_hamming", "occupancy_precision",
    "occupancy_recall", "occupancy_f1", "piece_accuracy_occupied",
    "square_accuracy_13", "exact_board", "turn_accuracy", "castling_exact",
    "ep_accuracy", "seed_board_nll_std",
)
RESULT_FIELDS = (
    "unit_id", "game_id", "twic_issue", "trajectory_source", "encoded_by",
    "condition", "ply", "generated_ply", "bucket", "generated_bucket",
    "fen", "common_reference_novel", "policy_top1", "policy_nll", *METRIC_FIELDS,
)


def output_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["output_dir"])


def split_path(config: dict) -> Path:
    return output_dir(config) / "splits.csv"


def ensure_splits(config: dict, resume: bool) -> list[dict[str, str]]:
    path = split_path(config)
    if resume and path.exists():
        return read_csv(path)
    rows = build_splits(config)
    write_csv(path, rows, SPLIT_FIELDS)
    selected = Counter((row["split"], row["twic_issue"]) for row in rows if row["selected"])
    atomic_json(output_dir(config) / "split_manifest.json", {
        "format": "state_probe_split_v1",
        "method": "SHA-256 modulo split, then independent SHA-256 rank within issue",
        "fractions": {"train": .70, "val": .15, "test": .15},
        "selected_counts": {f"{split}:{issue}": count for (split, issue), count in sorted(selected.items())},
        "pgn_sha256": sha256(resolve_path(config["paths"]["pgn"])),
        "index_sha256": sha256(resolve_path(config["paths"]["index"])),
        "config_sha256": config["_config_sha256"],
    })
    return [{key: str(value) for key, value in row.items()} for row in rows]


def stage_index_reference(config: dict, resume: bool) -> None:
    reference = resolve_path(config["paths"]["reference_pgn"])
    reference_index = resolve_path(config["paths"]["reference_index"])
    target = output_dir(config) / "novelty" / "common_reference.bloom"
    metadata_path = target.with_suffix(".json")
    settings = config["novelty"]
    source_sha256 = sha256(reference)
    if resume and target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("complete") and metadata.get("source_sha256") == source_sha256
                and metadata.get("capacity") == int(settings["capacity_positions"])
                and metadata.get("target_false_positive_rate") == float(settings["false_positive_rate"])):
            print("index-reference: matching complete index already exists", flush=True)
            return
    index = load_pgn_offsets(str(reference_index))
    start_game = 0
    if resume and target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("source_sha256") == source_sha256
                and metadata.get("capacity") == int(settings["capacity_positions"])
                and metadata.get("target_false_positive_rate") == float(settings["false_positive_rate"])):
            bloom = BloomIndex.load(target, metadata_path)
            start_game = int(metadata.get("games_scanned", 0))
        else:
            bloom = BloomIndex(int(settings["capacity_positions"]), float(settings["false_positive_rate"]))
    else:
        bloom = BloomIndex(int(settings["capacity_positions"]), float(settings["false_positive_rate"]))
    started = time.monotonic()
    checkpoint_every = int(settings.get("checkpoint_every_games", 50000))
    with reference.open("rb") as source:
        for game_position, row in enumerate(index["games"]):
            if game_position < start_game:
                continue
            game, _ = read_indexed_game(source, row)
            board = game.board()
            bloom.add_digest(position_digest(board))
            for move in game.mainline_moves():
                board.push(move)
                bloom.add_digest(position_digest(board))
            done = game_position + 1
            if done % checkpoint_every == 0 or done == len(index["games"]):
                bloom.save(target, metadata_path, {
                    "complete": done == len(index["games"]), "games_scanned": done,
                    "source_path": str(reference.resolve()), "source_sha256": source_sha256,
                    "source_index_sha256": sha256(reference_index),
                    "fingerprint": "BLAKE2b-128(piece placement, turn, clean castling, legal ep)",
                })
                progress("Reference index", done, len(index["games"]), started)


def _selected_rows(config: dict, split_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["game_id"]): row for row in split_rows if int(row["selected"])}


def _metadata_for_game(game, game_id: int, split_row: dict, tokenizer):
    states = list(history_tokens(game, tokenizer))
    # Probe positions match positions before recorded moves; omit the state after
    # the final move so joins to per_move.csv are one-to-one.
    states = states[:-1]
    total = len(states)
    metadata = []
    histories = []
    for ply, board, move_ids, piece_ids in states:
        target = board_target(board)
        metadata.append({
            "game_id": game_id, "twic_issue": int(game.headers["TWICIssue"]),
            "split": split_row["split"], "ply": ply, "game_plies": total,
            "bucket": bucket_for_ply(ply), "fen": board.fen(en_passant="legal"),
            "board": " ".join(map(str, target["board"])), "turn": target["turn"],
            "castling": " ".join(map(str, target["castling"])), "ep": target["ep"],
        })
        histories.append((move_ids, piece_ids))
    return metadata, histories


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def stage_extract_teacher(config: dict, resume: bool, device: torch.device) -> None:
    split_rows = ensure_splits(config, resume=True)
    selected = _selected_rows(config, split_rows)
    index = load_pgn_offsets(str(resolve_path(config["paths"]["index"])))
    pgn = resolve_path(config["paths"]["pgn"])
    models = load_decoders(config, device)
    tokenizer = models[MODEL_IDS[0]].decoder.tokenizer
    activation_dir = output_dir(config) / "activations"
    signature = config_hash({
        "config": config["_config_sha256"], "pgn": sha256(pgn),
        "checkpoints": {key: sha256(resolve_path(config["models"][key])) for key in MODEL_IDS},
    })
    started = time.monotonic()
    done = 0
    with pgn.open("rb") as source:
        for row in index["games"]:
            game_id = int(row["idx"])
            if game_id not in selected:
                continue
            split_row = selected[game_id]
            metadata_path = activation_dir / f"meta_{split_row['split']}_{game_id}_{signature}.csv"
            model_paths = {model_id: activation_dir / f"{model_id}_{split_row['split']}_{game_id}_{signature}.npz"
                           for model_id in MODEL_IDS}
            if resume and metadata_path.exists() and all(path.exists() for path in model_paths.values()):
                done += 1
                continue
            game, _ = read_indexed_game(source, row)
            metadata, histories = _metadata_for_game(game, game_id, split_row, tokenizer)
            initial_turn = 0 if game.board().turn == chess.WHITE else 1
            arrays = {}
            for model_id, model in models.items():
                arrays[model_id] = extract_condition_batch(
                    model, histories, initial_turn, device,
                    batch_size=int(config["extraction"]["batch_size"]),
                )
            write_csv(metadata_path, metadata, METADATA_FIELDS)
            for model_id, path in model_paths.items():
                _atomic_npz(path, **arrays[model_id])
            done += 1
            if done % 10 == 0 or done == len(selected):
                progress("Teacher extraction", done, len(selected), started)
    atomic_json(output_dir(config) / "extract_teacher_manifest.json", {
        "complete": done == len(selected), "games": done, "signature": signature,
        "conditions": list(CONDITIONS), "dtype": "float16",
    })


def _load_activation_split(config: dict, model_id: str, split: str, condition: str):
    directory = output_dir(config) / "activations"
    manifest_path = output_dir(config) / "extract_teacher_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("teacher extraction manifest missing")
    signature = json.loads(manifest_path.read_text(encoding="utf-8"))["signature"]
    metadata_files = sorted(directory.glob(f"meta_{split}_*_{signature}.csv"))
    activations, boards, turns, castling, eps, buckets, units, rows_out = [], [], [], [], [], [], [], []
    for metadata_path in metadata_files:
        parts = metadata_path.stem.split("_")
        game_id, signature = int(parts[-2]), parts[-1]
        npz_path = directory / f"{model_id}_{split}_{game_id}_{signature}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"missing activation shard: {npz_path}")
        metadata = read_csv(metadata_path)
        with np.load(npz_path) as payload:
            value = payload[condition].astype(np.float32)
        if len(value) != len(metadata):
            raise ValueError(f"activation/metadata length mismatch: {npz_path}")
        activations.append(value)
        boards.extend(np.asarray(row["board"].split(), dtype=np.int64) for row in metadata)
        turns.extend(int(row["turn"]) for row in metadata)
        castling.extend(np.asarray(row["castling"].split(), dtype=np.int64) for row in metadata)
        eps.extend(int(row["ep"]) for row in metadata)
        buckets.extend(row["bucket"] for row in metadata)
        units.extend(int(row["game_id"]) for row in metadata)
        rows_out.extend(metadata)
    if not activations:
        raise ValueError(f"no activation shards for {model_id}/{split}/{condition}")
    return {
        "x": np.concatenate(activations), "board": np.stack(boards),
        "turn": np.asarray(turns), "castling": np.stack(castling), "ep": np.asarray(eps),
        "bucket": np.asarray(buckets), "unit": np.asarray(units), "metadata": rows_out,
    }


def _tensor_data(data: dict, mean: np.ndarray, std: np.ndarray):
    return tuple(torch.from_numpy(value) for value in (
        ((data["x"] - mean) / std).astype(np.float32), data["board"].astype(np.int64),
        data["turn"].astype(np.int64), data["castling"].astype(np.int64), data["ep"].astype(np.int64),
    ))


@torch.no_grad()
def _mean_loss(probe, loader, device):
    total = 0.0
    count = 0
    probe.eval()
    for batch in loader:
        x, board, turn, castling, ep = (item.to(device) for item in batch)
        loss = probe_loss(probe(x), board, turn, castling, ep)
        total += float(loss) * len(x)
        count += len(x)
    return total / count


def _train_one(config, model_id, condition, architecture, seed, train, val, device, destination,
               experiment_signature):
    seed_everything(seed)
    mean = train["x"].mean(0).astype(np.float32)
    std = train["x"].std(0).astype(np.float32)
    std[std < 1e-6] = 1
    train_tensors = _tensor_data(train, mean, std)
    val_tensors = _tensor_data(val, mean, std)
    counts = Counter(train["bucket"])
    weights = torch.tensor([1 / counts[bucket] for bucket in train["bucket"]], dtype=torch.double)
    settings = config["probe"]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=int(settings["batch_size"]), sampler=sampler)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=int(settings["batch_size"]), shuffle=False)
    hidden_dim = 512 if architecture == "mlp" else None
    probe = StateProbe(hidden_dim=hidden_dim).to(device)
    lr = float(settings["mlp_lr"] if architecture == "mlp" else settings["linear_lr"])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=float(settings["weight_decay"]))
    best = math.inf
    best_state = None
    stale = 0
    history = []
    maximum = int(settings["max_epochs"])
    training_started = time.monotonic()
    for epoch in range(1, maximum + 1):
        probe.train()
        running = 0.0
        observations = 0
        for batch in train_loader:
            x, board, turn, castling, ep = (item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = probe_loss(probe(x), board, turn, castling, ep)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(x)
            observations += len(x)
        val_loss = _mean_loss(probe, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / observations, "val_loss": val_loss})
        progress(
            f"Probe {model_id}/{condition}/{architecture}/seed{seed}, val={val_loss:.5f}",
            epoch, maximum, training_started,
        )
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "format": "state_probe_v1", "model_id": model_id, "condition": condition,
        "architecture": architecture, "seed": seed, "mean": mean, "std": std,
        "state_dict": best_state, "best_val_loss": best, "history": history,
        "decoder_frozen": True,
        "experiment_signature": experiment_signature,
    }, temporary)
    os.replace(temporary, destination)


def stage_train_probes(config: dict, resume: bool, device: torch.device) -> None:
    probe_dir = output_dir(config) / "probes"
    seeds = [int(seed) for seed in config["probe"]["seeds"]]
    baselines = {}
    extraction_signature = json.loads(
        (output_dir(config) / "extract_teacher_manifest.json").read_text(encoding="utf-8")
    )["signature"]
    for model_id in MODEL_IDS:
        for condition in CONDITIONS:
            train = _load_activation_split(config, model_id, "train", condition)
            val = _load_activation_split(config, model_id, "val", condition)
            architectures = ("linear", "mlp") if condition == "final_w256" else ("linear",)
            if condition == "final_w256":
                all_boards = train["board"]
                majority = []
                for square in range(64):
                    majority.append(int(np.bincount(all_boards[:, square], minlength=13).argmax()))
                test_boards = _load_activation_split(config, model_id, "test", condition)["board"]
                baselines[model_id] = {
                    "evaluated_split": "selected held-out test games",
                    "empty_board_hamming": float((test_boards != 0).mean()),
                    "per_square_majority_hamming": float((test_boards != np.asarray(majority)).mean()),
                }
            for architecture in architectures:
                for seed in seeds:
                    destination = probe_dir / model_id / condition / f"{architecture}_seed{seed}.pt"
                    signature = config_hash({
                        "extraction": extraction_signature, "probe": config["probe"],
                        "model": model_id, "condition": condition,
                        "architecture": architecture, "seed": seed,
                    })
                    if resume and destination.exists():
                        existing = torch.load(destination, map_location="cpu", weights_only=False)
                        if existing.get("experiment_signature") == signature:
                            continue
                    _train_one(config, model_id, condition, architecture, seed,
                               train, val, device, destination, signature)
    atomic_json(output_dir(config) / "probe_baselines.json", baselines)
    atomic_json(output_dir(config) / "train_probes_manifest.json", {
        "complete": True, "seeds": seeds, "conditions": list(CONDITIONS),
        "linear_all_conditions": True, "mlp_final_w256_only": True,
        "signature": config_hash({"extraction": extraction_signature, "probe": config["probe"]}),
    })


def _encode_uci_history(board: chess.Board, moves: list[chess.Move], tokenizer):
    initial_turn = 0 if board.turn == chess.WHITE else 1
    move_ids = [tokenizer.bos_id]
    piece_ids = [tokenizer.p_bos_id]
    replay = board.copy(stack=False)
    for move in moves:
        if move not in replay.legal_moves:
            raise ValueError(f"illegal replay move {move.uci()} in {replay.fen()}")
        piece = replay.piece_at(move.from_square)
        move_ids.append(tokenizer.move2id[move.uci()])
        piece_ids.append(tokenizer.piece2id[piece.symbol().upper()])
        replay.push(move)
    return move_ids, piece_ids, initial_turn, replay


@torch.inference_mode()
def _greedy_move(model, board, move_ids, piece_ids, initial_turn, device):
    ids, pieces, start_turn = truncate_history(move_ids, piece_ids, initial_turn, 256)
    return model.sample_moves(
        torch.tensor([ids], dtype=torch.long, device=device), [board.fen()],
        piece_input_ids=torch.tensor([pieces], dtype=torch.long, device=device),
        start_turn=torch.tensor([start_turn], dtype=torch.long, device=device), greedy=True,
    )[0]


def stage_generate_selfplay(config: dict, resume: bool, device: torch.device) -> None:
    destination = output_dir(config) / "selfplay_trajectories.jsonl"
    rollout_signature = config_hash({
        "pgn": sha256(resolve_path(config["paths"]["pgn"])),
        "selfplay": config["selfplay"],
        "selected_split": sha256(split_path(config)) if split_path(config).exists() else None,
        "models": {model: sha256(resolve_path(config["models"][model])) for model in MODEL_IDS},
    })
    if resume and destination.exists():
        manifest_path = output_dir(config) / "selfplay_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("complete") and manifest.get("signature") == rollout_signature:
            print("generate-selfplay: complete trajectories already exist", flush=True)
            return
    split_rows = ensure_splits(config, resume=True)
    selected_test = {int(row["game_id"]): row for row in split_rows
                     if row["split"] == "test" and int(row["selected"])}
    index = load_pgn_offsets(str(resolve_path(config["paths"]["index"])))
    pgn = resolve_path(config["paths"]["pgn"])
    models = load_decoders(config, device)
    tokenizer = models[MODEL_IDS[0]].decoder.tokenizer
    prefix_plies = int(config["selfplay"]["prefix_plies"])
    max_generated = int(config["selfplay"]["max_generated_plies"])
    candidates = []
    seen = set()
    with pgn.open("rb") as source:
        for row in index["games"]:
            game_id = int(row["idx"])
            if game_id not in selected_test:
                continue
            game, _ = read_indexed_game(source, row)
            all_moves = list(game.mainline_moves())
            if len(all_moves) <= prefix_plies:
                continue
            board = game.board()
            for move in all_moves[:prefix_plies]:
                board.push(move)
            key = normalized_position(board) + "|" + " ".join(move.uci() for move in all_moves[:prefix_plies])
            if key in seen:
                continue
            seen.add(key)
            candidates.append((game_id, game, all_moves))
    started = time.monotonic()
    temporary = destination.with_suffix(".jsonl.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for game_id, game, human_moves in candidates:
            prefix = human_moves[:prefix_plies]
            for source_model, model in models.items():
                initial_board = game.board()
                move_ids, piece_ids, initial_turn, board = _encode_uci_history(initial_board, prefix, tokenizer)
                generated = []
                fens = []
                for _ in range(max_generated):
                    if board.is_game_over(claim_draw=True):
                        break
                    fens.append(board.fen(en_passant="legal"))
                    move = _greedy_move(model, board, move_ids, piece_ids, initial_turn, device)
                    piece = board.piece_at(move.from_square)
                    generated.append(move.uci())
                    move_ids.append(tokenizer.move2id[move.uci()])
                    piece_ids.append(tokenizer.piece2id[piece.symbol().upper()])
                    board.push(move)
                handle.write(json.dumps({
                    "opening_id": game_id, "twic_issue": int(game.headers["TWICIssue"]),
                    "trajectory_source": source_model, "initial_fen": initial_board.fen(),
                    "prefix_moves": [move.uci() for move in prefix],
                    "human_continuation": [move.uci() for move in human_moves[prefix_plies:]],
                    "generated_moves": generated, "fens_before_generated_moves": fens,
                    "terminated": board.is_game_over(claim_draw=True),
                }, separators=(",", ":")) + "\n")
                written += 1
                if written % 10 == 0 or written == len(candidates) * len(models):
                    progress("Self-play", written, len(candidates) * len(models), started)
    os.replace(temporary, destination)
    atomic_json(output_dir(config) / "selfplay_manifest.json", {
        "complete": True, "openings": len(candidates), "trajectories": written,
        "prefix_plies": prefix_plies, "max_generated_plies": max_generated,
        "policy": "deterministic greedy legal decoder self-play; same checkpoint controls both colors",
        "signature": rollout_signature,
    })


def _load_probe_ensemble(config, model_id, condition, architecture, device):
    probes = []
    mean = std = None
    for seed in config["probe"]["seeds"]:
        path = output_dir(config) / "probes" / model_id / condition / f"{architecture}_seed{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if mean is None:
            mean, std = checkpoint["mean"], checkpoint["std"]
        elif not (np.array_equal(mean, checkpoint["mean"]) and np.array_equal(std, checkpoint["std"])):
            raise ValueError(f"standardization mismatch in ensemble: {path}")
        probe = StateProbe(hidden_dim=512 if architecture == "mlp" else None)
        probe.load_state_dict(checkpoint["state_dict"], strict=True)
        probe.eval().to(device)
        probe.requires_grad_(False)
        probes.append(probe)
    return probes, mean, std


def _novelty_index(config):
    path = output_dir(config) / "novelty" / "common_reference.bloom"
    metadata = path.with_suffix(".json")
    if not path.exists() or not metadata.exists():
        raise FileNotFoundError("common-reference index missing; run --stage index-reference")
    return BloomIndex.load(path, metadata)


def _policy_lookup(config, wanted_games: set[int]):
    comparison = resolve_path(config["paths"]["comparison_output"])
    manifest_path = comparison / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("comparison manifest is missing; let the full comparison finish first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or not manifest.get("full_corpus"):
        raise ValueError("comparison manifest must report complete=true and full_corpus=true")
    result = {}
    with (comparison / "per_move.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            game_id = int(row["game_id"])
            if game_id in wanted_games and row["model_id"] in MODEL_IDS:
                result[(game_id, int(row["ply"]), row["model_id"])] = (
                    int(row["top1_correct"]), float(row["nll_legal"]),
                )
    return result


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


@torch.inference_mode()
def _score_activation_rows(config, model_id, condition, architecture, data, bloom, policy, device,
                           *, result_condition=None, loaded_ensemble=None):
    probes, mean, std = (loaded_ensemble or
                         _load_probe_ensemble(config, model_id, condition, architecture, device))
    rows = []
    batch_size = int(config["probe"]["batch_size"])
    for offset in range(0, len(data["x"]), batch_size):
        stop = min(offset + batch_size, len(data["x"]))
        x = torch.from_numpy(((data["x"][offset:stop] - mean) / std).astype(np.float32)).to(device)
        board = torch.from_numpy(data["board"][offset:stop]).long().to(device)
        turn = torch.from_numpy(data["turn"][offset:stop]).long().to(device)
        castling = torch.from_numpy(data["castling"][offset:stop]).long().to(device)
        ep = torch.from_numpy(data["ep"][offset:stop]).long().to(device)
        metrics = ensemble_metrics([probe(x) for probe in probes], board, turn, castling, ep)
        for local, index in enumerate(range(offset, stop)):
            metadata = data["metadata"][index]
            game_id, ply = int(metadata["game_id"]), int(metadata["ply"])
            policy_value = policy.get((game_id, ply, model_id), ("", ""))
            result = {
                "unit_id": game_id, "game_id": game_id, "twic_issue": metadata["twic_issue"],
                "trajectory_source": "human", "encoded_by": model_id,
                "condition": result_condition or condition, "ply": ply, "generated_ply": "",
                "bucket": metadata["bucket"], "generated_bucket": "", "fen": metadata["fen"],
                "common_reference_novel": int(not bloom.contains_digest(
                    hashlib.blake2b(normalized_position(chess.Board(metadata["fen"])).encode("ascii"), digest_size=16).digest()
                )), "policy_top1": policy_value[0], "policy_nll": policy_value[1],
            }
            result.update({key: float(metrics[key][local]) for key in METRIC_FIELDS})
            rows.append(result)
    return rows


def _stream_full_teacher(config, split_rows, bloom, policy, device, destination):
    """Evaluate the primary final-layer probe on every game in the 15% test pool."""
    test_games = {int(row["game_id"]): row for row in split_rows if row["split"] == "test"}
    index = load_pgn_offsets(str(resolve_path(config["paths"]["index"])))
    pgn = resolve_path(config["paths"]["pgn"])
    models = load_decoders(config, device)
    tokenizer = models[MODEL_IDS[0]].decoder.tokenizer
    ensembles = {model_id: _load_probe_ensemble(
        config, model_id, "final_w256", "linear", device
    ) for model_id in MODEL_IDS}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    done = 0
    with pgn.open("rb") as source, temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for index_row in index["games"]:
            game_id = int(index_row["idx"])
            if game_id not in test_games:
                continue
            game, _ = read_indexed_game(source, index_row)
            metadata, histories = _metadata_for_game(game, game_id, test_games[game_id], tokenizer)
            initial_turn = 0 if game.board().turn == chess.WHITE else 1
            targets = {
                "board": np.stack([np.asarray(row["board"].split(), dtype=np.int64) for row in metadata]),
                "turn": np.asarray([int(row["turn"]) for row in metadata]),
                "castling": np.stack([np.asarray(row["castling"].split(), dtype=np.int64) for row in metadata]),
                "ep": np.asarray([int(row["ep"]) for row in metadata]),
                "bucket": np.asarray([row["bucket"] for row in metadata]),
                "unit": np.full(len(metadata), game_id), "metadata": metadata,
            }
            for model_id, model in models.items():
                targets["x"] = extract_condition_batch(
                    model, histories, initial_turn, device,
                    batch_size=int(config["extraction"]["batch_size"]),
                    requested_conditions=("final_w256",),
                )["final_w256"].astype(np.float32)
                writer.writerows(_score_activation_rows(
                    config, model_id, "final_w256", "linear", targets,
                    bloom, policy, device, loaded_ensemble=ensembles[model_id],
                ))
            done += 1
            if done % 25 == 0 or done == len(test_games):
                progress("Full test evaluation", done, len(test_games), started)
    os.replace(temporary, destination)
    return done


@torch.inference_mode()
def _evaluate_selfplay(config, models, bloom, device):
    trajectory_path = output_dir(config) / "selfplay_trajectories.jsonl"
    tokenizer = models[MODEL_IDS[0]].decoder.tokenizer
    ensembles = {model_id: _load_probe_ensemble(config, model_id, "final_w256", "linear", device)
                 for model_id in MODEL_IDS}
    rows = []
    records = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line]
    started = time.monotonic()
    for record_index, record in enumerate(records, start=1):
        initial = chess.Board(record["initial_fen"])
        prefix = [chess.Move.from_uci(uci) for uci in record["prefix_moves"]]
        generated = [chess.Move.from_uci(uci) for uci in record["generated_moves"]]
        histories, boards = [], []
        for generated_ply in range(len(generated)):
            previous = prefix + generated[:generated_ply]
            ids, pieces, _, board = _encode_uci_history(initial, previous, tokenizer)
            recorded_fen = record["fens_before_generated_moves"][generated_ply]
            if board.fen(en_passant="legal") != recorded_fen:
                raise ValueError(
                    f"cross-replay FEN mismatch for opening {record['opening_id']} "
                    f"at generated ply {generated_ply}"
                )
            histories.append((ids, pieces))
            boards.append(board)
        if not histories:
            continue
        initial_turn = 0 if initial.turn == chess.WHITE else 1
        for encoded_by, model in models.items():
            activations = extract_condition_batch(
                model, histories, initial_turn, device,
                batch_size=int(config["extraction"]["batch_size"]),
                requested_conditions=("final_w256",),
            )["final_w256"].astype(np.float32)
            probes, mean, std = ensembles[encoded_by]
            for offset in range(0, len(activations), int(config["probe"]["batch_size"])):
                stop = min(offset + int(config["probe"]["batch_size"]), len(activations))
                x = torch.from_numpy(((activations[offset:stop] - mean) / std).astype(np.float32)).to(device)
                targets = [board_target(board) for board in boards[offset:stop]]
                board_tensor = torch.from_numpy(np.stack([target["board"] for target in targets])).long().to(device)
                turn = torch.tensor([target["turn"] for target in targets], device=device)
                castling = torch.from_numpy(np.stack([target["castling"] for target in targets])).long().to(device)
                ep = torch.tensor([target["ep"] for target in targets], device=device)
                metrics = ensemble_metrics([probe(x) for probe in probes], board_tensor, turn, castling, ep)
                for local, generated_ply in enumerate(range(offset, stop)):
                    board = boards[generated_ply]
                    absolute_ply = len(prefix) + generated_ply
                    result = {
                        "unit_id": record["opening_id"], "game_id": record["opening_id"],
                        "twic_issue": record["twic_issue"],
                        "trajectory_source": record["trajectory_source"], "encoded_by": encoded_by,
                        "condition": "final_w256", "ply": absolute_ply,
                        "generated_ply": generated_ply, "bucket": bucket_for_ply(absolute_ply),
                        "generated_bucket": generated_bucket(generated_ply),
                        "fen": board.fen(en_passant="legal"),
                        "common_reference_novel": int(not bloom.contains_digest(position_digest(board))),
                        "policy_top1": "", "policy_nll": "",
                    }
                    result.update({key: float(metrics[key][local]) for key in METRIC_FIELDS})
                    rows.append(result)
        if record_index % 10 == 0 or record_index == len(records):
            progress("Self-play evaluation", record_index, len(records), started)
    return rows


def _mean_summary(rows):
    groups = {}
    for row in rows:
        band = row["generated_bucket"] or row["bucket"]
        key = (row["trajectory_source"], row["encoded_by"], row["condition"], band)
        if key not in groups:
            groups[key] = {
                "n": 0, "units": set(), "novel": 0.0,
                "sums": {metric: 0.0 for metric in METRIC_FIELDS},
            }
        state = groups[key]
        state["n"] += 1
        state["units"].add(str(row["unit_id"]))
        state["novel"] += int(row["common_reference_novel"])
        for metric in METRIC_FIELDS:
            state["sums"][metric] += float(row[metric])
    result = []
    for key, state in sorted(groups.items()):
        item = {
            "trajectory_source": key[0], "encoded_by": key[1], "condition": key[2],
            "bucket": key[3], "n_units": len(state["units"]),
            "n_positions": state["n"], "novel_fraction": state["novel"] / state["n"],
        }
        for metric in METRIC_FIELDS:
            item[metric] = state["sums"][metric] / state["n"]
        result.append(item)
    return result


def _paired_bootstrap(rows, config):
    """Checkpoint deltas on identical human histories, clustered by game."""
    unit_values = defaultdict(lambda: defaultdict(
        lambda: {model: [0.0, 0] for model in MODEL_IDS}
    ))
    for row in rows:
        if row["trajectory_source"] == "human":
            for metric in ("board_nll", "occupied_nll", "board_hamming"):
                pair = unit_values[(row["condition"], row["bucket"], metric)][int(row["game_id"])][row["encoded_by"]]
                pair[0] += float(row[metric])
                pair[1] += 1
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    output = []
    for (condition, bucket, metric), games in sorted(unit_values.items()):
        units = []
        for values in games.values():
            baseline, balanced = values["decoder_baseline"], values["decoder_balanced"]
            if baseline[1] and baseline[1] == balanced[1]:
                units.append((balanced[0] - baseline[0], baseline[1]))
        units = np.asarray(units, dtype=np.float64)
        indices = rng.integers(0, len(units), size=(replicates, len(units)))
        sampled = units[indices].sum(1)
        draws = sampled[:, 0] / sampled[:, 1]
        low, high = np.percentile(draws, [2.5, 97.5])
        output.append({
            "condition": condition, "bucket": bucket, "metric": metric,
            "n_games": len(units), "balanced_minus_baseline": float(units[:, 0].sum() / units[:, 1].sum()),
            "ci_low": float(low), "ci_high": float(high),
        })
    return output


def _cross_replay_bootstrap(rows, config):
    units = defaultdict(lambda: defaultdict(lambda: {model: [0.0, 0] for model in MODEL_IDS}))
    for row in rows:
        if row["trajectory_source"] in MODEL_IDS and row["condition"] == "final_w256":
            for metric in ("board_nll", "occupied_nll", "board_hamming"):
                value = units[(row["trajectory_source"], row["generated_bucket"], metric)][int(row["unit_id"])][row["encoded_by"]]
                value[0] += float(row[metric])
                value[1] += 1
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    output = []
    for (source, bucket, metric), by_opening in sorted(units.items()):
        complete = [values for values in by_opening.values()
                    if all(values[model][1] for model in MODEL_IDS)]
        if not complete:
            continue
        array = np.asarray([[[values[model][0], values[model][1]] for model in MODEL_IDS]
                            for values in complete], dtype=np.float64)
        totals = array.sum(0)
        point = totals[1, 0] / totals[1, 1] - totals[0, 0] / totals[0, 1]
        draws = np.empty(replicates)
        for start in range(0, replicates, 64):
            stop = min(start + 64, replicates)
            sampled = array[rng.integers(0, len(array), size=(stop - start, len(array)))].sum(1)
            draws[start:stop] = (sampled[:, 1, 0] / sampled[:, 1, 1]
                                 - sampled[:, 0, 0] / sampled[:, 0, 1])
        low, high = np.percentile(draws, [2.5, 97.5])
        output.append({
            "trajectory_source": source, "bucket": bucket, "metric": metric,
            "n_openings": len(complete), "baseline_encoder": totals[0, 0] / totals[0, 1],
            "balanced_encoder": totals[1, 0] / totals[1, 1],
            "balanced_minus_baseline": point, "ci_low": float(low), "ci_high": float(high),
        })
    return output


def _drift_bootstrap(rows, config):
    # unit -> model/encoder -> early/late -> [sum, count]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"early": [0.0, 0], "late": [0.0, 0]}
    )))
    metrics = ("board_nll", "occupied_nll", "board_hamming")
    for row in rows:
        source = row["trajectory_source"]
        if source == "human":
            phase = "early" if row["bucket"] == "0_20" else ("late" if row["bucket"] == "100_plus" else None)
            group_prefix = (source, row["condition"])
        else:
            phase = "early" if row["generated_bucket"] == "0_10" else ("late" if row["generated_bucket"] == "80_120" else None)
            group_prefix = (source, "final_w256")
        if phase is None:
            continue
        for metric in metrics:
            value = groups[(*group_prefix, metric)][int(row["unit_id"])][row["encoded_by"]][phase]
            value[0] += float(row[metric])
            value[1] += 1
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    output = []
    for (source, condition, metric), units in sorted(groups.items()):
        complete = [values for values in units.values()
                    if all(values[model][phase][1] for model in MODEL_IDS for phase in ("early", "late"))]
        if not complete:
            continue
        array = np.asarray([[[item[model][phase] for phase in ("early", "late")]
                             for model in MODEL_IDS] for item in complete], dtype=np.float64)
        totals = array.sum(0)
        point_values = totals[:, 1, 0] / totals[:, 1, 1] - totals[:, 0, 0] / totals[:, 0, 1]
        sampled_changes = np.empty((replicates, len(MODEL_IDS)))
        for start in range(0, replicates, 64):
            stop = min(start + 64, replicates)
            sampled = array[rng.integers(0, len(array), size=(stop - start, len(array)))].sum(1)
            sampled_changes[start:stop] = (sampled[:, :, 1, 0] / sampled[:, :, 1, 1]
                                             - sampled[:, :, 0, 0] / sampled[:, :, 0, 1])
        early_bucket, late_bucket = (("0_20", "100_plus") if source == "human" else ("0_10", "80_120"))
        for model_index, model in enumerate(MODEL_IDS):
            draws = sampled_changes[:, model_index]
            low, high = np.percentile(draws, [2.5, 97.5])
            output.append({
                "trajectory_source": source, "encoded_by": model, "condition": condition,
                "metric": metric, "n_units": len(complete), "early_bucket": early_bucket,
                "late_bucket": late_bucket, "late_minus_early": point_values[model_index],
                "difference_in_differences": "", "ci_low": float(low), "ci_high": float(high),
            })
        did = point_values[1] - point_values[0]
        did_draws = sampled_changes[:, 1] - sampled_changes[:, 0]
        low, high = np.percentile(did_draws, [2.5, 97.5])
        output.append({
            "trajectory_source": source, "encoded_by": "balanced_minus_baseline",
            "condition": condition, "metric": metric, "n_units": len(complete),
            "early_bucket": early_bucket, "late_bucket": late_bucket,
            "late_minus_early": "", "difference_in_differences": did,
            "ci_low": float(low), "ci_high": float(high),
        })
    return output


def _trajectory_gap_bootstrap(rows, config):
    # Pair absolute plies within each opening before aggregating. Otherwise the
    # human opening bucket includes the easy prefix absent from self-play, and
    # different termination lengths change the position mix within later bands.
    positions = defaultdict(lambda: defaultdict(dict))
    metrics = ("board_nll", "occupied_nll", "board_hamming")
    for row in rows:
        if row["condition"] != "final_w256":
            continue
        source = row["trajectory_source"]
        key = (int(row["unit_id"]), row["encoded_by"])
        positions[key][source][int(row["ply"])]=row
    units = defaultdict(lambda: defaultdict(lambda: {"human": [0.0, 0], "selfplay": [0.0, 0]}))
    for (opening, encoded_by), by_source in positions.items():
        human = by_source.get("human", {})
        for source in MODEL_IDS:
            selfplay = by_source.get(source, {})
            for ply in sorted(human.keys() & selfplay.keys()):
                human_row, selfplay_row = human[ply], selfplay[ply]
                if human_row["bucket"] != selfplay_row["bucket"]:
                    raise ValueError("absolute-ply bucket mismatch in trajectory gap")
                for metric in metrics:
                    value = units[(source, encoded_by, human_row["bucket"], metric)][opening]
                    for label, row in (("human", human_row), ("selfplay", selfplay_row)):
                        value[label][0] += float(row[metric])
                        value[label][1] += 1
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    output = []
    for (source, encoded_by, bucket, metric), by_opening in sorted(units.items()):
        complete = [values for values in by_opening.values()
                    if values["human"][1] and values["selfplay"][1]]
        if not complete:
            continue
        array = np.asarray([[item["human"], item["selfplay"]] for item in complete], dtype=np.float64)
        totals = array.sum(0)
        human_mean, selfplay_mean = totals[:, 0] / totals[:, 1]
        draws = np.empty(replicates)
        for start in range(0, replicates, 64):
            stop = min(start + 64, replicates)
            sampled = array[rng.integers(0, len(array), size=(stop - start, len(array)))].sum(1)
            means = sampled[:, :, 0] / sampled[:, :, 1]
            draws[start:stop] = means[:, 1] - means[:, 0]
        low, high = np.percentile(draws, [2.5, 97.5])
        output.append({
            "trajectory_source": source, "encoded_by": encoded_by,
            "absolute_bucket": bucket, "metric": metric, "n_openings": len(complete),
            "human_mean": human_mean, "selfplay_mean": selfplay_mean,
            "selfplay_minus_human": selfplay_mean - human_mean,
            "ci_low": float(low), "ci_high": float(high),
        })
    return output


def _policy_association(rows):
    eligible = [row for row in rows if row["trajectory_source"] == "human" and row["policy_nll"] != ""]
    groups = defaultdict(list)
    for row in eligible:
        groups[(row["encoded_by"], row["bucket"])].append(row)
    output = []
    for (model, bucket), group in sorted(groups.items()):
        errors = np.asarray([float(row["board_nll"]) for row in group])
        boundaries = np.quantile(errors, [.25, .5, .75])
        quartiles = np.searchsorted(boundaries, errors, side="right") + 1
        for quartile in range(1, 5):
            selected = [row for row, q in zip(group, quartiles) if q == quartile]
            output.append({
                "encoded_by": model, "bucket": bucket, "probe_error_quartile": quartile,
                "n_positions": len(selected),
                "policy_top1": float(np.mean([int(row["policy_top1"]) for row in selected])),
                "policy_nll": float(np.mean([float(row["policy_nll"]) for row in selected])),
            })
    return output


def _rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def _spearman(x, y):
    if len(x) < 3:
        return math.nan
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _policy_spearman(rows, config):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row["trajectory_source"] == "human" and row["condition"] == "final_w256"
                and row["policy_nll"] != ""):
            groups[(row["encoded_by"], row["bucket"])][int(row["game_id"])].append(row)
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    replicates = int(config["bootstrap"]["replicates"])
    output = []
    for (model, bucket), games in sorted(groups.items()):
        pairs = np.asarray([
            (np.mean([float(row["board_nll"]) for row in game_rows]),
             np.mean([float(row["policy_nll"]) for row in game_rows]))
            for game_rows in games.values()
        ])
        if len(pairs) < 3:
            continue
        draws = []
        for _ in range(replicates):
            sample = pairs[rng.integers(0, len(pairs), size=len(pairs))]
            value = _spearman(sample[:, 0], sample[:, 1])
            if math.isfinite(value):
                draws.append(value)
        low, high = np.percentile(draws, [2.5, 97.5]) if draws else (math.nan, math.nan)
        output.append({
            "encoded_by": model, "bucket": bucket, "n_games": len(pairs),
            "spearman_game_means": _spearman(pairs[:, 0], pairs[:, 1]),
            "ci_low": float(low), "ci_high": float(high),
        })
    return output


def stage_evaluate(config: dict, resume: bool, device: torch.device) -> None:
    destination = output_dir(config) / "teacher_per_position.csv"
    diagnostic_destination = output_dir(config) / "teacher_diagnostics_per_position.csv"
    selfplay_destination = output_dir(config) / "selfplay_per_position.csv"
    split_rows = ensure_splits(config, resume=True)
    signature_inputs = {
        "evaluation": config.get("evaluation", {}),
        "splits": sha256(split_path(config)),
        "comparison": sha256(resolve_path(config["paths"]["comparison_output"]) / "manifest.json"),
        "novelty": sha256(output_dir(config) / "novelty" / "common_reference.json"),
        "probes": sha256(output_dir(config) / "train_probes_manifest.json"),
        "selfplay": sha256(output_dir(config) / "selfplay_manifest.json"),
    }
    evaluation_signature = config_hash(signature_inputs)
    existing_manifest_path = output_dir(config) / "evaluate_manifest.json"
    existing_manifest = (json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                         if existing_manifest_path.exists() else {})
    reuse = resume and existing_manifest.get("signature") == evaluation_signature
    components = dict(existing_manifest.get("component_sha256", {})) if reuse else {}

    def reusable(path):
        return (reuse and path.exists()
                and (existing_manifest.get("complete")
                     or components.get(path.name) == sha256(path)))

    def save_component(path):
        components[path.name] = sha256(path)
        atomic_json(existing_manifest_path, {
            "complete": False, "signature": evaluation_signature,
            "component_sha256": components,
        })
    all_test_games = {int(row["game_id"]) for row in split_rows if row["split"] == "test"}
    selected_test_games = {int(row["game_id"]) for row in split_rows
                           if row["split"] == "test" and int(row["selected"])}
    policy = _policy_lookup(config, all_test_games)
    bloom = _novelty_index(config)
    if not reusable(diagnostic_destination):
        diagnostic_rows = []
        for model_id in MODEL_IDS:
            for condition in CONDITIONS:
                data = _load_activation_split(config, model_id, "test", condition)
                diagnostic_rows.extend(_score_activation_rows(
                    config, model_id, condition, "linear", data, bloom, policy, device,
                    result_condition=("final_w256_sample" if condition == "final_w256" else condition),
                ))
                if condition == "final_w256":
                    diagnostic_rows.extend(_score_activation_rows(
                        config, model_id, condition, "mlp", data, bloom, policy, device,
                        result_condition="final_w256_mlp",
                    ))
        write_csv(diagnostic_destination, diagnostic_rows, RESULT_FIELDS)
        save_component(diagnostic_destination)
    else:
        print("evaluate: reusing teacher diagnostic rows", flush=True)
    diagnostic_rows = read_csv(diagnostic_destination)
    if not reusable(destination):
        if config.get("evaluation", {}).get("full_test_pool", True):
            _stream_full_teacher(config, split_rows, bloom, policy, device, destination)
        else:
            write_csv(destination,
                      (row for row in diagnostic_rows if row["condition"] == "final_w256"),
                      RESULT_FIELDS)
        save_component(destination)
    else:
        print("evaluate: reusing primary teacher rows", flush=True)
    if not reusable(selfplay_destination):
        models = load_decoders(config, device)
        selfplay_rows = _evaluate_selfplay(config, models, bloom, device)
        write_csv(selfplay_destination, selfplay_rows, RESULT_FIELDS)
        save_component(selfplay_destination)
    else:
        print("evaluate: reusing self-play rows", flush=True)
        selfplay_rows = read_csv(selfplay_destination)
    # The primary final-layer rows come from the complete test pool. Layer,
    # context, and MLP diagnostics use the selected CPU-first held-out sample.
    summary = _mean_summary(chain(_iter_csv(destination), iter(diagnostic_rows),
                                  iter(selfplay_rows)))
    summary_fields = ("trajectory_source", "encoded_by", "condition", "bucket", "n_units",
                      "n_positions", "novel_fraction", *METRIC_FIELDS)
    write_csv(output_dir(config) / "bucket_summary.csv", summary, summary_fields)
    deltas = _paired_bootstrap(chain(_iter_csv(destination), iter(diagnostic_rows)), config)
    write_csv(output_dir(config) / "paired_deltas.csv", deltas,
              ("condition", "bucket", "metric", "n_games", "balanced_minus_baseline", "ci_low", "ci_high"))
    interactions = _cross_replay_bootstrap(selfplay_rows, config)
    write_csv(output_dir(config) / "cross_replay_interactions.csv", interactions,
              ("trajectory_source", "bucket", "metric", "n_openings", "baseline_encoder", "balanced_encoder", "balanced_minus_baseline", "ci_low", "ci_high"))
    drift = _drift_bootstrap(chain(_iter_csv(destination), iter(diagnostic_rows), iter(selfplay_rows)), config)
    write_csv(output_dir(config) / "drift_interactions.csv", drift,
              ("trajectory_source", "encoded_by", "condition", "metric", "n_units", "early_bucket", "late_bucket", "late_minus_early", "difference_in_differences", "ci_low", "ci_high"))
    selected_primary = [dict(row, condition="final_w256") for row in diagnostic_rows
                        if row["condition"] == "final_w256_sample"]
    gaps = _trajectory_gap_bootstrap(selected_primary + selfplay_rows, config)
    write_csv(output_dir(config) / "trajectory_gaps.csv", gaps,
              ("trajectory_source", "encoded_by", "absolute_bucket", "metric", "n_openings", "human_mean", "selfplay_mean", "selfplay_minus_human", "ci_low", "ci_high"))
    novelty = [{key: row[key] for key in ("trajectory_source", "encoded_by", "condition", "bucket", "n_units", "n_positions", "novel_fraction")}
               for row in summary]
    write_csv(output_dir(config) / "novelty_summary.csv", novelty, novelty[0].keys())
    associations = _policy_association(selected_primary)
    write_csv(output_dir(config) / "policy_error_association.csv", associations,
              ("encoded_by", "bucket", "probe_error_quartile", "n_positions", "policy_top1", "policy_nll"))
    spearman = _policy_spearman(selected_primary, config)
    write_csv(output_dir(config) / "policy_probe_spearman.csv", spearman,
              ("encoded_by", "bucket", "n_games", "spearman_game_means", "ci_low", "ci_high"))
    atomic_json(output_dir(config) / "evaluate_manifest.json", {
        "complete": True, "teacher_rows": _csv_row_count(destination),
        "teacher_diagnostic_rows": len(diagnostic_rows), "selfplay_rows": len(selfplay_rows),
        "full_test_pool": bool(config.get("evaluation", {}).get("full_test_pool", True)),
        "test_games": len(all_test_games), "selected_diagnostic_test_games": len(selected_test_games),
        "bootstrap_unit_teacher": "complete game", "bootstrap_unit_selfplay": "opening",
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "signature": evaluation_signature,
        "component_sha256": components,
    })


def stage_plot(config: dict) -> None:
    from analysis.state_probe.plot_results import main as plot_main
    plot_main(["--output-dir", str(output_dir(config))])


STAGES = ("index-reference", "extract-teacher", "train-probes", "generate-selfplay", "evaluate", "plot")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/state_probe_twic_1662_1667.yaml")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    destination = output_dir(config)
    destination.mkdir(parents=True, exist_ok=True)
    ensure_splits(config, args.resume)
    device = choose_device(config.get("device", "auto"))
    stages = STAGES if args.stage == "all" else (args.stage,)
    started = datetime.now(timezone.utc)
    for stage in stages:
        print(f"\n=== state-probe stage: {stage} ===", flush=True)
        if stage == "index-reference":
            stage_index_reference(config, args.resume)
        elif stage == "extract-teacher":
            stage_extract_teacher(config, args.resume, device)
        elif stage == "train-probes":
            stage_train_probes(config, args.resume, device)
        elif stage == "generate-selfplay":
            stage_generate_selfplay(config, args.resume, device)
        elif stage == "evaluate":
            stage_evaluate(config, args.resume, device)
        elif stage == "plot":
            stage_plot(config)
    experiment_manifest_path = destination / "experiment_manifest.json"
    previous_manifest = (json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
                         if experiment_manifest_path.exists() else {})
    completed = set(previous_manifest.get("complete_stages", ())).union(stages)
    atomic_json(experiment_manifest_path, {
        "format": "decoder_state_probe_v1",
        "complete_stages": [stage for stage in STAGES if stage in completed],
        "started_at_utc": previous_manifest.get("started_at_utc", started.isoformat()),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "inputs": {
            "pgn": {"path": str(resolve_path(config["paths"]["pgn"])), "sha256": sha256(resolve_path(config["paths"]["pgn"]))},
            **{model: {"path": str(resolve_path(config["models"][model])), "sha256": sha256(resolve_path(config["models"][model]))}
               for model in MODEL_IDS},
        },
        "drift_definition": "history-based state recoverability and policy-induced trajectory shift; no recurrent state is carried",
    })
    print(f"State-probe stage(s) complete: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
