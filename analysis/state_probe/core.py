"""Core data, model, probe, novelty, and metric utilities."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Iterator

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from analysis.ply_comparison.aggregate import bucket_for_ply
from analysis.ply_comparison.evaluate import ROOT, load_weights, sha256
from dataset.offsets import load_pgn_offsets
from dataset.renderer import board_to_grid_ids
from models.chess_decoder_player import ChessDecoderPlayer


DECODER_CHECKPOINTS = {
    "decoder_baseline": ROOT / "chess_llm_decoder_v12.pt",
    "decoder_balanced": ROOT / "chess_llm_decoder_balanced_v1_final.pt",
}
MODEL_IDS = tuple(DECODER_CHECKPOINTS)
LAYERS = (3, 6, 9, 12)
CONDITIONS = (
    "final_w32", "final_w64", "final_w128", "final_w256",
    "layer3_w256", "layer6_w256", "layer9_w256",
)
GENERATED_BUCKETS = ("0_10", "10_20", "20_40", "40_80", "80_120")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("state-probe config must be a mapping")
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = sha256(path)
    for section, keys in {
        "paths": ("pgn", "index", "tokenizer", "comparison_output", "output_dir"),
        "models": MODEL_IDS,
    }.items():
        if section not in config:
            raise ValueError(f"missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"missing config key: {section}.{key}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(name)


def load_decoders(config: dict, device: torch.device) -> dict[str, ChessDecoderPlayer]:
    tokenizer = resolve_path(config["paths"]["tokenizer"])
    result = {}
    for model_id in MODEL_IDS:
        path = resolve_path(config["models"][model_id])
        model = ChessDecoderPlayer(
            tokenizer_path=str(tokenizer), model_dim=256, mlp_ratio=4.0,
            n_heads=4, n_layers=12, dropout=0.1, max_seq_len=256,
            tie_weights=True,
        )
        state = load_weights(path)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result[model_id] = model
    if result[MODEL_IDS[0]].decoder.tokenizer.id2move != result[MODEL_IDS[1]].decoder.tokenizer.id2move:
        raise ValueError("decoder tokenizers differ")
    return result


def progress(label: str, done: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed
    remaining = max(0, total - done)
    eta = remaining / rate if rate else math.inf
    eta_text = f"{eta / 60:.1f} min" if math.isfinite(eta) else "unknown"
    percent = 100 * done / total if total else 100.0
    print(f"{label}: {done:,}/{total:,} ({percent:.2f}%), {rate:.2f}/s, ETA {eta_text}", flush=True)


def read_indexed_game(source, row: dict) -> tuple[chess.pgn.Game, bytes]:
    source.seek(int(row["start"]))
    raw = source.read(int(row["end"]) - int(row["start"]))
    game = chess.pgn.read_game(io.StringIO(raw.decode("utf-8", errors="replace")))
    if game is None or game.errors:
        raise ValueError(f"invalid indexed PGN game {row['idx']}")
    return game, raw


def assign_split(raw: bytes) -> str:
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 100
    return "train" if value < 70 else ("val" if value < 85 else "test")


def build_splits(config: dict) -> list[dict]:
    pgn_path = resolve_path(config["paths"]["pgn"])
    index = load_pgn_offsets(str(resolve_path(config["paths"]["index"])))
    sample = config["sample_games_per_issue"]
    ranked: dict[tuple[str, int], list[tuple[bytes, dict]]] = {}
    started = time.monotonic()
    with pgn_path.open("rb") as source:
        for position, row in enumerate(index["games"], start=1):
            game, raw = read_indexed_game(source, row)
            issue = int(game.headers["TWICIssue"])
            split = assign_split(raw)
            item = {
                "game_id": int(row["idx"]), "twic_issue": issue, "split": split,
                "sha256": hashlib.sha256(raw).hexdigest(), "selected": 0,
            }
            ranked.setdefault((split, issue), []).append((hashlib.sha256(b"probe-rank\0" + raw).digest(), item))
            if position % 1000 == 0 or position == len(index["games"]):
                progress("Split scan", position, len(index["games"]), started)
    output = []
    for (split, issue), rows in sorted(ranked.items()):
        limit = int(sample[split])
        for selected_index, (rank, item) in enumerate(sorted(rows, key=lambda pair: pair[0])):
            item["selected"] = int(selected_index < limit)
            output.append(item)
    output.sort(key=lambda row: row["game_id"])
    return output


def write_csv(path: Path, rows: Iterable[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def board_target(board: chess.Board) -> dict:
    board_ids = board_to_grid_ids(board).reshape(-1).astype(np.int64)
    castling = np.asarray([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ], dtype=np.int64)
    return {
        "board": board_ids,
        "turn": int(board.turn == chess.WHITE),
        "castling": castling,
        "ep": 0 if board.ep_square is None else board.ep_square + 1,
    }


def history_tokens(game: chess.pgn.Game, tokenizer) -> Iterator[tuple[int, chess.Board, list[int], list[int]]]:
    board = game.board()
    move_ids = [tokenizer.bos_id]
    piece_ids = [tokenizer.p_bos_id]
    yield 0, board.copy(stack=False), move_ids.copy(), piece_ids.copy()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if move not in board.legal_moves:
            raise ValueError(f"illegal move at ply {ply - 1}")
        piece = board.piece_at(move.from_square)
        if piece is None or move.uci() not in tokenizer.move2id:
            raise ValueError(f"unencodable move at ply {ply - 1}")
        move_ids.append(tokenizer.move2id[move.uci()])
        piece_ids.append(tokenizer.piece2id[piece.symbol().upper()])
        board.push(move)
        yield ply, board.copy(stack=False), move_ids.copy(), piece_ids.copy()


def truncate_history(move_ids: list[int], piece_ids: list[int], initial_turn: int, window: int):
    start = max(0, len(move_ids) - window)
    first_history_ply = max(0, start - 1)
    return move_ids[start:], piece_ids[start:], (initial_turn + first_history_ply) % 2


@torch.inference_mode()
def extract_conditions(model: ChessDecoderPlayer, move_ids: list[int], piece_ids: list[int],
                       initial_turn: int, device: torch.device) -> dict[str, np.ndarray]:
    result = {}
    for window in (32, 64, 128):
        ids, pieces, start_turn = truncate_history(move_ids, piece_ids, initial_turn, window)
        output = model(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            piece_input_ids=torch.tensor([pieces], dtype=torch.long, device=device),
            start_turn=torch.tensor([start_turn], dtype=torch.long, device=device),
            capture_layers=(12,),
        )
        result[f"final_w{window}"] = output["hidden_states"][12][0, -1].float().cpu().numpy()
    ids, pieces, start_turn = truncate_history(move_ids, piece_ids, initial_turn, 256)
    output = model(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        piece_input_ids=torch.tensor([pieces], dtype=torch.long, device=device),
        start_turn=torch.tensor([start_turn], dtype=torch.long, device=device),
        capture_layers=LAYERS,
    )
    for layer in LAYERS:
        condition = "final_w256" if layer == 12 else f"layer{layer}_w256"
        result[condition] = output["hidden_states"][layer][0, -1].float().cpu().numpy()
    return result


@torch.inference_mode()
def extract_condition_batch(model: ChessDecoderPlayer,
                            histories: list[tuple[list[int], list[int]]],
                            initial_turn: int, device: torch.device,
                            batch_size: int = 64,
                            requested_conditions: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Extract all conditions while batching variable-length position histories."""
    requested = tuple(requested_conditions or CONDITIONS)
    unknown = set(requested).difference(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown representation conditions: {sorted(unknown)}")
    collected: dict[str, list[np.ndarray]] = {condition: [] for condition in requested}
    windows = []
    for window in (32, 64, 128):
        if f"final_w{window}" in requested:
            windows.append(window)
    if any(condition.endswith("_w256") or condition == "final_w256" for condition in requested):
        windows.append(256)
    for offset in range(0, len(histories), batch_size):
        batch = histories[offset:offset + batch_size]
        for window in windows:
            truncated = [truncate_history(ids, pieces, initial_turn, window)
                         for ids, pieces in batch]
            length = max(len(item[0]) for item in truncated)
            ids_tensor = torch.full((len(batch), length), model.pad_id,
                                    dtype=torch.long, device=device)
            piece_tensor = torch.full((len(batch), length),
                                      model.decoder.tokenizer.p_pad_id,
                                      dtype=torch.long, device=device)
            attention = torch.zeros((len(batch), length), dtype=torch.bool, device=device)
            last = []
            starts = []
            for index, (ids, pieces, start_turn) in enumerate(truncated):
                n = len(ids)
                ids_tensor[index, :n] = torch.tensor(ids, device=device)
                piece_tensor[index, :n] = torch.tensor(pieces, device=device)
                attention[index, :n] = True
                last.append(n - 1)
                starts.append(start_turn)
            if window == 256:
                capture = tuple(layer for layer in LAYERS
                                if ("final_w256" if layer == 12 else f"layer{layer}_w256") in requested)
            else:
                capture = (12,)
            output = model(
                input_ids=ids_tensor, piece_input_ids=piece_tensor,
                attention_mask=attention,
                start_turn=torch.tensor(starts, dtype=torch.long, device=device),
                capture_layers=capture,
            )
            row_index = torch.arange(len(batch), device=device)
            last_index = torch.tensor(last, device=device)
            if window < 256:
                value = output["hidden_states"][12][row_index, last_index]
                collected[f"final_w{window}"].append(value.float().cpu().numpy())
            else:
                for layer in LAYERS:
                    condition = "final_w256" if layer == 12 else f"layer{layer}_w256"
                    if condition in collected:
                        value = output["hidden_states"][layer][row_index, last_index]
                        collected[condition].append(value.float().cpu().numpy())
    return {key: np.concatenate(value).astype(np.float16)
            for key, value in collected.items()}


def normalized_position(board: chess.Board) -> str:
    return " ".join(board.fen(en_passant="legal").split()[:4])


def position_digest(board: chess.Board) -> bytes:
    return hashlib.blake2b(normalized_position(board).encode("ascii"), digest_size=16).digest()


class BloomIndex:
    """Double-hashed Bloom filter with portable metadata."""

    def __init__(self, capacity: int, false_positive_rate: float = 1e-4, *, bits=None):
        if capacity < 1 or not 0 < false_positive_rate < 1:
            raise ValueError("invalid Bloom-filter parameters")
        self.capacity = int(capacity)
        self.false_positive_rate = float(false_positive_rate)
        self.bit_count = int(math.ceil(-capacity * math.log(false_positive_rate) / math.log(2) ** 2))
        self.hash_count = max(1, int(round(self.bit_count / capacity * math.log(2))))
        self.bits = bytearray((self.bit_count + 7) // 8) if bits is None else bytearray(bits)
        if len(self.bits) != (self.bit_count + 7) // 8:
            raise ValueError("Bloom bit-array length mismatch")
        self.insertions = 0

    def _indices(self, digest: bytes):
        if len(digest) != 16:
            digest = hashlib.blake2b(digest, digest_size=16).digest()
        a = int.from_bytes(digest[:8], "little")
        b = int.from_bytes(digest[8:], "little") | 1
        for i in range(self.hash_count):
            yield (a + i * b) % self.bit_count

    def add_digest(self, digest: bytes) -> None:
        for index in self._indices(digest):
            self.bits[index >> 3] |= 1 << (index & 7)
        self.insertions += 1

    def contains_digest(self, digest: bytes) -> bool:
        return all(self.bits[index >> 3] & (1 << (index & 7)) for index in self._indices(digest))

    def save(self, path: Path, metadata_path: Path, extra: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(self.bits)
        os.replace(temporary, path)
        occupancy = sum(byte.bit_count() for byte in self.bits) / self.bit_count
        observed_fpr = occupancy ** self.hash_count
        atomic_json(metadata_path, {
            "format": "state_probe_bloom_v1", "capacity": self.capacity,
            "target_false_positive_rate": self.false_positive_rate,
            "estimated_false_positive_rate": observed_fpr,
            "bit_count": self.bit_count, "hash_count": self.hash_count,
            "insertions": self.insertions, **extra,
        })

    @classmethod
    def load(cls, path: Path, metadata_path: Path):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = cls(metadata["capacity"], metadata["target_false_positive_rate"], bits=path.read_bytes())
        result.insertions = metadata["insertions"]
        return result


class StateProbe(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.trunk = nn.Identity()
            head_dim = dim
        else:
            self.trunk = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            head_dim = hidden_dim
        self.occupancy = nn.Linear(head_dim, 64)
        self.piece = nn.Linear(head_dim, 64 * 12)
        self.turn = nn.Linear(head_dim, 2)
        self.castling = nn.Linear(head_dim, 4)
        self.ep = nn.Linear(head_dim, 65)

    def forward(self, x):
        h = self.trunk(x)
        return {
            "occupancy": self.occupancy(h),
            "piece": self.piece(h).reshape(-1, 64, 12),
            "turn": self.turn(h), "castling": self.castling(h), "ep": self.ep(h),
        }


def probe_loss(output: dict, board: torch.Tensor, turn: torch.Tensor,
               castling: torch.Tensor, ep: torch.Tensor) -> torch.Tensor:
    occupied = board > 0
    occupancy_loss = F.binary_cross_entropy_with_logits(output["occupancy"], occupied.float())
    if occupied.any():
        piece_loss = F.cross_entropy(output["piece"][occupied], board[occupied] - 1)
    else:
        piece_loss = output["piece"].sum() * 0
    return (occupancy_loss + piece_loss + F.cross_entropy(output["turn"], turn)
            + F.binary_cross_entropy_with_logits(output["castling"], castling.float())
            + F.cross_entropy(output["ep"], ep))


def ensemble_metrics(outputs: list[dict[str, torch.Tensor]], board: torch.Tensor,
                     turn: torch.Tensor, castling: torch.Tensor, ep: torch.Tensor) -> dict[str, np.ndarray]:
    occ_prob = torch.stack([torch.sigmoid(x["occupancy"]) for x in outputs]).mean(0)
    piece_prob = torch.stack([torch.softmax(x["piece"], -1) for x in outputs]).mean(0)
    turn_prob = torch.stack([torch.softmax(x["turn"], -1) for x in outputs]).mean(0)
    castle_prob = torch.stack([torch.sigmoid(x["castling"]) for x in outputs]).mean(0)
    ep_prob = torch.stack([torch.softmax(x["ep"], -1) for x in outputs]).mean(0)
    occupied = board > 0
    eps = 1e-7
    occ_nll = -(occupied * torch.log(occ_prob.clamp_min(eps))
                + (~occupied) * torch.log((1 - occ_prob).clamp_min(eps)))
    piece_nll = torch.zeros_like(occ_nll)
    if occupied.any():
        occupied_piece_prob = piece_prob[occupied].gather(
            1, (board[occupied] - 1).unsqueeze(1)
        ).squeeze(1)
        piece_nll[occupied] = -torch.log(occupied_piece_prob.clamp_min(eps))
    pred_occ = occ_prob >= 0.5
    pred_piece = piece_prob.argmax(-1) + 1
    pred_board = torch.where(pred_occ, pred_piece, 0)
    true_positive = (pred_occ & occupied).sum(-1).float()
    precision = true_positive / pred_occ.sum(-1).clamp_min(1)
    recall = true_positive / occupied.sum(-1).clamp_min(1)
    occupied_count = occupied.sum(-1).clamp_min(1)
    seed_board_nll = []
    for output in outputs:
        seed_occ = torch.sigmoid(output["occupancy"])
        seed_piece = torch.softmax(output["piece"], -1)
        seed_occ_nll = -(occupied * torch.log(seed_occ.clamp_min(eps))
                         + (~occupied) * torch.log((1 - seed_occ).clamp_min(eps)))
        seed_piece_nll = torch.zeros_like(seed_occ_nll)
        if occupied.any():
            selected = seed_piece[occupied].gather(
                1, (board[occupied] - 1).unsqueeze(1)
            ).squeeze(1)
            seed_piece_nll[occupied] = -torch.log(selected.clamp_min(eps))
        seed_board_nll.append((seed_occ_nll + seed_piece_nll).mean(-1))
    seed_std = torch.stack(seed_board_nll).std(0, unbiased=False)
    return {
        "board_nll": (occ_nll + piece_nll).mean(-1).cpu().numpy(),
        "occupied_nll": ((occ_nll + piece_nll) * occupied).sum(-1).div(occupied_count).cpu().numpy(),
        "board_hamming": (pred_board != board).float().mean(-1).cpu().numpy(),
        "occupancy_precision": precision.cpu().numpy(),
        "occupancy_recall": recall.cpu().numpy(),
        "occupancy_f1": (2 * precision * recall / (precision + recall).clamp_min(eps)).cpu().numpy(),
        "piece_accuracy_occupied": (((pred_piece == board) & occupied).sum(-1).float() / occupied_count).cpu().numpy(),
        "square_accuracy_13": (pred_board == board).float().mean(-1).cpu().numpy(),
        "exact_board": (pred_board == board).all(-1).float().cpu().numpy(),
        "turn_accuracy": (turn_prob.argmax(-1) == turn).float().cpu().numpy(),
        "castling_exact": ((castle_prob >= .5) == castling.bool()).all(-1).float().cpu().numpy(),
        "ep_accuracy": (ep_prob.argmax(-1) == ep).float().cpu().numpy(),
        "seed_board_nll_std": seed_std.cpu().numpy(),
    }


def generated_bucket(ply: int) -> str:
    for start, stop in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 120)):
        if start <= ply < stop:
            return f"{start}_{stop}"
    raise ValueError(f"generated ply outside configured range: {ply}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
