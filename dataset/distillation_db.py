"""distillation_db.py

Efficient Stockfish distillation database builder + PyTorch datasets.

Goals:
  - Offline: for a sampled subset of plies in each PGN game, store:
      * FEN at that ply (position before the played move)
      * PGN prefix (SAN) up to that ply (optional convenience)
      * UCI move history prefix up to that ply (robust for tokenization)
      * Top-K Stockfish moves with (cp/mate, depth)
  - Online: load those records efficiently and construct teacher probability
    distributions (decoder: vocab distribution; encoder: 64x64 from->to + promo).

Design choices:
  - Storage: sharded SQLite (one file per process) for fast random access and
    parallel writes.
  - Build parallelism: multiprocessing with 1 Stockfish process per worker.
  - Reading: one SQLite connection per DataLoader worker process.
"""

from __future__ import annotations

import io
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import chess
import chess.pgn
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.stockfish_teacher import StockfishUCI, _score_to_scalar, _uci_to_from_to_promo

# Import helpers: this repo sometimes exposes these modules either as top-level
# files (e.g. offsets.py) or inside a package (e.g. dataset/offsets.py).
from dataset.offsets import load_pgn_offsets
from dataset.renderer import board_to_grid_ids



def _ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _export_pgn_prefix(game: chess.pgn.Game, ply: int) -> str:
    """Export SAN moves up to `ply` (number of half-moves)."""
    g2 = chess.pgn.Game()
    g2.headers.clear()
    node = g2
    moves = list(game.mainline_moves())
    for i in range(min(ply, len(moves))):
        node = node.add_variation(moves[i])
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return g2.accept(exporter).strip()


def _softmax_from_scores(scores: torch.Tensor, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-6)
    x = scores / tau
    x = x - x.max()
    return torch.softmax(x, dim=-1)


def _scores_to_probs(
    cps: Sequence[Optional[int]],
    mates: Sequence[Optional[int]],
    *,
    tau: float,
    mate_cp: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    scalars = torch.tensor(
        [_score_to_scalar(cp, mate, mate_cp=mate_cp) for (cp, mate) in zip(cps, mates)],
        dtype=torch.float32,
        device=device,
    )
    return _softmax_from_scores(scalars, tau=tau)


@dataclass(frozen=True)
class DistillRecord:
    game_list_idx: int
    ply_idx: int
    fen: str
    pgn_san_prefix: str
    uci_prefix: List[str]
    uci_topk: List[str]
    cp_topk: List[Optional[int]]
    mate_topk: List[Optional[int]]
    depth_topk: List[int]


class DistillationShardWriter:
    """One-writer SQLite shard (safe per process)."""

    def __init__(self, shard_path: str) -> None:
        self.shard_path = shard_path
        self.conn = sqlite3.connect(self.shard_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._create_schema()
        self._pending: List[Tuple[Any, ...]] = []

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              game_list_idx INTEGER NOT NULL,
              ply_idx INTEGER NOT NULL,
              fen TEXT NOT NULL,
              pgn_san_prefix TEXT NOT NULL,
              uci_prefix_json TEXT NOT NULL,
              uci_topk_json TEXT NOT NULL,
              cp_topk_json TEXT NOT NULL,
              mate_topk_json TEXT NOT NULL,
              depth_topk_json TEXT NOT NULL
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_game_ply ON samples(game_list_idx, ply_idx);")
        self.conn.commit()

    def add(self, rec: DistillRecord) -> None:
        self._pending.append(
            (
                int(rec.game_list_idx),
                int(rec.ply_idx),
                rec.fen,
                rec.pgn_san_prefix,
                json.dumps(rec.uci_prefix, ensure_ascii=False),
                json.dumps(rec.uci_topk, ensure_ascii=False),
                json.dumps(rec.cp_topk),
                json.dumps(rec.mate_topk),
                json.dumps(rec.depth_topk),
            )
        )

    def flush(self, chunk_size: int = 1024) -> None:
        if not self._pending:
            return
        cur = self.conn.cursor()
        while self._pending:
            chunk = self._pending[:chunk_size]
            self._pending = self._pending[chunk_size:]
            cur.executemany(
                """
                INSERT INTO samples(
                  game_list_idx, ply_idx, fen, pgn_san_prefix,
                  uci_prefix_json, uci_topk_json, cp_topk_json, mate_topk_json, depth_topk_json
                ) VALUES(?,?,?,?,?,?,?,?,?);
                """,
                chunk,
            )
        self.conn.commit()

    def close(self) -> None:
        self.flush()
        self.conn.commit()
        self.conn.close()


class DistillationShardReader:
    """One-reader SQLite shard (connection opened lazily per process)."""

    def __init__(self, shard_path: str) -> None:
        self.shard_path = shard_path
        self._conn: Optional[sqlite3.Connection] = None

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_conn"] = None
        return st

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.shard_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def __len__(self) -> int:
        conn = self._get_conn()
        (n,) = conn.execute("SELECT COUNT(*) FROM samples;").fetchone()
        return int(n)
    
    def get_by_index(self, idx0: int) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM samples WHERE id = ?;",
            (int(idx0) + 1,),
        ).fetchone()
        if row is None:
            raise IndexError(idx0)
        return dict(row)

    # def get_by_index(self, idx0: int) -> Dict[str, Any]:
    #     conn = self._get_conn()
    #     row = conn.execute(
    #         "SELECT * FROM samples ORDER BY id LIMIT 1 OFFSET ?;",
    #         (int(idx0),),
    #     ).fetchone()
    #     if row is None:
    #         raise IndexError(idx0)
    #     return dict(row)


def _worker_build_shard(
    *,
    shard_path: str,
    pgn_path: str,
    index_path: str,
    samples: Sequence[Tuple[int, int]],
    engine_path: str,
    movetime_ms: int,
    k: int,
    min_depth: int,
    seed: int,
    log_every: int = 0,
) -> Dict[str, Any]:
    """Worker entrypoint: write one shard."""
    _ = random.Random(seed)
    engine = StockfishUCI(
        engine_path=engine_path,
        threads=1,
        hash_mb=256,
        multipv=k,
    )

    idx = load_pgn_offsets(index_path)
    games = idx["games"]
    fh = open(pgn_path, "rb")
    writer = DistillationShardWriter(shard_path)

    def read_game(game_list_idx: int) -> Optional[chess.pgn.Game]:
        g = games[game_list_idx]
        fh.seek(g["start"])
        raw = fh.read(g["end"] - g["start"])
        txt = raw.decode("utf-8", errors="ignore")
        try:
            return chess.pgn.read_game(io.StringIO(txt))
        except Exception:
            return None

    game_cache: Dict[int, Optional[chess.pgn.Game]] = {}
    n_ok = 0
    n_bad = 0
    for j, (game_list_idx, ply_idx) in enumerate(samples):
        if game_list_idx not in game_cache:
            game_cache[game_list_idx] = read_game(game_list_idx)
        game = game_cache[game_list_idx]
        if game is None:
            n_bad += 1
            continue

        moves = list(game.mainline_moves())
        if ply_idx < 0 or ply_idx >= len(moves):
            n_bad += 1
            continue

        board = game.board()
        try:
            for t in range(ply_idx):
                board.push(moves[t])
        except Exception:
            n_bad += 1
            continue

        fen = board.fen()
        pgn_prefix = _export_pgn_prefix(game, ply=ply_idx)
        uci_prefix = [mv.uci() for mv in moves[:ply_idx]]

        lines = engine.analyze_fen(
            fen,
            movetime_ms=movetime_ms,
            multipv=k,
            min_depth=min_depth,
            read_timeout_s=max(5.0, movetime_ms / 1000.0 * 3),
        )

        uci_topk = [ln.uci for ln in lines]
        cp_topk = [ln.cp for ln in lines]
        mate_topk = [ln.mate for ln in lines]
        depth_topk = [int(ln.depth) for ln in lines]

        writer.add(
            DistillRecord(
                game_list_idx=int(game_list_idx),
                ply_idx=int(ply_idx),
                fen=fen,
                pgn_san_prefix=pgn_prefix,
                uci_prefix=uci_prefix,
                uci_topk=uci_topk,
                cp_topk=cp_topk,
                mate_topk=mate_topk,
                depth_topk=depth_topk,
            )
        )
        n_ok += 1
        if len(writer._pending) >= 1024:
            writer.flush()
        if log_every and (j + 1) % log_every == 0:
            pass

    writer.close()
    fh.close()
    engine.close()
    return {"written": n_ok, "skipped": n_bad, "shard_path": shard_path}


class DistillationDBBuilder:
    """Build sharded SQLite DB from a precomputed samples JSON."""

    def __init__(
        self,
        *,
        pgn_path: str,
        index_path: str,
        samples_path: str,
        engine_path: str,
        out_dir: str,
        movetime_ms: int = 100,
        k: int = 5,
        min_depth: int = 1,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        self.pgn_path = pgn_path
        self.index_path = index_path
        self.samples_path = samples_path
        self.engine_path = engine_path
        self.out_dir = out_dir
        self.movetime_ms = int(movetime_ms)
        self.k = int(k)
        self.min_depth = int(min_depth)
        self.num_workers = max(1, int(num_workers))
        self.seed = int(seed)

    def _load_samples(self) -> List[Tuple[int, int]]:
        with open(self.samples_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return [(int(a), int(b)) for a, b in payload["samples"]]

    def build(self, *, overwrite: bool = False) -> str:
        _ensure_dir(self.out_dir)
        meta_path = os.path.join(self.out_dir, "meta.json")
        if os.path.exists(meta_path) and not overwrite:
            return meta_path

        samples = self._load_samples()
        rng = random.Random(self.seed)
        rng.shuffle(samples)

        shards: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_workers)]
        for i, s in enumerate(samples):
            shards[i % self.num_workers].append(s)

        shard_paths = [os.path.join(self.out_dir, f"shard_{i:03d}.sqlite") for i in range(self.num_workers)]
        if overwrite:
            for sp in shard_paths:
                if os.path.exists(sp):
                    os.remove(sp)


        total_samples = len(samples)
        done_samples = 0
        t0 = time.time()

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
            futs = []
            shard_sizes = []
            for i, (sp, shard_samples) in enumerate(zip(shard_paths, shards)):
                shard_sizes.append(len(shard_samples))
                futs.append(
                    ex.submit(
                        _worker_build_shard,
                        shard_path=sp,
                        pgn_path=self.pgn_path,
                        index_path=self.index_path,
                        samples=shard_samples,
                        engine_path=self.engine_path,
                        movetime_ms=self.movetime_ms,
                        k=self.k,
                        min_depth=self.min_depth,
                        seed=self.seed + 1000 * i,
                    )
                )

            # Map future -> shard size so we can update done count on completion
            fut2n = {fu: n for fu, n in zip(futs, shard_sizes)}

            for fu in as_completed(futs):
                res = fu.result()
                results.append(res)
                done_samples += fut2n[fu]

                elapsed = time.time() - t0
                rate = done_samples / max(elapsed, 1e-6)
                remaining = (total_samples - done_samples) / max(rate, 1e-6)

                print(
                    f"[distill_db] {done_samples}/{total_samples} "
                    f"({done_samples/total_samples:.1%}) | "
                    f"{rate:.1f} pos/s | ETA {remaining/60:.1f} min"
                )

        # results: List[Dict[str, Any]] = []
        # with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
        #     futs = []
        #     for i, (sp, shard_samples) in enumerate(zip(shard_paths, shards)):
        #         futs.append(
        #             ex.submit(
        #                 _worker_build_shard,
        #                 shard_path=sp,
        #                 pgn_path=self.pgn_path,
        #                 index_path=self.index_path,
        #                 samples=shard_samples,
        #                 engine_path=self.engine_path,
        #                 movetime_ms=self.movetime_ms,
        #                 k=self.k,
        #                 min_depth=self.min_depth,
        #                 seed=self.seed + 1000 * i,
        #             )
        #         )
        #     for fu in futs:
        #         results.append(fu.result())

        shard_counts: List[int] = []
        for sp in shard_paths:
            shard_counts.append(len(DistillationShardReader(sp)))

        meta = {
            "format": "sqlite_shards_v1",
            "pgn_path": os.path.abspath(self.pgn_path),
            "index_path": os.path.abspath(self.index_path),
            "samples_path": os.path.abspath(self.samples_path),
            "engine": {"movetime_ms": self.movetime_ms, "k": self.k, "min_depth": self.min_depth},
            "num_shards": self.num_workers,
            "shards": [{"path": os.path.abspath(sp), "count": int(c)} for sp, c in zip(shard_paths, shard_counts)],
            "build_results": results,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta_path


class _MultiShardIndex:
    def __init__(self, meta_path: str) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.shards = [(s["path"], int(s["count"])) for s in meta["shards"]]
        self.cum: List[int] = []
        total = 0
        for _, c in self.shards:
            total += c
            self.cum.append(total)
        self.total = total

    def locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        prev = 0
        for shard_id, c in enumerate(self.cum):
            if idx < c:
                return shard_id, idx - prev
            prev = c
        raise IndexError(idx)


class DistillationDecoderDataset(Dataset):
    """Loads records and returns history tokens + sparse teacher over vocab."""

    def __init__(
        self,
        *,
        meta_path: str,
        move2id: Dict[str, int],
        vocab_size: int,
        bos_id: Optional[int] = None,
        unk_id: Optional[int] = None,
        tau: float = 50.0,
        mate_cp: int = 20000,
        return_text: bool = False,
    ) -> None:
        self.index = _MultiShardIndex(meta_path)
        self.readers = [DistillationShardReader(p) for (p, _) in self.index.shards]
        self.move2id = move2id
        self.vocab_size = int(vocab_size)
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.tau = float(tau)
        self.mate_cp = int(mate_cp)
        self.return_text = bool(return_text)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["readers"] = [DistillationShardReader(p) for (p, _) in self.index.shards]
        return st

    def __len__(self) -> int:
        return self.index.total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local = self.index.locate(idx)
        row = self.readers[shard_id].get_by_index(local)

        uci_prefix = json.loads(row["uci_prefix_json"])
        ids: List[int] = []
        if self.bos_id is not None:
            ids.append(int(self.bos_id))
        for u in uci_prefix:
            if u in self.move2id:
                ids.append(int(self.move2id[u]))
            elif self.unk_id is not None:
                ids.append(int(self.unk_id))

        uci_topk = json.loads(row["uci_topk_json"])
        cp_topk = json.loads(row["cp_topk_json"])
        mate_topk = json.loads(row["mate_topk_json"])

        kept_ids: List[int] = []
        kept_cp: List[Optional[int]] = []
        kept_mate: List[Optional[int]] = []
        kept_moves: List[str] = []
        for u, cp, mate in zip(uci_topk, cp_topk, mate_topk):
            if u in self.move2id:
                kept_ids.append(int(self.move2id[u]))
                kept_cp.append(cp)
                kept_mate.append(mate)
                kept_moves.append(u)

        if len(kept_ids) == 0:
            return {"valid": False}

        probs = _scores_to_probs(kept_cp, kept_mate, tau=self.tau, mate_cp=self.mate_cp).cpu()

        out: Dict[str, Any] = {
            "valid": True,
            "fen": row["fen"],
            "history_ids": torch.tensor(ids, dtype=torch.long),
            "teacher_move_ids": torch.tensor(kept_ids, dtype=torch.long),
            "teacher_probs": probs.to(dtype=torch.float32),
        }
        if self.return_text:
            out["pgn_san_prefix"] = row["pgn_san_prefix"]
            out["uci_topk"] = kept_moves
        return out


class DistillationEncoderDataset(Dataset):
    """Loads records and returns teacher sparse from->to distribution (+ promo if any)."""

    PROMO_TO_ID = {"n": 0, "b": 1, "r": 2, "q": 3}

    def __init__(
        self,
        *,
        meta_path: str,
        tau: float = 50.0,
        mate_cp: int = 20000,
        return_text: bool = False,
        return_state_tensors: bool = True,
    ) -> None:
        self.index = _MultiShardIndex(meta_path)
        self.readers = [DistillationShardReader(p) for (p, _) in self.index.shards]
        self.tau = float(tau)
        self.mate_cp = int(mate_cp)
        self.return_text = bool(return_text)
        self.return_state_tensors = bool(return_state_tensors)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["readers"] = [DistillationShardReader(p) for (p, _) in self.index.shards]
        return st

    def __len__(self) -> int:
        return self.index.total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local = self.index.locate(idx)
        row = self.readers[shard_id].get_by_index(local)

        uci_topk = json.loads(row["uci_topk_json"])
        cp_topk = json.loads(row["cp_topk_json"])
        mate_topk = json.loads(row["mate_topk_json"])

        parsed: List[Tuple[int, int, Optional[str], Optional[int], Optional[int]]] = []
        for u, cp, mate in zip(uci_topk, cp_topk, mate_topk):
            try:
                f, t, promo = _uci_to_from_to_promo(u)
                parsed.append((int(f), int(t), promo, cp, mate))
            except Exception:
                continue

        if len(parsed) == 0:
            return {"valid": False}

        probs = _scores_to_probs(
            [p[3] for p in parsed],
            [p[4] for p in parsed],
            tau=self.tau,
            mate_cp=self.mate_cp,
        ).cpu()

        from_ids = torch.tensor([p[0] for p in parsed], dtype=torch.long)
        to_ids = torch.tensor([p[1] for p in parsed], dtype=torch.long)

        promo_mass = torch.zeros((4,), dtype=torch.float32)
        for pr, p in zip(probs, parsed):
            promo = p[2]
            if promo in self.PROMO_TO_ID:
                promo_mass[self.PROMO_TO_ID[promo]] += pr
        p_promo = None
        if float(promo_mass.sum()) > 0:
            p_promo = (promo_mass / promo_mass.sum()).cpu()

        out: Dict[str, Any] = {
            "valid": True,
            "fen": row["fen"],
            "teacher_from": from_ids,
            "teacher_to": to_ids,
            "teacher_probs": probs.to(dtype=torch.float32),
        }

        if self.return_state_tensors:
            fen = row["fen"]
            board = chess.Board(fen)

            # grid: (8,8) with your existing convention
            grid = board_to_grid_ids(board)
            labels64 = torch.from_numpy(grid.reshape(-1)).long()  # [64]

            # one-hot: [64,13]
            piece_probs = torch.nn.functional.one_hot(labels64, num_classes=13).float()

            # IMPORTANT: match ChessGameSampleDataset convention: 1=white, 0=black
            turn = torch.tensor(1 if board.turn else 0, dtype=torch.long) #Obs: for decoder model the convention is 0=white, 1=black, but for encoder it's 1=white, 0=black.

            ck  = int(board.has_kingside_castling_rights(chess.WHITE))
            cq  = int(board.has_queenside_castling_rights(chess.WHITE))
            ck2 = int(board.has_kingside_castling_rights(chess.BLACK))
            cq2 = int(board.has_queenside_castling_rights(chess.BLACK))
            castling = torch.tensor([ck, cq, ck2, cq2], dtype=torch.long)

            ep_square = torch.tensor(board.ep_square if board.ep_square is not None else -1, dtype=torch.long)

            out.update({
                "labels64": labels64,
                "piece_probs": piece_probs,
                "turn": turn,
                "castling": castling,
                "ep_square": ep_square,
            })
        if p_promo is not None:
            out["teacher_promo_probs"] = p_promo
        if self.return_text:
            out["pgn_san_prefix"] = row["pgn_san_prefix"]
        return out


def collate_distill_decoder(batch: List[Dict[str, Any]], *, pad_id: int) -> Dict[str, Any]:
    batch = [b for b in batch if b.get("valid", False)]
    if len(batch) == 0:
        return {"valid": False}

    B = len(batch)
    lens = [int(b["history_ids"].numel()) for b in batch]
    T = max(lens)
    hist = torch.full((B, T), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.bool)
    for i, b in enumerate(batch):
        x = b["history_ids"]
        hist[i, : x.numel()] = x
        attn[i, : x.numel()] = True

    k_lens = [int(b["teacher_move_ids"].numel()) for b in batch]
    K = max(k_lens)
    teacher_ids = torch.full((B, K), -1, dtype=torch.long)
    teacher_p = torch.zeros((B, K), dtype=torch.float32)
    teacher_mask = torch.zeros((B, K), dtype=torch.bool)
    for i, b in enumerate(batch):
        ids = b["teacher_move_ids"]
        pr = b["teacher_probs"]
        teacher_ids[i, : ids.numel()] = ids
        teacher_p[i, : pr.numel()] = pr
        teacher_mask[i, : ids.numel()] = True

    return {
        "valid": True,
        "fen": [b["fen"] for b in batch],
        "history_ids": hist,
        "history_attention": attn,
        "teacher_move_ids": teacher_ids,
        "teacher_probs": teacher_p,
        "teacher_mask": teacher_mask,
    }


def collate_distill_encoder(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b.get("valid", False)]
    if len(batch) == 0:
        return {"valid": False}

    B = len(batch)
    k_lens = [int(b["teacher_from"].numel()) for b in batch]
    K = max(k_lens)
    f = torch.full((B, K), -1, dtype=torch.long)
    t = torch.full((B, K), -1, dtype=torch.long)
    p = torch.zeros((B, K), dtype=torch.float32)
    m = torch.zeros((B, K), dtype=torch.bool)

    any_promo = any("teacher_promo_probs" in b for b in batch)
    promo = torch.zeros((B, 4), dtype=torch.float32) if any_promo else None

    for i, b in enumerate(batch):
        fi = b["teacher_from"]
        ti = b["teacher_to"]
        pi = b["teacher_probs"]
        f[i, : fi.numel()] = fi
        t[i, : ti.numel()] = ti
        p[i, : pi.numel()] = pi
        m[i, : fi.numel()] = True
        if promo is not None and "teacher_promo_probs" in b:
            promo[i] = b["teacher_promo_probs"]

    out = {
        "valid": True,
        "fen": [b["fen"] for b in batch],
        "teacher_from": f,
        "teacher_to": t,
        "teacher_probs": p,
        "teacher_mask": m,
    }

    has_state = (
        "piece_probs" in batch[0]
        and "turn" in batch[0]
        and "castling" in batch[0]
        and "ep_square" in batch[0]
    )

    if has_state:
        piece_probs = torch.stack([b["piece_probs"] for b in batch], dim=0)  # [B,64,13]
        labels64    = torch.stack([b["labels64"] for b in batch], dim=0)     # [B,64]
        turn        = torch.stack([b["turn"] for b in batch], dim=0)         # [B]
        castling    = torch.stack([b["castling"] for b in batch], dim=0)     # [B,4]
        ep_square   = torch.stack([b["ep_square"] for b in batch], dim=0)    # [B]

        out["piece_probs"] = piece_probs
        out["labels64"] = labels64
        out["turn"] = turn
        out["castling"] = castling
        out["ep_square"] = ep_square
    if promo is not None:
        out["teacher_promo_probs"] = promo
    return out