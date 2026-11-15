"""Transactional per-game persistence and streaming, reproducible CSV exports."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sqlite3

from .common import CATEGORIES

POSITION_FIELDS = ("ply", "bucket", "fen_before", "target_uci", "n_legal", "turn_white", "is_promotion")
PREDICTION_FIELDS = ("ply", "model", "predicted_uci", "target_logp", "predicted_logp",
                     "top1_correct", "top5_correct", "pair_correct", "entropy")


@contextmanager
def run_lock(directory: Path):
    """OS-held lock released even on interruption; no stale PID-file recovery."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(f"another process is using {directory}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            release()


class ResultStore:
    """A game becomes resumable only after every position and policy is saved."""

    def __init__(self, path: Path, identity: dict, *, resume: bool):
        existed = path.exists()
        if existed and not resume:
            raise FileExistsError(f"result database exists: {path}; use --resume")
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY, category TEXT NOT NULL, n_positions INTEGER NOT NULL,
                headers_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS positions (
                game_id TEXT NOT NULL REFERENCES games(game_id), ply INTEGER NOT NULL,
                bucket TEXT NOT NULL, fen_before TEXT NOT NULL, target_uci TEXT NOT NULL,
                n_legal INTEGER NOT NULL, turn_white INTEGER NOT NULL, is_promotion INTEGER NOT NULL,
                PRIMARY KEY(game_id, ply));
            CREATE TABLE IF NOT EXISTS predictions (
                game_id TEXT NOT NULL, ply INTEGER NOT NULL, model TEXT NOT NULL,
                predicted_uci TEXT NOT NULL, target_logp REAL NOT NULL, predicted_logp REAL NOT NULL,
                top1_correct INTEGER NOT NULL, top5_correct INTEGER NOT NULL,
                pair_correct INTEGER NOT NULL, entropy REAL NOT NULL,
                PRIMARY KEY(game_id, ply, model),
                FOREIGN KEY(game_id, ply) REFERENCES positions(game_id, ply));
        """)
        encoded = json.dumps(identity, sort_keys=True)
        previous = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if previous and previous[0] != encoded:
            self.connection.close()
            raise ValueError("evaluation identity changed (sample, checkpoints, inference code/settings or runtime); use a new output directory")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (encoded,))

    def close(self):
        self.connection.close()

    def completed_ids(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT game_id FROM games")}

    def save_game(self, game: dict, positions: list[dict], predictions: list[dict]):
        n = game["game_plies"]
        if len(positions) != n or {p["ply"] for p in positions} != set(range(n)):
            raise ValueError("incomplete game positions")
        keys = {(p["ply"], p["model"]) for p in predictions}
        if len(predictions) != n * 3 or keys != {(i, m) for i in range(n) for m in CATEGORIES}:
            raise ValueError("incomplete or duplicate game predictions")
        gid = game["game_id"]
        with self.connection:
            self.connection.execute("INSERT INTO games VALUES (?, ?, ?, ?)",
                                    (gid, game["category"], n, json.dumps(game["headers"], sort_keys=True)))
            self.connection.executemany("INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                        [(gid, *(p[k] for k in POSITION_FIELDS)) for p in positions])
            self.connection.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                        [(gid, *(p[k] for k in PREDICTION_FIELDS)) for p in predictions])

    def export_csv(self, directory: Path):
        query = """
            SELECT g.game_id, g.category AS test_category, p.ply, g.n_positions AS game_plies,
                   p.bucket, p.fen_before, p.target_uci, p.n_legal, p.turn_white, p.is_promotion,
                   r.model, r.predicted_uci, r.target_logp, -r.target_logp AS nll_legal,
                   r.predicted_logp, r.top1_correct, r.top5_correct, r.pair_correct, r.entropy
            FROM games g JOIN positions p USING(game_id) JOIN predictions r USING(game_id, ply)
            ORDER BY g.category, g.game_id, p.ply, r.model
        """
        temporary = directory / "per_move.csv.tmp"
        cursor = self.connection.execute(query)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([col[0] for col in cursor.description])
            writer.writerows(cursor)
        temporary.replace(directory / "per_move.csv")

    def validate_complete(self, games: list[dict]):
        expected = {g["game_id"]: g["game_plies"] for g in games}
        actual = {r["game_id"]: r["n_positions"] for r in self.connection.execute("SELECT * FROM games")}
        if expected != actual:
            raise ValueError(f"evaluation incomplete: {len(actual)}/{len(expected)} games saved")
        position_counts = {row[0]: row[1] for row in self.connection.execute(
            "SELECT game_id, COUNT(*) FROM positions GROUP BY game_id")}
        if position_counts != expected:
            raise ValueError("database position counts differ from the frozen games")
        bad = self.connection.execute("""
            SELECT p.game_id, p.ply FROM positions p LEFT JOIN predictions r USING(game_id, ply)
            GROUP BY p.game_id, p.ply HAVING COUNT(r.model) != 3
        """).fetchone()
        if bad:
            raise ValueError("database has an incomplete policy comparison")
