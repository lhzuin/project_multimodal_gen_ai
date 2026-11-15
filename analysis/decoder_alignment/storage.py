"""Game-atomic storage for immutable baseline imports and new predictions."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3


BASE_FIELDS = (
    "ply", "game_plies", "bucket", "relative_bucket", "twic_issue", "date",
    "white_elo", "black_elo", "fen_before", "target_uci", "n_legal", "context_len",
    "context_truncated", "predicted_uci", "top1_correct", "top5_correct", "nll_legal",
)
CANDIDATE_FIELDS = ("ply", "predicted_uci", "top1_correct", "top5_correct", "nll_legal")


@contextmanager
def run_lock(directory: Path):
    """Hold a portable OS lock; interruption releases it without stale PID files."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
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


class Store:
    """A committed game always has a complete baseline or candidate sequence."""

    def __init__(self, path: Path, identity: dict, *, resume: bool):
        if path.exists() and not resume:
            raise FileExistsError(f"{path} exists; use --resume")
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY, n_positions INTEGER NOT NULL,
                candidate_complete INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS baseline (
                game_id INTEGER NOT NULL REFERENCES games(game_id), ply INTEGER NOT NULL,
                game_plies INTEGER NOT NULL, bucket TEXT NOT NULL, relative_bucket TEXT NOT NULL,
                twic_issue INTEGER NOT NULL, date TEXT NOT NULL, white_elo INTEGER NOT NULL,
                black_elo INTEGER NOT NULL, fen_before TEXT NOT NULL, target_uci TEXT NOT NULL,
                n_legal INTEGER NOT NULL, context_len INTEGER NOT NULL, context_truncated INTEGER NOT NULL,
                predicted_uci TEXT NOT NULL, top1_correct INTEGER NOT NULL, top5_correct INTEGER NOT NULL,
                nll_legal REAL NOT NULL, PRIMARY KEY(game_id, ply));
            CREATE TABLE IF NOT EXISTS candidate (
                game_id INTEGER NOT NULL, ply INTEGER NOT NULL, predicted_uci TEXT NOT NULL,
                top1_correct INTEGER NOT NULL, top5_correct INTEGER NOT NULL, nll_legal REAL NOT NULL,
                PRIMARY KEY(game_id, ply), FOREIGN KEY(game_id, ply) REFERENCES baseline(game_id, ply));
        """)
        try:
            self.pin("identity", identity)
        except BaseException:
            self.close()
            raise

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                            (key, json.dumps(value, sort_keys=True)))

    def pin(self, key, value):
        old = self.get(key)
        if old is not None and old != value:
            raise ValueError(f"{key} changed: inputs, inference code/runtime, or settings differ; use a new output directory")
        self.set(key, value)

    def close(self):
        self.db.close()

    def imported(self):
        return {r[0] for r in self.db.execute("SELECT game_id FROM games")}

    def completed(self):
        return {r[0] for r in self.db.execute("SELECT game_id FROM games WHERE candidate_complete=1")}

    def baseline_rows(self, game_id):
        return list(self.db.execute("SELECT * FROM baseline WHERE game_id=? ORDER BY ply", (game_id,)))

    def save_baseline(self, game_id, rows):
        if not rows or {r["ply"] for r in rows} != set(range(len(rows))) or len(rows) != rows[0]["game_plies"]:
            raise ValueError(f"incomplete or duplicate baseline sequence: game {game_id}")
        with self.db:
            self.db.execute("INSERT INTO games(game_id,n_positions) VALUES (?, ?)", (game_id, len(rows)))
            self.db.executemany("INSERT INTO baseline VALUES (" + ",".join("?" for _ in range(18)) + ")",
                                [(game_id, *(r[k] for k in BASE_FIELDS)) for r in rows])

    def save_candidate(self, game_id, rows):
        entry = self.db.execute("SELECT n_positions,candidate_complete FROM games WHERE game_id=?", (game_id,)).fetchone()
        if not entry or entry[1] or len(rows) != entry[0] or {r["ply"] for r in rows} != set(range(entry[0])):
            raise ValueError(f"incomplete, duplicate, or already saved candidate: game {game_id}")
        with self.db:
            self.db.executemany("INSERT INTO candidate VALUES (?, ?, ?, ?, ?, ?)",
                                [(game_id, *(r[k] for k in CANDIDATE_FIELDS)) for r in rows])
            self.db.execute("UPDATE games SET candidate_complete=1 WHERE game_id=?", (game_id,))

    def validate_complete(self, expected):
        if self.imported() != expected or self.completed() != expected:
            raise ValueError("evaluation is incomplete; resume before summarizing")
        bad = self.db.execute("""
            SELECT g.game_id FROM games g LEFT JOIN candidate c USING(game_id)
            GROUP BY g.game_id HAVING COUNT(c.ply) != g.n_positions
        """).fetchone()
        if bad:
            raise ValueError(f"incomplete candidate rows: {bad[0]}")
