#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def eprint(*a, **k):  # stderr helper
    print(*a, file=sys.stderr, **k)


# ----------------------------
# Path normalization
# ----------------------------

def to_workspace_relative(path_str: str, workspace_root: Path) -> str:
    """
    Convert any absolute path into a path relative to workspace_root if possible.
    Otherwise, try stripping the known marker ".../multimodal/project_multimodal_gen_ai/".
    Otherwise, return as-is.
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    # If it's already relative, keep it.
    if not p.is_absolute():
        return path_str.replace("\\", "/")

    # Try to relativize against workspace_root
    try:
        rel = p.resolve().relative_to(workspace_root.resolve())
        return rel.as_posix()
    except Exception:
        pass

    # Fallback: strip common marker
    s = path_str.replace("\\", "/")
    marker = "/multimodal/project_multimodal_gen_ai/"
    if marker in s:
        return s.split(marker, 1)[1].lstrip("/")

    marker2 = "/project_multimodal_gen_ai/"
    if marker2 in s:
        return s.split(marker2, 1)[1].lstrip("/")

    return s


def extract_suffix_int(name: str) -> Optional[int]:
    m = re.search(r"(\d+)(?!.*\d)", name)
    return int(m.group(1)) if m else None


def atomic_write_json(path: Path, obj: dict, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


@dataclass
class RunBundle:
    tag: int
    distill_dir: Path
    pgn: Path
    index_json: Path
    samples_json: Path
    shard_paths: List[Path]


def discover_runs(workspace_root: Path, db_glob: str, processed_dir_name: str) -> List[RunBundle]:
    runs: List[RunBundle] = []
    proc_dir = workspace_root / processed_dir_name

    for d in sorted(workspace_root.glob(db_glob)):
        if not d.is_dir():
            continue
        tag = extract_suffix_int(d.name)
        if tag is None:
            continue

        shard_paths = sorted(d.glob("shard_*.sqlite"))
        if not shard_paths:
            continue

        pgn = proc_dir / f"games_stockfish{tag}.pgn"
        idx = proc_dir / f"games_stockfish{tag}_index.json"
        smp = proc_dir / f"games_stockfish{tag}_samples.json"

        missing = [p for p in [pgn, idx, smp] if not p.exists()]
        if missing:
            eprint(f"[WARN] distill_db{tag}: missing processed files: {missing}. Skipping.")
            continue

        runs.append(RunBundle(tag=tag, distill_dir=d, pgn=pgn, index_json=idx, samples_json=smp, shard_paths=shard_paths))

    runs.sort(key=lambda r: r.tag)
    return runs


# ----------------------------
# PGN + Index merge
# ----------------------------

def merge_pgn_and_index(
    runs: List[RunBundle],
    out_pgn: Path,
    out_index_json: Path,
    workspace_root: Path,
) -> Tuple[int, Dict[int, Tuple[int, int]]]:
    """
    Concatenate all PGNs (binary) and shift byte offsets in index.json accordingly.
    Also reindex game idx so it's contiguous.

    Returns:
      total_num_games,
      per_tag_offsets: {tag: (base_byte_offset, base_game_idx_offset)}
    """
    out_pgn.parent.mkdir(parents=True, exist_ok=True)

    merged_games: List[dict] = []
    per_tag_offsets: Dict[int, Tuple[int, int]] = {}

    byte_cursor = 0
    game_cursor = 0

    with out_pgn.open("wb") as w:
        for run in runs:
            per_tag_offsets[run.tag] = (byte_cursor, game_cursor)

            data = run.pgn.read_bytes()
            w.write(data)
            base_byte = byte_cursor
            byte_cursor += len(data)

            idx_obj = json.loads(run.index_json.read_text(encoding="utf-8"))
            num_games = int(idx_obj["num_games"])
            games = idx_obj["games"]
            if len(games) != num_games:
                raise RuntimeError(f"Index mismatch tag={run.tag}: num_games={num_games} len(games)={len(games)}")

            for g in games:
                merged_games.append(
                    {
                        "idx": int(g["idx"]) + game_cursor,
                        "start": int(g["start"]) + base_byte,
                        "end": int(g["end"]) + base_byte,
                    }
                )

            game_cursor += num_games

    # Use actual output file stats
    st = out_pgn.stat()
    merged_index = {
        "pgn_path": to_workspace_relative(str(out_pgn), workspace_root),
        "pgn_size": int(st.st_size),
        "pgn_mtime": float(st.st_mtime),
        "num_games": len(merged_games),
        "games": merged_games,
    }
    atomic_write_json(out_index_json, merged_index, indent=2)

    return len(merged_games), per_tag_offsets


# ----------------------------
# samples.json merge
# ----------------------------

def merge_samples_json(
    runs: List[RunBundle],
    out_samples_json: Path,
    out_pgn: Path,
    out_index_json: Path,
    per_tag_offsets: Dict[int, Tuple[int, int]],
    workspace_root: Path,
) -> int:
    merged_samples: List[List[int]] = []
    ref_meta: Optional[dict] = None

    for run in runs:
        obj = json.loads(run.samples_json.read_text(encoding="utf-8"))

        if ref_meta is None:
            ref_meta = {k: obj.get(k) for k in ["sample_ratio", "max_positions_per_game", "seed"]}
        else:
            for k in ["sample_ratio", "max_positions_per_game", "seed"]:
                if obj.get(k) != ref_meta.get(k):
                    eprint(f"[WARN] samples meta differs tag={run.tag} key={k}: {obj.get(k)} != {ref_meta.get(k)} (keeping first)")

        _, base_game = per_tag_offsets[run.tag]
        for pair in obj["samples"]:
            gi = int(pair[0]) + base_game
            pi = int(pair[1])
            merged_samples.append([gi, pi])

    assert ref_meta is not None

    merged_obj = {
        "pgn_path": to_workspace_relative(str(out_pgn), workspace_root),
        "index_path": to_workspace_relative(str(out_index_json), workspace_root),
        "sample_ratio": ref_meta["sample_ratio"],
        "max_positions_per_game": ref_meta["max_positions_per_game"],
        "seed": ref_meta["seed"],
        "num_samples": len(merged_samples),
        "samples": merged_samples,
    }
    atomic_write_json(out_samples_json, merged_obj, indent=2)
    return len(merged_samples)


# ----------------------------
# SQLite merge
# ----------------------------

def get_samples_table_schema(db_path: Path) -> str:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='samples'"
        ).fetchone()
        if not row or not row[0]:
            raise RuntimeError(f"No 'samples' table found in {db_path}")
        return row[0]
    finally:
        con.close()


def get_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    if not cols:
        raise RuntimeError(f"Could not read columns for table '{table}'")
    return cols


def merge_sqlite_shards(
    runs: List[RunBundle],
    out_sqlite: Path,
    batch_size: int = 5000,
) -> int:
    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if out_sqlite.exists():
        out_sqlite.unlink()

    schema_sql = get_samples_table_schema(runs[0].shard_paths[0])

    out_con = sqlite3.connect(str(out_sqlite))
    try:
        out_con.execute("PRAGMA journal_mode=WAL;")
        out_con.execute("PRAGMA synchronous=NORMAL;")
        out_con.execute(schema_sql)

        out_cols = get_table_columns(out_con, "samples")
        if "id" not in out_cols:
            raise RuntimeError("Expected 'id' column in samples table")
        insert_cols = [c for c in out_cols if c != "id"]

        placeholders = ",".join(["?"] * len(insert_cols))
        insert_sql = f"INSERT INTO samples ({','.join(insert_cols)}) VALUES ({placeholders})"

        total_written = 0
        out_cur = out_con.cursor()

        for run in runs:
            for shard in run.shard_paths:
                src_con = sqlite3.connect(str(shard))
                try:
                    src_cols = get_table_columns(src_con, "samples")
                    if src_cols != out_cols:
                        raise RuntimeError(
                            f"Schema mismatch in {shard}\nsrc_cols={src_cols}\nout_cols={out_cols}"
                        )

                    sel = f"SELECT {','.join(insert_cols)} FROM samples ORDER BY id"
                    cur = src_con.cursor()
                    cur.execute(sel)

                    while True:
                        rows = cur.fetchmany(batch_size)
                        if not rows:
                            break
                        out_cur.executemany(insert_sql, rows)
                        total_written += len(rows)

                    out_con.commit()
                    eprint(f"[OK] merged {shard} (cumulative rows={total_written})")
                finally:
                    src_con.close()

        out_count = out_con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        if out_count != total_written:
            eprint(f"[WARN] count mismatch: inserted={total_written}, count(*)={out_count}")
        return int(out_count)
    finally:
        out_con.close()


# ----------------------------
# meta.json rebuild
# ----------------------------

def read_first_meta(runs: List[RunBundle]) -> Optional[dict]:
    for run in runs:
        m = run.distill_dir / "meta.json"
        if m.exists():
            return json.loads(m.read_text(encoding="utf-8"))
    return None


def build_merged_meta(
    out_meta: Path,
    out_pgn: Path,
    out_index_json: Path,
    out_samples_json: Path,
    out_sqlite: Path,
    total_sql_rows: int,
    engine_fallback: dict,
    workspace_root: Path,
) -> None:
    meta = {
        "format": "sqlite_shards_v1",
        "pgn_path": to_workspace_relative(str(out_pgn), workspace_root),
        "index_path": to_workspace_relative(str(out_index_json), workspace_root),
        "samples_path": to_workspace_relative(str(out_samples_json), workspace_root),
        "engine": engine_fallback,
        "num_shards": 1,
        "shards": [
            {"path": to_workspace_relative(str(out_sqlite), workspace_root), "count": int(total_sql_rows)}
        ],
        "build_results": [
            {"written": int(total_sql_rows), "skipped": 0, "shard_path": to_workspace_relative(str(out_sqlite), workspace_root)}
        ],
    }
    atomic_write_json(out_meta, meta, indent=2)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workspace-root",
        required=True,
        help="Folder that contains distill_dbXX/ and processed_games/ (no matter what abs paths are in JSONs).",
    )
    ap.add_argument("--db-glob", default="distill_db*", help="Glob for distill DB folders (default distill_db*)")
    ap.add_argument("--processed-dir", default="processed_games", help="Processed games dir name under workspace-root")
    ap.add_argument("--out-dir", default="merged_distill", help="Output dir name under workspace-root")
    args = ap.parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    out_dir = (workspace_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(workspace_root, args.db_glob, args.processed_dir)
    if not runs:
        raise SystemExit("No runs found. Check --workspace-root, --db-glob, and that processed_games has the files.")

    eprint(f"[INFO] Found {len(runs)} runs: {[r.tag for r in runs]}")

    out_pgn = out_dir / "games_stockfish_merged.pgn"
    out_index = out_dir / "games_stockfish_merged_index.json"
    out_samples = out_dir / "games_stockfish_merged_samples.json"
    out_sqlite = out_dir / "merged.sqlite"
    out_meta = out_dir / "meta.json"

    total_games, per_tag_offsets = merge_pgn_and_index(runs, out_pgn, out_index, workspace_root)
    eprint(f"[OK] merged PGN+index: total_games={total_games}")

    total_samples = merge_samples_json(runs, out_samples, out_pgn, out_index, per_tag_offsets, workspace_root)
    eprint(f"[OK] merged samples.json: total_samples={total_samples}")

    total_rows = merge_sqlite_shards(runs, out_sqlite)
    eprint(f"[OK] merged sqlite: total_rows={total_rows}")

    meta0 = read_first_meta(runs)
    engine = meta0.get("engine") if isinstance(meta0, dict) and isinstance(meta0.get("engine"), dict) else {"movetime_ms": 500, "k": 5, "min_depth": 1}

    build_merged_meta(out_meta, out_pgn, out_index, out_samples, out_sqlite, total_rows, engine, workspace_root)
    eprint(f"[OK] wrote merged meta.json -> {out_meta}")

    print("\nDONE ✅")
    print("Merged outputs (workspace-relative paths used inside JSON):")
    print(f"  {to_workspace_relative(str(out_pgn), workspace_root)}")
    print(f"  {to_workspace_relative(str(out_index), workspace_root)}")
    print(f"  {to_workspace_relative(str(out_samples), workspace_root)}")
    print(f"  {to_workspace_relative(str(out_sqlite), workspace_root)}")
    print(f"  {to_workspace_relative(str(out_meta), workspace_root)}")


if __name__ == "__main__":
    main()