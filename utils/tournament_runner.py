
#!/usr/bin/env python3
"""
Tournament runner: model (decoder or encoder) vs Stockfish, starting from an opening suite JSON.

- Sample a percentage of openings (reproducible seed)
- For each opening, play paired games (swap colors)
- Configure Stockfish (threads/hash/skill/limit strength/elo) + search limits
- Run your model through your existing UCI wrapper (uci_engine.py) so it's plug-and-play
- Aggregate score: (wins + 0.5*draws) / N_games
- Optional PGN + JSON stats output

Dependencies
  pip install chess

Example
  python tournament_runner.py \
    --openings_json opening_suite.json --pct 0.2 --seed 0 \
    --stockfish /path/to/stockfish \
    --movetime_ms 100 --threads 1 --hash_mb 256 \
    --model_type decoder --ckpt /path/to/chess_llm_decoder_v9.pt \
    --tokenizer_path /path/to/chess_uci_vocab.json \
    --out_dir out_tournament
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import shutil
import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn



device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ----------------------------
# Utilities
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(d: str) -> None:
    if d:
        os.makedirs(d, exist_ok=True)

def _pick_fen(opening_entry: Dict[str, Any]) -> str:
    # robust to different suite schemas
    for k in ("fen", "fen_after_prefix", "start_fen", "fen_after"):
        if k in opening_entry and isinstance(opening_entry[k], str) and opening_entry[k].strip():
            return opening_entry[k].strip()
    raise KeyError(f"Could not find a FEN field in opening entry keys={list(opening_entry.keys())}")



def _pick_prefix(opening_entry: Dict[str, Any]) -> List[str]:
    """Return opening UCI prefix (list of UCI moves) if present, else empty list."""
    pref = opening_entry.get("uci_prefix")
    if isinstance(pref, list) and all(isinstance(x, str) for x in pref):
        return [x.strip() for x in pref if x and x.strip()]
    return []
def _opening_id(opening_entry: Dict[str, Any], idx: int) -> str:
    for k in ("id", "key", "opening_id"):
        if k in opening_entry and isinstance(opening_entry[k], str) and opening_entry[k].strip():
            return opening_entry[k].strip()
    return f"opening_{idx:05d}"

def _limit_from_args(args) -> chess.engine.Limit:
    # Prefer explicit depth/nodes if provided, else movetime_ms.
    # Stockfish will respect all; your model UCI wrapper currently supports movetime.
    time_s = None
    if args.movetime_ms is not None and args.movetime_ms > 0:
        time_s = float(args.movetime_ms) / 1000.0

    return chess.engine.Limit(
        time=time_s,
        depth=args.depth if args.depth and args.depth > 0 else None,
        nodes=args.nodes if args.nodes and args.nodes > 0 else None,
    )

def _score_from_result(result: str, model_is_white: bool) -> float:
    """
    Map PGN result to model score in {1, 0.5, 0}.
    result is "1-0", "0-1", "1/2-1/2", or "*" (unknown).
    """
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if model_is_white else 0.0
    if result == "0-1":
        return 0.0 if model_is_white else 1.0
    # unknown => treat as draw-ish
    return 0.5


# ----------------------------
# Engine launchers
# ----------------------------

def launch_stockfish(stockfish_path: str, threads: int, hash_mb: int,
                    skill: Optional[int], limit_strength: bool, uci_elo: Optional[int]) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Best-effort configuration (ignore unknown options gracefully)
    opts = {}
    if threads is not None:
        opts["Threads"] = int(threads)
    if hash_mb is not None:
        opts["Hash"] = int(hash_mb)

    # Skill Level is usually 0..20 on Stockfish
    if skill is not None:
        opts["Skill Level"] = int(skill)

    # LimitStrength + UCI_Elo exist on many Stockfish builds
    opts["UCI_LimitStrength"] = bool(limit_strength)
    if uci_elo is not None:
        opts["UCI_Elo"] = int(uci_elo)

    if opts:
        try:
            eng.configure(opts)
        except Exception:
            # Some builds reject some keys; try individually
            for k, v in list(opts.items()):
                try:
                    eng.configure({k: v})
                except Exception:
                    pass

    return eng


def launch_model_uci_engine(args) -> chess.engine.SimpleEngine:
    """
    Launch your model through the provided UCI wrapper (python script) as a UCI engine subprocess.

    We set:
      - cwd to repo root (parent of utils/)
      - PYTHONPATH to repo root so `import models.*` works in the subprocess

    Note: This is for the model UCI engine (not Stockfish).
    """
    model_uci_abs = os.path.abspath(args.model_uci_py)
    if not os.path.exists(model_uci_abs):
        raise FileNotFoundError(f"--model_uci_py not found: {model_uci_abs}")

    utils_dir = os.path.dirname(model_uci_abs)
    repo_root = os.path.dirname(utils_dir)
    cmd = [args.python, "-u", model_uci_abs,
           "--model_type", args.model_type,
           "--ckpt", args.ckpt,
           "--device", device,
           "--temperature", str(args.temperature),
           "--topk", str(args.topk)]
    if args.greedy:
        cmd.append("--greedy")
    
    if args.model_type == "encoder":
        cmd += ["--version", str(args.version)]

    if args.model_type == "decoder":
        if not args.tokenizer_path:
            raise SystemExit("--tokenizer_path is required for decoder model_type.")
        cmd += ["--tokenizer_path", args.tokenizer_path,
                "--max_seq_len", str(args.max_seq_len),
                "--n_layers", str(args.n_layers)]
    else:
        cmd += ["--img_size", str(args.img_size)]
        if args.vit_path:
            cmd += ["--vit_path", args.vit_path]

    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if args.verbose:
        print("[tournament] launching model UCI:", " ".join(cmd), flush=True)
        print("[tournament] cwd:", repo_root, flush=True)

    eng = chess.engine.SimpleEngine.popen_uci(cmd, timeout=float(180.0), cwd=repo_root, env=env)
    return eng



# ----------------------------
# Game runner
# ----------------------------

@dataclass
class PlayedGame:
    opening_id: str
    start_fen: str                 # effective start FEN (after prefix, if any)
    uci_prefix: List[str]          # opening moves played before start_fen (for decoder history)
    model_color: str               # "white" or "black"
    result: str                    # "1-0", "0-1", "1/2-1/2", "*"
    termination: str
    plies: int
    moves_uci: List[str]


""" def play_single_game(
    *,
    opening_id: str,
    opening_entry: Dict[str, Any],
    model_eng: chess.engine.SimpleEngine,
    opp_eng: chess.engine.SimpleEngine,
    model_is_white: bool,
    limit: chess.engine.Limit,
    max_plies: int,
) -> PlayedGame:

    Plays one game starting from an opening entry.

    IMPORTANT for decoder evaluation:
    - If `opening_entry["uci_prefix"]` is present, we *build the board from startpos* and push that prefix.
      This ensures python-chess will send `position startpos moves ...` to both engines, so the decoder
      sees the full opening history.
    - The "effective start FEN" is then the FEN after the prefix (typically opening_entry["fen_after_prefix"]).
      We store that in the result, and `moves_uci` contains only moves played AFTER the prefix.
    
    uci_prefix = _pick_prefix(opening_entry)

    # Build board WITH HISTORY for engine communication
    if uci_prefix:
        board = chess.Board()  # startpos
        for u in uci_prefix:
            mv = chess.Move.from_uci(u)
            if mv not in board.legal_moves:
                print(f"Warning: Illegal move {u} in opening prefix for opening_id={opening_id}. Ignoring the rest of the prefix.", file=sys.stderr)
                # ValueError(f"Illegal uci_prefix move {u} for opening_id={opening_id}")
            board.push(mv)
        effective_start_fen = board.fen()
    else:
        effective_start_fen = _pick_fen(opening_entry)
        board = chess.Board(effective_start_fen)

    moves_uci: List[str] = []
    termination = "normal"

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = "game_over"
            break

        to_move_is_white = board.turn == chess.WHITE
        engine = model_eng if (to_move_is_white == model_is_white) else opp_eng

        try:
            res = engine.play(board, limit)
        except chess.engine.EngineTerminatedError:
            termination = "engine_terminated"
            if engine is model_eng:
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  "0-1" if model_is_white else "1-0",
                                  termination, board.ply(), moves_uci)
            else:
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  "1-0" if model_is_white else "0-1",
                                  termination, board.ply(), moves_uci)
        except Exception:
            termination = "engine_error"
            if engine is model_eng:
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  "0-1" if model_is_white else "1-0",
                                  termination, board.ply(), moves_uci)
            else:
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  "1-0" if model_is_white else "0-1",
                                  termination, board.ply(), moves_uci)

        mv = res.move
        if mv is None or mv not in board.legal_moves:
            termination = "illegal_move"
            if to_move_is_white == model_is_white:
                result = "0-1" if model_is_white else "1-0"
            else:
                result = "1-0" if model_is_white else "0-1"
            return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                              "white" if model_is_white else "black",
                              result, termination, board.ply(), moves_uci)

        board.push(mv)
        moves_uci.append(mv.uci())

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    if result == "*" and termination == "normal":
        termination = "max_plies"

    return PlayedGame(
        opening_id=opening_id,
        start_fen=effective_start_fen,
        uci_prefix=uci_prefix,
        model_color="white" if model_is_white else "black",
        result=result,
        termination=termination,
        plies=board.ply(),
        moves_uci=moves_uci,
    )
 """



def play_single_game(
    *,
    opening_id: str,
    opening_entry: Dict[str, Any],
    model_eng: chess.engine.SimpleEngine,
    opp_eng: chess.engine.SimpleEngine,
    model_is_white: bool,
    limit: chess.engine.Limit,
    max_plies: int,
) -> PlayedGame:
    """
    Plays one game starting from an opening entry.

    Robustness goals:
    - Never raise: always return a PlayedGame so tournaments keep going.
    - Defensive opening setup (prefix/FEN) with clear termination codes.
    - Stop prefix application at the first illegal/unparseable move.
    """

    # Defaults so we can always return something even if setup fails early
    uci_prefix: List[str] = []
    moves_uci: List[str] = []
    termination: str = "normal"
    effective_start_fen: str = chess.STARTING_FEN  # safe fallback
    board: Optional[chess.Board] = None

    def _return_setup_loss(term: str) -> PlayedGame:
        # If we fail before a real game starts, treat as loss for "model side"
        # (consistent with your engine_error handling).
        result = "0-1" if model_is_white else "1-0"
        plies = board.ply() if board is not None else 0
        return PlayedGame(
            opening_id=opening_id,
            start_fen=effective_start_fen,
            uci_prefix=uci_prefix,
            model_color="white" if model_is_white else "black",
            result=result,
            termination=term,
            plies=plies,
            moves_uci=moves_uci,
        )

    try:
        # -------- Opening setup (prefix or FEN) --------
        try:
            uci_prefix = _pick_prefix(opening_entry) or []
        except Exception as e:
            print(
                f"[{opening_id}] Warning: _pick_prefix failed ({type(e).__name__}: {e}). Using no prefix.",
                file=sys.stderr,
            )
            uci_prefix = []

        if uci_prefix:
            board = chess.Board()  # startpos
            for u in uci_prefix:
                try:
                    mv = chess.Move.from_uci(u)
                except ValueError:
                    print(
                        f"[{opening_id}] Warning: Bad UCI '{u}' in prefix. Stopping prefix here.",
                        file=sys.stderr,
                    )
                    termination = "bad_prefix_uci"
                    break

                if mv not in board.legal_moves:
                    print(
                        f"[{opening_id}] Warning: Illegal prefix move '{u}'. Stopping prefix here.",
                        file=sys.stderr,
                    )
                    termination = "illegal_prefix_move"
                    break

                board.push(mv)

            effective_start_fen = board.fen()

            # If prefix was problematic, we still continue the game from the last valid position.
            # (If you prefer: immediately return a setup loss instead, call _return_setup_loss(termination).)

        else:
            try:
                effective_start_fen = _pick_fen(opening_entry)
                board = chess.Board(effective_start_fen)
            except Exception as e:
                print(
                    f"[{opening_id}] Error: Failed to build board from FEN "
                    f"({type(e).__name__}: {e}). Fallback to startpos and mark setup error.",
                    file=sys.stderr,
                )
                board = chess.Board()
                effective_start_fen = board.fen()
                return _return_setup_loss("bad_start_fen")

        # If something went wrong during prefix application and board is None (shouldn’t happen), fail safely
        if board is None:
            return _return_setup_loss("setup_error")

        # -------- Main play loop --------
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                termination = "game_over"
                break

            to_move_is_white = board.turn == chess.WHITE
            engine = model_eng if (to_move_is_white == model_is_white) else opp_eng

            try:
                res = engine.play(board, limit)
            except chess.engine.EngineTerminatedError:
                termination = "engine_terminated"
                # engine died -> side-to-move loses (same logic you had)
                if engine is model_eng:
                    result = "0-1" if model_is_white else "1-0"
                else:
                    result = "1-0" if model_is_white else "0-1"
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  result, termination, board.ply(), moves_uci)
            except chess.engine.EngineError as e:
                termination = "engine_error"
                # treat like engine failure -> side-to-move loses
                if engine is model_eng:
                    result = "0-1" if model_is_white else "1-0"
                else:
                    result = "1-0" if model_is_white else "0-1"
                print(f"[{opening_id}] EngineError: {e}", file=sys.stderr)
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  result, termination, board.ply(), moves_uci)
            except Exception as e:
                termination = "engine_exception"
                if engine is model_eng:
                    result = "0-1" if model_is_white else "1-0"
                else:
                    result = "1-0" if model_is_white else "0-1"
                print(f"[{opening_id}] Unexpected engine exception: {type(e).__name__}: {e}", file=sys.stderr)
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  result, termination, board.ply(), moves_uci)

            mv = getattr(res, "move", None)
            if mv is None or mv not in board.legal_moves:
                termination = "illegal_move"
                # illegal move => side-to-move loses
                if to_move_is_white == model_is_white:
                    result = "0-1" if model_is_white else "1-0"
                else:
                    result = "1-0" if model_is_white else "0-1"
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  result, termination, board.ply(), moves_uci)

            # board.push itself can (rarely) throw if something is inconsistent; guard it
            try:
                board.push(mv)
            except Exception as e:
                termination = "push_exception"
                # treat as illegal move by side-to-move
                if to_move_is_white == model_is_white:
                    result = "0-1" if model_is_white else "1-0"
                else:
                    result = "1-0" if model_is_white else "0-1"
                print(f"[{opening_id}] push() failed: {type(e).__name__}: {e}", file=sys.stderr)
                return PlayedGame(opening_id, effective_start_fen, uci_prefix,
                                  "white" if model_is_white else "black",
                                  result, termination, board.ply(), moves_uci)

            moves_uci.append(mv.uci())

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        if result == "*" and termination == "normal":
            termination = "max_plies"

        return PlayedGame(
            opening_id=opening_id,
            start_fen=effective_start_fen,
            uci_prefix=uci_prefix,
            model_color="white" if model_is_white else "black",
            result=result,
            termination=termination,
            plies=board.ply(),
            moves_uci=moves_uci,
        )

    except Exception as e:
        # Absolute last-resort: never crash tournament
        print(f"[{opening_id}] FATAL in play_single_game: {type(e).__name__}: {e}", file=sys.stderr)
        return _return_setup_loss("fatal_exception")

def to_pgn(game: PlayedGame, model_name: str, opp_name: str) -> str:
    g = chess.pgn.Game()
    g.headers["Event"] = "Model vs Engine Tournament"
    g.headers["Site"] = "local"
    g.headers["Date"] = time.strftime("%Y.%m.%d")
    g.headers["Round"] = game.opening_id
    if game.model_color == "white":
        g.headers["White"] = model_name
        g.headers["Black"] = opp_name
    else:
        g.headers["White"] = opp_name
        g.headers["Black"] = model_name
    g.headers["Result"] = game.result
    g.headers["FEN"] = game.start_fen
    g.headers["SetUp"] = "1"
    g.headers["OpeningID"] = game.opening_id
    if game.uci_prefix:
        g.headers["OpeningPrefix"] = " ".join(game.uci_prefix[:32])
    g.headers["Termination"] = game.termination

    board = chess.Board(game.start_fen)
    node = g
    for u in game.moves_uci:
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            break
        board.push(mv)
        node = node.add_variation(mv)
    return str(g)


# ----------------------------
# Tournament loop
# ----------------------------

@dataclass
class TournamentSummary:
    n_openings_total: int
    n_openings_sampled: int
    n_games: int
    wins: int
    draws: int
    losses: int
    score: float
    score_formula: str
    seed: int
    pct: float
    limits: Dict[str, Any]
    stockfish_config: Dict[str, Any]
    model_config: Dict[str, Any]


def run_tournament(args) -> Tuple[TournamentSummary, List[PlayedGame]]:
    suite = _load_json(args.openings_json)
    if not isinstance(suite, list) or not suite:
        raise SystemExit("Opening suite JSON must be a non-empty list.")

    rng = random.Random(args.seed)
    pct = _clamp(float(args.pct), 0.0, 1.0)
    n_total = len(suite)
    n_sample = max(1, int(math.ceil(pct * n_total)))

    indices = list(range(n_total))
    rng.shuffle(indices)
    sel_idx = indices[:n_sample]
    selected = [(i, suite[i]) for i in sel_idx]

    # Engines
    limit = _limit_from_args(args)
    stockfish = launch_stockfish(
        args.stockfish,
        threads=args.threads,
        hash_mb=args.hash_mb,
        skill=args.skill_level,
        limit_strength=args.limit_strength,
        uci_elo=args.uci_elo,
    )
    model_eng = launch_model_uci_engine(args)

    played: List[PlayedGame] = []

    # Stats
    wins = draws = losses = 0

    try:
        for j, (idx, entry) in enumerate(selected):
            opening_id = _opening_id(entry, idx)
            start_fen = _pick_fen(entry)

            # paired games
            for model_is_white in (True, False):
                g = play_single_game(
                    opening_id=opening_id,
                    opening_entry=entry,
                    model_eng=model_eng,
                    opp_eng=stockfish,
                    model_is_white=model_is_white,
                    limit=limit,
                    max_plies=args.max_plies,
                )
                played.append(g)

                s = _score_from_result(g.result, model_is_white=model_is_white)
                if s == 1.0:
                    wins += 1
                elif s == 0.5:
                    draws += 1
                else:
                    losses += 1

                if args.verbose:
                    print(f"[{j+1}/{n_sample}] {opening_id} "
                          f"model={'W' if model_is_white else 'B'} "
                          f"result={g.result} term={g.termination} plies={g.plies}")
    finally:
        # Always close engines
        try:
            model_eng.quit()
        except Exception:
            pass
        try:
            stockfish.quit()
        except Exception:
            pass

    n_games = len(played)
    score = (wins + 0.5 * draws) / max(1, n_games)

    summary = TournamentSummary(
        n_openings_total=n_total,
        n_openings_sampled=n_sample,
        n_games=n_games,
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        score_formula="(wins + 0.5*draws) / N_games",
        seed=int(args.seed),
        pct=float(pct),
        limits={
            "movetime_ms": args.movetime_ms,
            "depth": args.depth,
            "nodes": args.nodes,
            "max_plies": args.max_plies,
        },
        stockfish_config={
            "stockfish": args.stockfish,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "skill_level": args.skill_level,
            "limit_strength": args.limit_strength,
            "uci_elo": args.uci_elo,
        },
        model_config={
            "model_uci_py": args.model_uci_py,
            "python": args.python,
            "model_type": args.model_type,
            "ckpt": args.ckpt,
            "tokenizer_path": args.tokenizer_path,
            "device": device,
            "temperature": args.temperature,
            "topk": args.topk,
            "greedy": bool(args.greedy),
            "max_seq_len": args.max_seq_len,
            "n_layers": args.n_layers,
            "img_size": args.img_size,
            "vit_path": args.vit_path,
            "version": args.version,
        },
    )

    return summary, played


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openings_json", required=True, help="Path to opening suite JSON (list of dicts with a FEN field).")
    ap.add_argument("--pct", type=float, default=1.0, help="Fraction of openings to sample, in [0,1].")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")

    # Opponent engine (Stockfish for now)
    ap.add_argument("--stockfish", required=True, help="Path to Stockfish binary.")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash_mb", type=int, default=256)
    ap.add_argument("--skill_level", type=int, default=None, help="Stockfish Skill Level (0..20).")
    ap.add_argument("--limit_strength", action="store_true", help="Enable UCI_LimitStrength for Stockfish.")
    ap.add_argument("--uci_elo", type=int, default=None, help="Set UCI_Elo if supported (requires --limit_strength).")

    # Search limits
    ap.add_argument("--movetime_ms", type=int, default=100, help="Move time in ms (works for both engines).")
    ap.add_argument("--depth", type=int, default=None, help="Optional depth limit (Stockfish).")
    ap.add_argument("--nodes", type=int, default=None, help="Optional node limit (Stockfish).")
    ap.add_argument("--max_plies", type=int, default=400, help="Safety cap on game length.")

    # Model UCI engine wrapper
    ap.add_argument("--model_uci_py", default="tournament_uci_engine_patched.py", help="Path to your UCI wrapper script.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to run the model UCI engine.")
    ap.add_argument("--model_type", choices=["decoder", "encoder"], required=True)
    ap.add_argument("--ckpt", required=True)

    # Model sampling params (forwarded to uci_engine.py)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--greedy", action="store_true")

    # Decoder-only
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=8)

    # Encoder-only
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--vit_path", default=None)
    ap.add_argument("--version", type=int, default=2, help="Model version (1 or 2) for encoder; ignored for decoder.")

    # Output
    ap.add_argument("--out_dir", default="tournament_out")
    ap.add_argument("--write_pgn", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    _ensure_dir(args.out_dir)

    summary, games = run_tournament(args)

    # Write summary + raw games
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    games_path = os.path.join(args.out_dir, "games.jsonl")
    with open(games_path, "w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(asdict(g)) + "\n")

    if args.write_pgn:
        pgn_path = os.path.join(args.out_dir, "games.pgn")
        model_name = f"model_{args.model_type}"
        opp_name = "stockfish"
        with open(pgn_path, "w", encoding="utf-8") as f:
            for g in games:
                f.write(to_pgn(g, model_name=model_name, opp_name=opp_name))
                f.write("\n\n")

    print("=== Tournament Summary ===")
    print(f"Openings: {summary.n_openings_sampled}/{summary.n_openings_total} (pct={summary.pct}, seed={summary.seed})")
    print(f"Games: {summary.n_games} (paired colors)")
    print(f"W/D/L: {summary.wins}/{summary.draws}/{summary.losses}")
    print(f"Score: {summary.score:.4f}  [{summary.score_formula}]")
    print(f"Saved: {summary_path}")
    print(f"Saved: {games_path}")
    if args.write_pgn:
        print(f"Saved: {os.path.join(args.out_dir, 'games.pgn')}")


if __name__ == "__main__":
    main()
