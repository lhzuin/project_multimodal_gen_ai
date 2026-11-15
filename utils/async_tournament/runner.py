"""Bounded concurrent scheduling around the original tournament game function."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
import time
from typing import Any

from rl_stockfish_service import AsyncRolloutClient, EnginePoolConfig, SearchLimit
from train.rl.server import LocalStockfishServer
from utils import tournament_runner as legacy
from .engines import EngineBridge, ModelPool, OpponentClient


def build_parser():
    parser = legacy.build_parser()
    parser.description = (
        "Parallel model-vs-Stockfish tournament using the async service"
    )
    parser.set_defaults(model_uci_py="utils/tournament_uci_engine.py")
    parser.add_argument(
        "--max_concurrent_games",
        type=int,
        default=4,
        help="bounded number of simultaneous full games",
    )
    parser.add_argument(
        "--stockfish_workers",
        type=int,
        default=2,
        help="Stockfish processes; --threads/--hash_mb apply to EACH",
    )
    parser.add_argument(
        "--model_workers",
        type=int,
        default=1,
        help="UCI model replicas; each loads a complete model",
    )
    parser.add_argument("--stockfish_rpc_batch_size", type=int, default=16)
    parser.add_argument("--queue_capacity", type=int, default=128)
    parser.add_argument("--clear_hash_before_search", action="store_true")
    parser.add_argument("--server_mode", choices=("local", "remote"), default="local")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8765)
    parser.add_argument("--server_timeout_s", type=float, default=30.0)
    return parser


def pool_config(args) -> EnginePoolConfig:
    limit = legacy._limit_from_args(args)
    return EnginePoolConfig(
        executable=args.stockfish,
        workers=args.stockfish_workers,
        threads_per_engine=args.threads,
        hash_mb_per_engine=args.hash_mb,
        queue_capacity=args.queue_capacity,
        skill_level=args.skill_level,
        limit_strength=args.limit_strength,
        uci_elo=args.uci_elo,
        restart_attempts=0,  # Original game loop adjudicates a failed search.
        clear_hash_before_search=getattr(args, "clear_hash_before_search", False),
        default_limit=SearchLimit(
            movetime_ms=None if limit.time is None else round(limit.time * 1000),
            depth=limit.depth,
            nodes=limit.nodes,
        ),
    )


def validate(args):
    for name in (
        "max_concurrent_games",
        "stockfish_workers",
        "model_workers",
        "stockfish_rpc_batch_size",
        "queue_capacity",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.max_plies < 0:
        raise ValueError("--max_plies cannot be negative")
    if args.queue_capacity < args.stockfish_workers:
        raise ValueError("--queue_capacity must be at least --stockfish_workers")
    if args.server_timeout_s <= 0:
        raise ValueError("--server_timeout_s must be positive")
    pool_config(args)


def validate_server(health, expected: EnginePoolConfig, batch_size: int):
    """Never silently report CLI strength/settings different from the server's."""
    if health.get("api_version") != "1" or "play_positions" not in health.get(
        "capabilities", []
    ):
        raise ValueError("server must support API v1 play_positions")
    actual = health.get("opponent_pool", {}).get("configuration")
    if actual is None:
        raise ValueError(
            "server must publish opponent configuration for tournament auditing"
        )
    expected_values = expected.public_dict()
    for key in (
        "workers",
        "threads_per_engine",
        "hash_mb_per_engine",
        "skill_level",
        "limit_strength",
        "uci_elo",
        "default_limit",
        "restart_attempts",
        "clear_hash_before_search",
    ):
        actual_value = actual.get(key, False) if key == "clear_hash_before_search" else actual.get(key)
        if actual_value != expected_values[key]:
            raise ValueError(
                f"server opponent setting {key}: expected {expected_values[key]!r}, got {actual.get(key)!r}"
            )
    maximum = health.get("max_batch_size", 0)
    if batch_size > maximum:
        raise ValueError(
            f"--stockfish_rpc_batch_size exceeds server max_batch_size={maximum}"
        )


async def run_games(args, server, *, launch_model=None):
    """Return results in original sampled-opening / white-then-black order.

    Inject a compatible model launcher and server for deterministic parity tests.
    Only max_concurrent_games game threads/futures are created at any time.
    """
    validate(args)
    health = await server.health()
    validate_server(health, pool_config(args), args.stockfish_rpc_batch_size)
    n_total, pct, selected = legacy.select_openings(args)
    jobs = [
        (index, entry, color) for index, entry in selected for color in (True, False)
    ]
    games: list[Any] = [None] * len(jobs)
    limit = legacy._limit_from_args(args)
    loop = asyncio.get_running_loop()
    opponent = OpponentClient(server, args.stockfish_rpc_batch_size)
    game_executor = ThreadPoolExecutor(
        max_workers=args.max_concurrent_games, thread_name_prefix="tournament-game"
    )
    bridge_model = bridge_opponent = None
    tasks: list[asyncio.Task] = []
    started = time.monotonic()
    success = False
    pool = ModelPool(
        launch_model or functools.partial(legacy.launch_model_uci_engine, args),
        args.model_workers,
    )
    try:
        await pool.__aenter__()
        bridge_model = EngineBridge(pool.play, loop)
        bridge_opponent = EngineBridge(opponent.play, loop)
        cursor = 0

        async def worker():
            nonlocal cursor
            while cursor < len(jobs):
                job_index = cursor
                cursor += 1
                index, entry, white = jobs[job_index]
                opening_id = legacy._opening_id(entry, index)
                # The sequential runner probes the FEN before calling the game
                # function, even when a prefix exists. Keep that same behavior.
                legacy._pick_fen(entry)
                game = await loop.run_in_executor(
                    game_executor,
                    functools.partial(
                        legacy.play_single_game,
                        opening_id=opening_id,
                        opening_entry=entry,
                        model_eng=bridge_model,
                        opp_eng=bridge_opponent,
                        model_is_white=white,
                        limit=limit,
                        max_plies=args.max_plies,
                    ),
                )
                if opponent.failure is not None:
                    raise RuntimeError(
                        "Stockfish service communication failed; tournament aborted without final score"
                    ) from opponent.failure
                games[job_index] = game
                if args.verbose:
                    print(
                        f"[{job_index + 1}/{len(jobs)}] {opening_id} model={'W' if white else 'B'} result={game.result} term={game.termination} plies={game.plies}",
                        flush=True,
                    )

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(len(jobs), args.max_concurrent_games))
        ]
        await asyncio.gather(*tasks)
        success = True
    finally:
        if bridge_model is not None:
            bridge_model.close()
        if bridge_opponent is not None:
            bridge_opponent.close()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await opponent.close()
        await pool.close(abort=not success)
        # Keep the event loop available until all sync game calls have unwound.
        await asyncio.to_thread(game_executor.shutdown, wait=True, cancel_futures=True)

    summary = legacy.build_summary(args, n_total, len(selected), pct, games)
    execution = {
        "runner": "async-stockfish-v1",
        "elapsed_seconds": time.monotonic() - started,
        "parallelism": {
            name: getattr(args, name)
            for name in (
                "max_concurrent_games",
                "stockfish_workers",
                "model_workers",
                "stockfish_rpc_batch_size",
                "queue_capacity",
            )
        },
        "server_mode": args.server_mode,
        "server_address": (
            None
            if args.server_mode == "local"
            else [args.server_host, args.server_port]
        ),
        "server_health_at_start": health,
        "server_health_at_end": await server.health(),
        "protocol": "legacy paired full games; no CP adjudication; '*' scores 0.5",
        "reproducibility": "seed selects openings only, as in the original; concurrent model RNG consumption, engine caches and wall-time searches can change individual trajectories",
    }
    return summary, games, execution


async def run_tournament(args):
    validate(args)
    if args.server_mode == "remote":
        return await run_games(
            args,
            AsyncRolloutClient(
                host=args.server_host,
                port=args.server_port,
                timeout_s=args.server_timeout_s,
            ),
        )
    config = pool_config(args)
    # The evaluator is lazy and never queried: full-game tournaments do NOT
    # use horizon/centipawn scores. No evaluator subprocess is launched.
    evaluator = EnginePoolConfig(
        executable=args.stockfish, workers=1, hash_mb_per_engine=1
    )
    async with LocalStockfishServer(config, evaluator) as server:
        return await run_games(args, server)


def main():
    args = build_parser().parse_args()
    summary, games, execution = asyncio.run(run_tournament(args))
    legacy._ensure_dir(args.out_dir)
    legacy.write_outputs(args, summary, games)
    with (Path(args.out_dir) / "execution.json").open("w", encoding="utf-8") as handle:
        json.dump(execution, handle, indent=2)
        handle.write("\n")
