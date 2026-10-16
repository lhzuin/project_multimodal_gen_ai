"""End-to-end example: a uniformly sampled legal policy versus Stockfish."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import uuid

from .client import AsyncRolloutClient


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    client = AsyncRolloutClient(args.host, args.port, timeout_s=30)
    run_id = uuid.uuid4().hex[:8]
    games = [
        {
            "game_id": f"random-{run_id}-{index}",
            "policy_color": "white" if index % 2 == 0 else "black",
            "horizon_plies": args.horizon_plies,
            "metadata": {"group_id": index // args.group_size, "sample_index": index},
        }
        for index in range(args.games)
    ]
    await client.create_games(games)

    completed = 0
    while completed < args.games:
        turns = await client.poll_policy_turns(
            max_items=args.batch_size,
            timeout_ms=100,
        )
        if turns:
            # Replace this list comprehension with one batched policy forward
            # pass and sample from the masked move distribution.
            actions = [
                {
                    "request_id": turn["request_id"],
                    "move_uci": rng.choice(turn["legal_moves_uci"]),
                }
                for turn in turns
            ]
            await client.submit_actions(actions)

        results = await client.poll_results(
            max_items=args.batch_size,
            timeout_ms=10,
        )
        for result in results:
            completed += 1
            print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--horizon-plies", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.games < 1 or args.group_size < 1 or args.batch_size < 1:
        parser.error("games, group-size, and batch-size must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
