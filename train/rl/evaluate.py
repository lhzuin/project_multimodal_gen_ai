"""Read-only paired short-horizon evaluation, independent of RL optimization.

Compare both policies on exactly the same selected held-out positions. This is
not an Elo estimator or a full-game tournament. Engine scheduling and limited
strength can remain stochastic even with a fixed policy sampling seed.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Sequence

from rl_stockfish_service import AsyncRolloutClient, ServiceConfig
from train.rloo.config import RLOOConfig
from .contracts import PositionPrompt
from .config_io import load_config
from .launch import _pool
from .policies import load_policy, select_device
from .rollouts import RolloutCollector
from .server import LocalStockfishServer


def select_prompts(
    rows: Sequence[dict[str, Any]], count: int, seed: int
) -> list[PositionPrompt]:
    """Seeded random subset, reused unchanged for both policies; no score selection."""
    if not 2 <= count <= len(rows):
        raise ValueError("positions must be between 2 and the held-out database size")
    prompts = [PositionPrompt.from_mapping(row) for row in rows]
    if len({p.entry_id for p in prompts}) != len(prompts):
        raise ValueError("evaluation entry IDs must be unique")
    return random.Random(seed).sample(prompts, count)


def comparison_summary(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if baseline.keys() != candidate.keys() or len(baseline) < 2:
        raise ValueError("evaluation needs the same two or more completed positions")
    differences = [
        candidate[key]["mean_reward"] - baseline[key]["mean_reward"] for key in baseline
    ]
    delta = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    return {
        "positions": len(differences),
        "baseline_mean_reward": statistics.mean(
            v["mean_reward"] for v in baseline.values()
        ),
        "candidate_mean_reward": statistics.mean(
            v["mean_reward"] for v in candidate.values()
        ),
        "paired_reward_difference": delta,
        "paired_standard_error": se,
        "approximate_95pct_interval": [delta - 1.96 * se, delta + 1.96 * se],
        "interpretation": "Descriptive normal approximation over position-level differences, not an Elo or full-game strength claim; use a larger sample and multiple seeds.",
    }


async def evaluate_policy(
    server: Any, policy: Any, prompts: Sequence[PositionPrompt], config: RLOOConfig
) -> dict[str, Any]:
    """Collect only: no reference, optimizer, gradients, or checkpoint writes."""
    collector = RolloutCollector(
        server, policy, None, config, random.Random(config.seed)
    )
    results: dict[str, Any] = {}
    for start in range(0, len(prompts), config.training_batch_size):
        batch = prompts[start : start + config.training_batch_size]
        trajectories = await collector.collect(batch)
        grouped: dict[str, list[Any]] = {p.entry_id: [] for p in batch}
        for trajectory in trajectories:
            if (
                trajectory.result is None
                or trajectory.result.get("status") != "completed"
            ):
                raise RuntimeError(
                    "incomplete evaluation rollout; comparison must not silently drop positions"
                )
            grouped[trajectory.result["metadata"]["entry_id"]].append(trajectory)
        for prompt in batch:
            group = grouped[prompt.entry_id]
            if len(group) != config.rollouts_per_entry:
                raise RuntimeError("incomplete evaluation group")
            scores = [t.score_cp for t in group]
            rewards = [math.tanh(score / config.reward_scale_cp) for score in scores]
            results[prompt.entry_id] = {
                "mean_reward": statistics.mean(rewards),
                "scores_cp": scores,
                "rewards": rewards,
                "bucket": prompt.metadata.get("bucket", "unknown"),
            }
        print(
            json.dumps(
                {"evaluated_positions": len(results), "total_positions": len(prompts)}
            ),
            flush=True,
        )
    return results


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    rows = json.loads(args.database.read_text())
    # Refuse the configured training manifest, even if a different spelling or
    # symlink was supplied. Other supplied files are the caller's held-out data.
    if args.database.resolve() == Path(config["database"]["manifest"]).resolve():
        raise ValueError("supply validation/test data, not the training manifest")
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    prompts = select_prompts(rows, args.positions, args.seed)
    for prompt in prompts:
        prompt.validate(config["model_type"])
    rollout = replace(
        RLOOConfig(**config.get("rloo", {})),
        kl_beta=0,
        randomize_policy_color=False,
        seed=args.seed,
        rollouts_per_entry=args.rollouts,
    )
    baseline_path = args.baseline or Path(config["checkpoint_path"])

    async def compare(server: Any) -> dict[str, Any]:
        outcomes = []
        for label, path in (("baseline", baseline_path), ("candidate", args.candidate)):
            print(f"Evaluating {label}: {path}", flush=True)
            policy = load_policy(
                path, config["model_type"], config.get("model_kwargs", {})
            )
            policy.model.to(select_device(rollout.device)).eval().requires_grad_(False)
            outcomes.append(await evaluate_policy(server, policy, prompts, rollout))
            del policy
        return {
            "summary": comparison_summary(*outcomes),
            "baseline": outcomes[0],
            "candidate": outcomes[1],
        }

    server_cfg = config.get("server", {})
    if server_cfg.get("mode", "remote") == "local":
        async with LocalStockfishServer(
            _pool(server_cfg["opponent"]),
            _pool(server_cfg["evaluator"]),
            ServiceConfig(**server_cfg.get("service", {})),
        ) as server:
            result = await compare(server)
    else:
        result = await compare(
            AsyncRolloutClient(
                host=server_cfg.get("host", "127.0.0.1"),
                port=server_cfg.get("port", 8765),
                timeout_s=server_cfg.get("timeout_s", 30.0),
            )
        )
    result.update(
        seed=args.seed,
        rollouts_per_position=args.rollouts,
        baseline_checkpoint=str(baseline_path),
        candidate_checkpoint=str(args.candidate),
        database=str(args.database),
        database_sha256=hashlib.sha256(args.database.read_bytes()).hexdigest(),
        config=config,
        selected_entry_ids=[p.entry_id for p in prompts],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="exported policy weights, NOT the resumable trainer checkpoint",
    )
    parser.add_argument(
        "--baseline", type=Path, help="defaults to the config's original checkpoint"
    )
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
