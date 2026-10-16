"""Bounded concurrent rollout collection, independent of the RL objective."""

from __future__ import annotations

import asyncio
import copy
import random
import uuid
from collections import deque
from typing import Any, Mapping, Sequence

import torch

from .config import RolloutConfig
from .contracts import (
    PolicyAdapter,
    PolicyDecision,
    PositionPrompt,
    RolloutServer,
    Trajectory,
)


class RolloutCollector:
    """Collect on-policy groups while keeping model weights fixed throughout.

    The server queues have one logical consumer. Use a dedicated server instance
    for this trainer; request IDs prevent duplicates but do not route work across
    unrelated trainers. No optimizer update may run concurrently with collect().
    """

    def __init__(
        self,
        server: RolloutServer,
        policy: PolicyAdapter,
        reference: PolicyAdapter | None,
        config: RolloutConfig,
        rng: random.Random,
    ) -> None:
        self.server = server
        self.policy = policy
        self.reference = reference
        self.config = config
        self.rng = rng
        self.sampling_rng = torch.Generator(device="cpu").manual_seed(config.seed)

    def _sample(
        self, observations: Sequence[Mapping[str, Any]]
    ) -> list[PolicyDecision]:
        # This worker call lets an embedded coordinator process other games while
        # accelerator inference is busy. All model access remains serial.
        with torch.no_grad():
            distributions = self.policy.log_distributions(
                observations, self.config.temperature
            )
            reference = (
                self.reference.log_distributions(observations, self.config.temperature)
                if self.reference is not None
                else [None] * len(observations)
            )
            decisions = []
            for observation, log_probs, ref in zip(
                observations, distributions, reference, strict=True
            ):
                if not torch.isfinite(log_probs).all():
                    raise ValueError(
                        "policy produced nonfinite legal log probabilities"
                    )
                index = int(
                    torch.multinomial(
                        log_probs.exp().cpu(), 1, generator=self.sampling_rng
                    ).item()
                )
                decisions.append(
                    PolicyDecision(
                        observation=copy.deepcopy(dict(observation)),
                        action_index=index,
                        old_log_prob=float(log_probs[index].item()),
                        reference_log_probs=(
                            None if ref is None else tuple(ref.cpu().tolist())
                        ),
                    )
                )
            return decisions

    async def collect(self, prompts: Sequence[PositionPrompt]) -> list[Trajectory]:
        health = await self.server.health()
        if health.get("api_version") != "1":
            raise ValueError("Stockfish server must implement API version 1")
        run_id = uuid.uuid4().hex
        pending: deque[dict[str, Any]] = deque()
        trajectories: dict[str, Trajectory] = {}
        for entry_index, prompt in enumerate(prompts):
            group_id = f"{run_id}-{entry_index}"
            # A rollout group compares the same environment, including color.
            if self.config.randomize_policy_color:
                color = self.rng.choice(("white", "black"))
            else:
                # The selected position is AFTER the supplied opening history,
                # not necessarily the side to move in the base FEN.
                color = "white" if prompt.board().turn else "black"
            for sample_index in range(self.config.rollouts_per_entry):
                game_id = f"{group_id}-{sample_index}"
                trajectories[game_id] = Trajectory(game_id, group_id)
                metadata = dict(prompt.metadata)
                metadata.update(
                    {
                        "entry_id": prompt.entry_id,
                        "group_id": group_id,
                        "sample_index": sample_index,
                    }
                )
                pending.append(
                    {
                        "game_id": game_id,
                        "start_fen": prompt.start_fen,
                        "opening_moves_uci": list(prompt.moves_uci),
                        "policy_color": color,
                        "horizon_plies": self.config.ply_depth,
                        "max_plies": max(400, self.config.ply_depth),
                        "metadata": metadata,
                    }
                )

        created: set[str] = set()
        active: set[str] = set()
        completed: set[str] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.rollout_timeout_s

        async def fill_slots() -> None:
            batch = []
            while (
                pending and len(active) + len(batch) < self.config.max_concurrent_games
            ):
                batch.append(pending.popleft())
            if batch:
                # Register before sending: on a lost network ack cleanup still
                # knows which IDs may have been created at the server.
                ids = {game["game_id"] for game in batch}
                created.update(ids)
                response = await self.server.create_games(batch)
                if {game["game_id"] for game in response} != ids:
                    raise RuntimeError("server did not acknowledge all requested games")
                active.update(ids)

        self.policy.model.eval()
        if self.reference is not None:
            self.reference.model.eval()
        try:
            await fill_slots()
            while active or pending:
                progress = False
                results = await self.server.poll_results(
                    max_items=self.config.max_concurrent_games,
                    timeout_ms=0,
                )
                for result in results:
                    game_id = result["game_id"]
                    if game_id not in active:
                        raise RuntimeError(
                            "received unrelated/duplicate result; use a dedicated server"
                        )
                    trajectories[game_id].result = result
                    active.remove(game_id)
                    completed.add(game_id)
                    progress = True
                await fill_slots()
                if not active:
                    break
                observations = await self.server.poll_policy_turns(
                    max_items=self.config.sampling_batch_size,
                    timeout_ms=self.config.poll_timeout_ms,
                )
                if observations:
                    if any(turn["game_id"] not in active for turn in observations):
                        raise RuntimeError(
                            "received unrelated policy work; use a dedicated server"
                        )
                    decisions = await asyncio.to_thread(self._sample, observations)
                    actions = []
                    for decision in decisions:
                        observation = decision.observation
                        trajectories[observation["game_id"]].decisions.append(decision)
                        actions.append(
                            {
                                "request_id": observation["request_id"],
                                "move_uci": observation["legal_moves_uci"][
                                    decision.action_index
                                ],
                            }
                        )
                    acknowledgements = await self.server.submit_actions(actions)
                    if len(acknowledgements) != len(actions) or any(
                        not ack["accepted"] for ack in acknowledgements
                    ):
                        raise RuntimeError(
                            f"server rejected sampled legal actions: {acknowledgements}"
                        )
                    progress = True
                if progress:
                    deadline = loop.time() + self.config.rollout_timeout_s
                elif loop.time() >= deadline:
                    raise TimeoutError(
                        "Stockfish rollout collection made no progress before timeout"
                    )
            return list(trajectories.values())
        finally:
            # Polling is destructive. Cleanup avoids stranded waiting-policy
            # games on exceptions, and releases terminal state after consumption.
            unfinished = list(created - completed)
            if unfinished:
                await self.server.cancel_games(unfinished)
                # Completion may race with an exception. Consume results for
                # cancelled/just-finished games before reusing this dedicated
                # environment for the next collection batch.
                while await self.server.poll_results(
                    max_items=self.config.max_concurrent_games,
                    timeout_ms=0,
                ):
                    pass
            if created:
                await self.server.release_games(list(created))
