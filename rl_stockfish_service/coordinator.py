"""Concurrent game state machine connecting policy batches to Stockfish."""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import chess

from .config import ServiceConfig
from .protocol import EngineEvaluation, GameSpec, GameStatus, PolicyColor, Termination


class EnginePool(Protocol):
    async def play(self, board: chess.Board, limit: Any = None) -> chess.Move: ...

    async def evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
        *,
        mate_score_cp: int,
        limit: Any = None,
    ) -> EngineEvaluation: ...

    def metrics(self) -> dict[str, Any]: ...


@dataclass
class _Game:
    game_id: str
    spec: GameSpec
    board: chess.Board
    policy_color: chess.Color
    moves_uci: list[str]
    rollout_plies: int = 0
    status: GameStatus = GameStatus.QUEUED
    termination: Termination | None = None
    pending_request_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RolloutCoordinator:
    """Owns game state while engines and the trainer own only computations.

    No model is loaded here. Whenever it is the policy's turn, an immutable
    observation is placed on ``policy_queue``. This lets one trainer batch
    observations from many games and sample all moves in one accelerator call.
    """

    def __init__(
        self,
        *,
        opponent_pool: EnginePool,
        evaluator_pool: EnginePool,
        config: ServiceConfig,
    ) -> None:
        self.opponent_pool = opponent_pool
        self.evaluator_pool = evaluator_pool
        self.config = config
        self.policy_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=config.policy_queue_capacity
        )
        self.result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=config.result_queue_capacity
        )
        self._games: dict[str, _Game] = {}
        self._request_to_game: dict[str, str] = {}
        self._rng = random.Random(config.random_seed)
        self._background: set[asyncio.Task[None]] = set()
        self._game_tasks: dict[str, asyncio.Task[None]] = {}

    async def create_games(self, specs: Iterable[GameSpec]) -> list[dict[str, Any]]:
        staged: list[tuple[str, GameSpec]] = []
        staged_ids: set[str] = set()
        for spec in specs:
            game_id = spec.game_id or uuid.uuid4().hex
            if game_id in self._games or game_id in staged_ids:
                raise ValueError(f"duplicate game_id: {game_id}")
            staged.append((game_id, spec))
            staged_ids.add(game_id)

        created: list[dict[str, Any]] = []
        for game_id, spec in staged:
            board = chess.Board(spec.start_fen)
            moves = []
            for uci in spec.opening_moves_uci:
                move = chess.Move.from_uci(uci)
                board.push(move)
                moves.append(uci)
            policy_color = self._resolve_color(spec.policy_color)
            game = _Game(
                game_id=game_id,
                spec=spec,
                board=board,
                policy_color=policy_color,
                moves_uci=moves,
            )
            self._games[game_id] = game
            created.append(self._game_summary(game))
            self._schedule(game)
        return created

    def _resolve_color(self, color: PolicyColor) -> chess.Color:
        if color == PolicyColor.WHITE:
            return chess.WHITE
        if color == PolicyColor.BLACK:
            return chess.BLACK
        return chess.WHITE if self._rng.randrange(2) == 0 else chess.BLACK

    def _schedule(self, game: _Game) -> None:
        task = asyncio.create_task(self._advance(game))
        self._background.add(task)
        self._game_tasks[game.game_id] = task

        def remove_task(completed: asyncio.Task[None]) -> None:
            self._background.discard(completed)
            if self._game_tasks.get(game.game_id) is completed:
                self._game_tasks.pop(game.game_id, None)

        task.add_done_callback(remove_task)

    async def _advance(self, game: _Game) -> None:
        async with game.lock:
            try:
                while not self._is_terminal_status(game.status):
                    game.status = GameStatus.RUNNING

                    if game.board.is_game_over(claim_draw=True):
                        await self._finish_terminal_board(game)
                        return
                    if game.rollout_plies >= game.spec.horizon_plies:
                        await self._finish_with_evaluation(game, Termination.HORIZON)
                        return
                    if game.rollout_plies >= game.spec.max_plies:
                        await self._finish_with_evaluation(game, Termination.MAX_PLIES)
                        return

                    if game.board.turn == game.policy_color:
                        await self._queue_policy_turn(game)
                        return

                    move = await self.opponent_pool.play(game.board)
                    if move not in game.board.legal_moves:
                        raise RuntimeError(
                            f"Stockfish returned illegal move {move.uci()}"
                        )
                    game.board.push(move)
                    game.moves_uci.append(move.uci())
                    game.rollout_plies += 1
            except Exception as exc:
                game.error = f"{type(exc).__name__}: {exc}"
                await self._finish(
                    game, Termination.ENGINE_ERROR, reward_cp=0, evaluation=None
                )

    async def _queue_policy_turn(self, game: _Game) -> None:
        request_id = uuid.uuid4().hex
        game.pending_request_id = request_id
        game.status = GameStatus.WAITING_POLICY
        self._request_to_game[request_id] = game.game_id
        observation = {
            "request_id": request_id,
            "game_id": game.game_id,
            "fen": game.board.fen(),
            "start_fen": game.spec.start_fen,
            "moves_uci": list(game.moves_uci),
            "legal_moves_uci": [move.uci() for move in game.board.legal_moves],
            "policy_color": "white" if game.policy_color == chess.WHITE else "black",
            "rollout_ply": game.rollout_plies,
            "absolute_ply": game.board.ply(),
            "horizon_plies": game.spec.horizon_plies,
            "metadata": dict(game.spec.metadata),
        }
        await self.policy_queue.put(observation)

    async def submit_actions(
        self, actions: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tasks = [self._submit_one_action(action) for action in actions]
        return list(await asyncio.gather(*tasks))

    async def _submit_one_action(self, action: dict[str, Any]) -> dict[str, Any]:
        request_id = str(action.get("request_id", ""))
        move_uci = str(action.get("move_uci", ""))
        game_id = self._request_to_game.get(request_id)
        if game_id is None:
            return {
                "request_id": request_id,
                "accepted": False,
                "error": "unknown request_id",
            }
        game = self._games.get(game_id)
        if game is None:
            return {
                "request_id": request_id,
                "accepted": False,
                "error": "game was released",
            }

        async with game.lock:
            if (
                game.pending_request_id != request_id
                or game.status != GameStatus.WAITING_POLICY
            ):
                return {
                    "request_id": request_id,
                    "accepted": False,
                    "error": "stale action",
                }
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                move = None

            del self._request_to_game[request_id]
            game.pending_request_id = None
            if move is None or move not in game.board.legal_moves:
                await self._finish(
                    game,
                    Termination.ILLEGAL_POLICY_MOVE,
                    reward_cp=-self.config.mate_score_cp,
                    evaluation=None,
                )
                return {
                    "request_id": request_id,
                    "game_id": game_id,
                    "accepted": False,
                    "error": f"illegal policy move: {move_uci}",
                }

            game.board.push(move)
            game.moves_uci.append(move.uci())
            game.rollout_plies += 1
            game.status = GameStatus.RUNNING

        # The action endpoint is an enqueue/ack boundary. Continue the game in
        # the background so a slow Stockfish search never holds up the trainer.
        self._schedule(game)
        return {"request_id": request_id, "game_id": game_id, "accepted": True}

    async def _finish_terminal_board(self, game: _Game) -> None:
        outcome = game.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            termination = Termination.DRAW
            reward_cp = 0
        else:
            termination = Termination.CHECKMATE
            reward_cp = (
                self.config.mate_score_cp
                if outcome.winner == game.policy_color
                else -self.config.mate_score_cp
            )
        await self._finish(game, termination, reward_cp=reward_cp, evaluation=None)

    async def _finish_with_evaluation(
        self, game: _Game, termination: Termination
    ) -> None:
        game.status = GameStatus.EVALUATING
        evaluation = await self.evaluator_pool.evaluate(
            game.board,
            game.policy_color,
            mate_score_cp=self.config.mate_score_cp,
        )
        await self._finish(
            game,
            termination,
            reward_cp=evaluation.centipawns,
            evaluation=evaluation,
        )

    async def _finish(
        self,
        game: _Game,
        termination: Termination,
        *,
        reward_cp: int,
        evaluation: EngineEvaluation | None,
    ) -> None:
        if game.pending_request_id is not None:
            self._request_to_game.pop(game.pending_request_id, None)
            game.pending_request_id = None
        game.termination = termination
        game.status = (
            GameStatus.FAILED
            if termination == Termination.ENGINE_ERROR
            else GameStatus.COMPLETED
        )
        bounded_reward = math.tanh(reward_cp / self.config.reward_scale_cp)
        result = {
            "game_id": game.game_id,
            "status": game.status,
            "termination": termination,
            "reward": {
                "centipawns": reward_cp,
                "tanh": bounded_reward,
                "perspective": "policy",
                "scale_cp": self.config.reward_scale_cp,
            },
            "evaluation": None if evaluation is None else evaluation.to_dict(),
            "result": game.board.result(claim_draw=True),
            "final_fen": game.board.fen(),
            "moves_uci": list(game.moves_uci),
            "rollout_plies": game.rollout_plies,
            "absolute_ply": game.board.ply(),
            "policy_color": "white" if game.policy_color == chess.WHITE else "black",
            "metadata": dict(game.spec.metadata),
            "error": game.error,
        }
        game.result = result
        await self.result_queue.put(result)

    async def poll_policy_turns(
        self, max_items: int, timeout_s: float
    ) -> list[dict[str, Any]]:
        # Cancellation can leave an observation already buffered. Never hand a
        # trainer a request that the coordinator can no longer accept.
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
        while True:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            items = await self._poll_queue(self.policy_queue, max_items, remaining)
            current = [item for item in items if self._is_current_observation(item)]
            if current or remaining <= 0 or not items:
                return current

    async def poll_results(
        self, max_items: int, timeout_s: float
    ) -> list[dict[str, Any]]:
        return await self._poll_queue(self.result_queue, max_items, timeout_s)

    async def _poll_queue(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        max_items: int,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        max_items = min(max(1, int(max_items)), self.config.max_batch_size)
        items: list[dict[str, Any]] = []
        if queue.empty() and timeout_s > 0:
            try:
                items.append(await asyncio.wait_for(queue.get(), timeout=timeout_s))
            except TimeoutError:
                return []
        while len(items) < max_items:
            try:
                items.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def get_games(self, game_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        games = (
            self._games.values()
            if game_ids is None
            else (
                self._games[game_id] for game_id in game_ids if game_id in self._games
            )
        )
        return [self._game_summary(game) for game in games]

    async def cancel_games(self, game_ids: Iterable[str]) -> list[dict[str, Any]]:
        output = []
        for game_id in game_ids:
            game = self._games.get(str(game_id))
            if game is None:
                output.append(
                    {
                        "game_id": str(game_id),
                        "cancelled": False,
                        "error": "unknown game",
                    }
                )
                continue
            # An advance task can be blocked on a full policy/result queue while
            # holding the game lock. Cancel it first so cancellation never
            # requires the trainer to resume consuming that queue.
            task = self._game_tasks.get(game.game_id)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            async with game.lock:
                if self._is_terminal_status(game.status):
                    output.append(
                        {
                            "game_id": game.game_id,
                            "cancelled": False,
                            "error": "already terminal",
                        }
                    )
                    continue
                game.status = GameStatus.CANCELLED
                game.termination = Termination.CANCELLED
                if game.pending_request_id:
                    self._request_to_game.pop(game.pending_request_id, None)
                    game.pending_request_id = None
                output.append({"game_id": game.game_id, "cancelled": True})
        return output

    def release_games(self, game_ids: Iterable[str]) -> list[dict[str, Any]]:
        output = []
        for game_id in game_ids:
            game = self._games.get(str(game_id))
            if game is None:
                output.append(
                    {
                        "game_id": str(game_id),
                        "released": False,
                        "error": "unknown game",
                    }
                )
            elif not self._is_terminal_status(game.status):
                output.append(
                    {
                        "game_id": game.game_id,
                        "released": False,
                        "error": "game is active",
                    }
                )
            else:
                del self._games[game.game_id]
                output.append({"game_id": game.game_id, "released": True})
        return output

    def health(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in GameStatus}
        for game in self._games.values():
            counts[game.status.value] += 1
        return {
            "api_version": "1",
            "status": "ok",
            "games": counts,
            "policy_queue_size": self.policy_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "opponent_pool": self.opponent_pool.metrics(),
            "evaluator_pool": self.evaluator_pool.metrics(),
        }

    @staticmethod
    def _is_terminal_status(status: GameStatus) -> bool:
        return status in {GameStatus.COMPLETED, GameStatus.FAILED, GameStatus.CANCELLED}

    def _is_current_observation(self, observation: dict[str, Any]) -> bool:
        request_id = str(observation.get("request_id", ""))
        game_id = self._request_to_game.get(request_id)
        if game_id is None:
            return False
        game = self._games.get(game_id)
        return bool(
            game
            and game.status == GameStatus.WAITING_POLICY
            and game.pending_request_id == request_id
        )

    @staticmethod
    def _game_summary(game: _Game) -> dict[str, Any]:
        return {
            "game_id": game.game_id,
            "status": game.status,
            "policy_color": "white" if game.policy_color == chess.WHITE else "black",
            "rollout_plies": game.rollout_plies,
            "absolute_ply": game.board.ply(),
            "pending_request_id": game.pending_request_id,
            "termination": game.termination,
            "error": game.error,
            "result": game.result,
        }

    async def wait_background(self) -> None:
        if self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def shutdown(self) -> None:
        """Cancel in-flight state machines without relying on queue consumers."""
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for game in self._games.values():
            if not self._is_terminal_status(game.status):
                if game.pending_request_id:
                    self._request_to_game.pop(game.pending_request_id, None)
                    game.pending_request_id = None
                game.status = GameStatus.CANCELLED
                game.termination = Termination.CANCELLED
