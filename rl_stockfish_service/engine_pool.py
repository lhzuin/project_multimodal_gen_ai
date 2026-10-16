"""Bounded asynchronous work queue over independent Stockfish processes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Literal

import chess
import chess.engine

from .config import EnginePoolConfig, SearchLimit
from .protocol import EngineEvaluation


LOGGER = logging.getLogger(__name__)


@dataclass
class _EngineRequest:
    kind: Literal["play", "evaluate"]
    board: chess.Board
    limit: SearchLimit
    perspective: chess.Color | None
    mate_score_cp: int
    future: asyncio.Future[object]


class StockfishPool:
    """A pool where each worker exclusively owns one Stockfish subprocess.

    Multiple engine processes give predictable CPU parallelism and avoid
    locking a shared UCI process. Calls are backpressured by a bounded queue.
    Blocking python-chess calls run in helper threads; the asyncio event loop
    remains free to coordinate policy batches and network clients.
    """

    def __init__(self, config: EnginePoolConfig, *, name: str) -> None:
        self.config = config
        self.name = name
        self._queue: asyncio.Queue[_EngineRequest | None] = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False
        self._closed = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._busy = 0
        self._engine_restarts = 0

    async def start(self) -> None:
        if self._started:
            return
        self.config.validate_executable()
        self._started = True
        self._tasks = [
            asyncio.create_task(
                self._worker(worker_id), name=f"{self.name}-{worker_id}"
            )
            for worker_id in range(self.config.workers)
        ]

    async def play(
        self, board: chess.Board, limit: SearchLimit | None = None
    ) -> chess.Move:
        result = await self._submit(
            kind="play",
            board=board,
            limit=limit or self.config.default_limit,
            perspective=None,
            mate_score_cp=100_000,
        )
        if not isinstance(result, chess.Move):
            raise TypeError("Stockfish play worker returned a non-move")
        return result

    async def evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
        *,
        mate_score_cp: int,
        limit: SearchLimit | None = None,
    ) -> EngineEvaluation:
        result = await self._submit(
            kind="evaluate",
            board=board,
            limit=limit or self.config.default_limit,
            perspective=perspective,
            mate_score_cp=mate_score_cp,
        )
        if not isinstance(result, EngineEvaluation):
            raise TypeError("Stockfish evaluation worker returned an invalid result")
        return result

    async def _submit(
        self,
        *,
        kind: Literal["play", "evaluate"],
        board: chess.Board,
        limit: SearchLimit,
        perspective: chess.Color | None,
        mate_score_cp: int,
    ) -> object:
        if not self._started or self._closed:
            raise RuntimeError(f"Stockfish pool {self.name!r} is not running")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        request = _EngineRequest(
            kind=kind,
            # Retain move history so the UCI engine can reason about repetition
            # instead of receiving only the current FEN.
            board=board.copy(stack=True),
            limit=limit,
            perspective=perspective,
            mate_score_cp=mate_score_cp,
            future=future,
        )
        await self._queue.put(request)
        self._submitted += 1
        return await future

    async def _worker(self, worker_id: int) -> None:
        engine: chess.engine.SimpleEngine | None = None
        try:
            while True:
                request = await self._queue.get()
                is_busy = False
                try:
                    if request is None:
                        return
                    if request.future.cancelled():
                        continue
                    self._busy += 1
                    is_busy = True
                    error: BaseException | None = None
                    for attempt in range(self.config.restart_attempts + 1):
                        try:
                            if engine is None:
                                engine = await asyncio.to_thread(self._launch_engine)
                            result = await asyncio.to_thread(
                                self._execute, engine, request
                            )
                            if not request.future.done():
                                request.future.set_result(result)
                            self._completed += 1
                            error = None
                            break
                        # Surface every ordinary request/engine failure to the
                        # caller. Otherwise an unexpected python-chess error
                        # could kill the worker and leave its Future hanging.
                        except Exception as exc:
                            error = exc
                            LOGGER.warning(
                                "%s worker %d failed attempt %d: %s",
                                self.name,
                                worker_id,
                                attempt + 1,
                                exc,
                            )
                            await self._close_engine(engine)
                            engine = None
                            self._engine_restarts += 1
                    if error is not None and not request.future.done():
                        request.future.set_exception(error)
                        self._failed += 1
                finally:
                    if is_busy:
                        self._busy -= 1
                    self._queue.task_done()
        except asyncio.CancelledError:
            raise
        finally:
            await self._close_engine(engine)

    def _launch_engine(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(
            str(self.config.validate_executable())
        )
        options: dict[str, object] = {
            "Threads": self.config.threads_per_engine,
            "Hash": self.config.hash_mb_per_engine,
        }
        if self.config.skill_level is not None:
            options["Skill Level"] = self.config.skill_level
        if self.config.limit_strength:
            options["UCI_LimitStrength"] = True
        if self.config.uci_elo is not None:
            options["UCI_Elo"] = self.config.uci_elo

        unsupported = [name for name in options if name not in engine.options]
        if unsupported:
            engine.quit()
            raise RuntimeError(f"Stockfish does not support options: {unsupported}")
        engine.configure(options)
        return engine

    @staticmethod
    def _execute(engine: chess.engine.SimpleEngine, request: _EngineRequest) -> object:
        if request.kind == "play":
            result = engine.play(request.board, request.limit.to_chess())
            if result.move is None or result.move not in request.board.legal_moves:
                raise chess.engine.EngineError("Stockfish returned no legal move")
            return result.move

        if request.perspective is None:
            raise ValueError("evaluation requests require a perspective")
        info = engine.analyse(
            request.board,
            request.limit.to_chess(),
            info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC,
        )
        pov_score = info.get("score")
        if pov_score is None:
            raise chess.engine.EngineError("Stockfish analysis returned no score")
        score = pov_score.pov(request.perspective)
        cp = score.score(mate_score=request.mate_score_cp)
        if cp is None:
            raise chess.engine.EngineError(
                "Stockfish score could not be converted to centipawns"
            )
        return EngineEvaluation(
            centipawns=int(cp),
            mate=score.mate(),
            depth=None if info.get("depth") is None else int(info["depth"]),
            nodes=None if info.get("nodes") is None else int(info["nodes"]),
        )

    async def _close_engine(self, engine: chess.engine.SimpleEngine | None) -> None:
        if engine is None:
            return
        try:
            await asyncio.wait_for(asyncio.to_thread(engine.quit), timeout=5.0)
        except Exception:
            with contextlib.suppress(Exception):
                engine.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        for _ in self._tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def metrics(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "workers": self.config.workers,
            "busy_workers": self._busy,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.config.queue_capacity,
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
            "engine_restarts": self._engine_restarts,
        }
