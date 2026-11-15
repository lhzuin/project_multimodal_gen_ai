"""Async UCI ownership and safe bridges for the legacy synchronous game loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
from collections import deque
import threading
from typing import Any, Callable
import uuid

import chess
import chess.engine


class EngineBridge:
    """Expose sync ``play`` to game threads without blocking the event loop.

    Game threads use a dedicated executor, not asyncio's default executor:
    otherwise waiting games could occupy every thread needed by UCI searches.
    """

    def __init__(self, play: Callable[..., Any], loop: asyncio.AbstractEventLoop):
        self.play_async = play
        self.loop = loop
        self._closed = False
        self._pending: set[concurrent.futures.Future] = set()
        self._lock = threading.Lock()

    def play(self, board: chess.Board, limit: chess.engine.Limit):
        with self._lock:
            if self._closed:
                raise RuntimeError("tournament engine bridge closed")
            future = asyncio.run_coroutine_threadsafe(
                self.play_async(board.copy(stack=True), limit), self.loop
            )
            self._pending.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for future in self._pending:
                future.cancel()


class ModelPool:
    """One unchanged legacy UCI wrapper per model worker; no sampling rewrite.

    Whole board histories are sent on every query, so a worker can safely serve
    moves from different games. Each engine is exclusively owned during a play.
    """

    def __init__(self, launch: Callable[[], Any], workers: int):
        self.launch = launch
        self.workers = workers
        self.engines: list[Any] = []
        self.available: asyncio.Queue = asyncio.Queue(maxsize=workers)
        self.closed = False

    async def __aenter__(self):
        try:
            # Sequential startup bounds transient model-loading memory.
            for _ in range(self.workers):
                startup = asyncio.create_task(asyncio.to_thread(self.launch))
                try:
                    engine = await asyncio.shield(startup)
                except asyncio.CancelledError:
                    # to_thread cannot stop a subprocess launch already in
                    # progress. Reap its eventual engine rather than orphan it.
                    try:
                        engine = await startup
                    except Exception:
                        pass
                    else:
                        with contextlib.suppress(Exception):
                            engine.close()
                    raise
                self.engines.append(engine)
                self.available.put_nowait(engine)
        except BaseException:
            await self.close(abort=True)
            raise
        return self

    async def play(self, board: chess.Board, limit: chess.engine.Limit):
        engine = await self.available.get()
        try:
            return await asyncio.to_thread(engine.play, board, limit)
        finally:
            if not self.closed:
                self.available.put_nowait(engine)

    async def close(self, *, abort: bool = False):
        self.closed = True
        for engine in self.engines:
            if abort:
                # Interrupt an in-flight UCI call before awaiting game threads.
                with contextlib.suppress(Exception):
                    engine.close()
            else:
                try:
                    await asyncio.wait_for(asyncio.to_thread(engine.quit), 5.0)
                except Exception:
                    with contextlib.suppress(Exception):
                        engine.close()
        self.engines.clear()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close(abort=exc_type is not None)


class OpponentClient:
    """Coalesce concurrent move requests into bounded server batches.

    Only engine-reported search errors become game forfeits. Connection errors,
    malformed responses, or unsupported RPCs mark the run as failed rather than
    producing a spurious sweep of Stockfish losses.
    """

    def __init__(self, server: Any, batch_size: int):
        self.server = server
        self.batch_size = batch_size
        self.pending: deque = deque()
        self.tasks: set[asyncio.Task] = set()
        self.scheduled = False
        self.closed = False
        self.failure: Exception | None = None

    async def play(self, board: chess.Board, limit: chess.engine.Limit):
        if self.closed:
            raise RuntimeError("opponent client closed")
        if self.failure is not None:
            raise self.failure
        future = asyncio.get_running_loop().create_future()
        position = {
            "id": uuid.uuid4().hex,
            "start_fen": board.root().fen(en_passant="fen"),
            "moves_uci": [m.uci() for m in board.move_stack],
            "limit": {
                "movetime_ms": None if limit.time is None else round(limit.time * 1000),
                "depth": limit.depth,
                "nodes": limit.nodes,
            },
        }
        self.pending.append((position, future))
        if not self.scheduled:
            self.scheduled = True
            asyncio.get_running_loop().call_soon(self._flush)
        return await future

    def _flush(self):
        self.scheduled = False
        while self.pending and not self.closed:
            batch = [
                self.pending.popleft()
                for _ in range(min(self.batch_size, len(self.pending)))
            ]
            task = asyncio.create_task(self._send(batch))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _send(self, batch):
        try:
            responses = await self.server.play_positions(
                [position for position, _ in batch]
            )
            by_id = {response["id"]: response for response in responses}
            if len(by_id) != len(responses) or set(by_id) != {
                p["id"] for p, _ in batch
            }:
                raise RuntimeError(
                    "Stockfish response IDs do not match position requests"
                )
            for position, future in batch:
                if future.done():
                    continue
                response = by_id[position["id"]]
                error = response.get("error")
                if error is not None:
                    kind = error["type"]
                    cls = {
                        "EngineTerminatedError": chess.engine.EngineTerminatedError,
                        "EngineError": chess.engine.EngineError,
                    }.get(kind, RuntimeError)
                    future.set_exception(cls(error["message"]))
                else:
                    move = response["move_uci"]
                    move = None if move is None else chess.Move.from_uci(move)
                    future.set_result(chess.engine.PlayResult(move, None))
        except asyncio.CancelledError:
            for _, future in batch:
                future.cancel()
            raise
        except Exception as exc:
            self.failure = exc
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)

    async def close(self):
        self.closed = True
        for _, future in self.pending:
            future.cancel()
        self.pending.clear()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
