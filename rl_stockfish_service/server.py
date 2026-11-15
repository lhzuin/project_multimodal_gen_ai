"""JSON-lines TCP transport and command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
from collections.abc import Mapping
from typing import Any

from .config import EnginePoolConfig, SearchLimit, ServiceConfig
from .coordinator import RolloutCoordinator
from .engine_pool import StockfishPool
from .protocol import GameSpec
from .position_play import play_positions


LOGGER = logging.getLogger(__name__)


class RolloutTCPServer:
    """A small local RPC server with one JSON request and response per line.

    The transport is intentionally separate from the coordinator. It can be
    replaced by HTTP/gRPC or a distributed broker without changing rollout
    semantics or engine scheduling.
    """

    def __init__(
        self,
        coordinator: RolloutCoordinator,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_request_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.coordinator = coordinator
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self._server: asyncio.AbstractServer | None = None

    @property
    def sockets(self) -> tuple[Any, ...]:
        if self._server is None or self._server.sockets is None:
            return ()
        return tuple(self._server.sockets)

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            limit=self.max_request_bytes + 1,
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("server has not been started")
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            while True:
                try:
                    raw = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await self._write_error(
                        writer, None, "RequestTooLarge", "request exceeds limit"
                    )
                    return
                if not raw:
                    return
                if len(raw) > self.max_request_bytes:
                    await self._write_error(
                        writer, None, "RequestTooLarge", "request exceeds limit"
                    )
                    return
                request_id: Any = None
                try:
                    request = json.loads(raw)
                    if not isinstance(request, Mapping):
                        raise ValueError("request must be a JSON object")
                    request_id = request.get("id")
                    operation = request.get("op")
                    payload = request.get("payload", {})
                    if not isinstance(operation, str) or not operation:
                        raise ValueError("op must be a non-empty string")
                    if not isinstance(payload, Mapping):
                        raise ValueError("payload must be a JSON object")
                    result = await self._dispatch(operation, payload)
                    response = {"id": request_id, "ok": True, "result": result}
                except Exception as exc:
                    LOGGER.debug("request from %s failed", peer, exc_info=True)
                    response = {
                        "id": request_id,
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                writer.write(
                    json.dumps(
                        response, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                    + b"\n"
                )
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            return
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def _write_error(
        self,
        writer: asyncio.StreamWriter,
        request_id: Any,
        error_type: str,
        message: str,
    ) -> None:
        response = {
            "id": request_id,
            "ok": False,
            "error": {"type": error_type, "message": message},
        }
        writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        await writer.drain()

    async def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> Any:
        if operation == "health":
            return self.coordinator.health()
        if operation == "play_positions":
            positions = [
                self._mapping(value, "position")
                for value in self._list(payload, "positions")
            ]
            return await play_positions(
                self.coordinator.opponent_pool,
                positions,
                max_batch_size=self.coordinator.config.max_batch_size,
            )
        if operation == "create_games":
            values = self._list(payload, "games")
            specs = [
                GameSpec.from_dict(self._mapping(value, "game")) for value in values
            ]
            return await self.coordinator.create_games(specs)
        if operation == "poll_policy_turns":
            return await self.coordinator.poll_policy_turns(
                int(payload.get("max_items", 64)),
                self._timeout_seconds(payload),
            )
        if operation == "submit_actions":
            actions = [
                dict(self._mapping(value, "action"))
                for value in self._list(payload, "actions")
            ]
            return await self.coordinator.submit_actions(actions)
        if operation == "poll_results":
            return await self.coordinator.poll_results(
                int(payload.get("max_items", 64)),
                self._timeout_seconds(payload),
            )
        if operation == "get_games":
            game_ids = payload.get("game_ids")
            if game_ids is None:
                return self.coordinator.get_games()
            values = self._as_list(game_ids, "game_ids")
            return self.coordinator.get_games(str(value) for value in values)
        if operation == "cancel_games":
            return await self.coordinator.cancel_games(
                str(value) for value in self._list(payload, "game_ids")
            )
        if operation == "release_games":
            return self.coordinator.release_games(
                str(value) for value in self._list(payload, "game_ids")
            )
        raise ValueError(f"unknown operation: {operation}")

    @staticmethod
    def _mapping(value: Any, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        return value

    @staticmethod
    def _as_list(value: Any, name: str) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON array")
        return value

    @classmethod
    def _list(cls, payload: Mapping[str, Any], name: str) -> list[Any]:
        if name not in payload:
            raise ValueError(f"missing {name}")
        return cls._as_list(payload[name], name)

    @staticmethod
    def _timeout_seconds(payload: Mapping[str, Any]) -> float:
        timeout_ms = int(payload.get("timeout_ms", 1000))
        if timeout_ms < 0 or timeout_ms > 300_000:
            raise ValueError("timeout_ms must be in [0, 300000]")
        return timeout_ms / 1000.0


def _search_limit(args: argparse.Namespace, prefix: str) -> SearchLimit:
    movetime_ms = getattr(args, f"{prefix}_movetime_ms") or None
    depth = getattr(args, f"{prefix}_depth") or None
    nodes = getattr(args, f"{prefix}_nodes") or None
    return SearchLimit(movetime_ms=movetime_ms, depth=depth, nodes=nodes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Asynchronous Stockfish rollout service"
    )
    parser.add_argument(
        "--stockfish", required=True, help="path to the Stockfish executable"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--play-workers", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--play-threads", type=int, default=1)
    parser.add_argument("--eval-threads", type=int, default=1)
    parser.add_argument("--play-hash-mb", type=int, default=64)
    parser.add_argument("--eval-hash-mb", type=int, default=128)
    parser.add_argument("--queue-capacity", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument(
        "--play-movetime-ms", type=int, default=50, help="0 disables this limit"
    )
    parser.add_argument("--play-depth", type=int)
    parser.add_argument("--play-nodes", type=int)
    parser.add_argument(
        "--eval-movetime-ms", type=int, default=100, help="0 disables this limit"
    )
    parser.add_argument("--eval-depth", type=int)
    parser.add_argument("--eval-nodes", type=int)
    parser.add_argument("--limit-strength", action="store_true")
    parser.add_argument("--uci-elo", type=int)
    parser.add_argument("--skill-level", type=int)
    parser.add_argument(
        "--restart-attempts",
        type=int,
        default=1,
        help="retries per failed search; use 0 for tournament forfeits",
    )
    parser.add_argument("--mate-score-cp", type=int, default=100_000)
    parser.add_argument("--reward-scale-cp", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    service_config = ServiceConfig(
        host=args.host,
        port=args.port,
        policy_queue_capacity=args.queue_capacity,
        result_queue_capacity=args.queue_capacity,
        max_batch_size=args.max_batch_size,
        mate_score_cp=args.mate_score_cp,
        reward_scale_cp=args.reward_scale_cp,
        random_seed=args.seed,
    )
    common = {
        "executable": args.stockfish,
        "queue_capacity": args.queue_capacity,
        "limit_strength": args.limit_strength,
        "uci_elo": args.uci_elo,
        "skill_level": args.skill_level,
        "restart_attempts": args.restart_attempts,
    }
    opponent_pool = StockfishPool(
        EnginePoolConfig(
            **common,
            workers=args.play_workers,
            threads_per_engine=args.play_threads,
            hash_mb_per_engine=args.play_hash_mb,
            default_limit=_search_limit(args, "play"),
        ),
        name="opponent",
    )
    evaluator_pool = StockfishPool(
        EnginePoolConfig(
            executable=args.stockfish,
            workers=args.eval_workers,
            threads_per_engine=args.eval_threads,
            hash_mb_per_engine=args.eval_hash_mb,
            queue_capacity=args.queue_capacity,
            default_limit=_search_limit(args, "eval"),
        ),
        name="evaluator",
    )
    coordinator = RolloutCoordinator(
        opponent_pool=opponent_pool,
        evaluator_pool=evaluator_pool,
        config=service_config,
    )
    server = RolloutTCPServer(
        coordinator,
        host=service_config.host,
        port=service_config.port,
        max_request_bytes=service_config.max_request_bytes,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        await opponent_pool.start()
        await evaluator_pool.start()
        await server.start()
        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, stop.set)

        LOGGER.info(
            "rollout API listening on %s:%d (Stockfish CPU threads: play=%d, eval=%d)",
            service_config.host,
            service_config.port,
            args.play_workers * args.play_threads,
            args.eval_workers * args.eval_threads,
        )
        await stop.wait()
    finally:
        await server.close()
        await coordinator.shutdown()
        await opponent_pool.close()
        await evaluator_pool.close()


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
