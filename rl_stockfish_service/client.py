"""Async JSON-lines client intended for the future RL trainer."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Iterable, Mapping


class RolloutAPIError(RuntimeError):
    pass


class AsyncRolloutClient:
    """Small dependency-free client with one connection per operation.

    One connection per operation lets policy and result long-polls run
    concurrently without head-of-line blocking. Calls are expected to be
    batched, so connection overhead is negligible on the local host.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout_s: float = 30.0,
        *,
        max_response_bytes: int = 8 * 1024 * 1024,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = max_response_bytes

    async def _request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        request_id = uuid.uuid4().hex
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.host, self.port, limit=self.max_response_bytes
            ),
            timeout=self.timeout_s,
        )
        try:
            message = {
                "id": request_id,
                "op": operation,
                "payload": dict(payload or {}),
            }
            writer.write(
                json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            await writer.drain()
            raw = await asyncio.wait_for(
                reader.readline(),
                timeout=self.timeout_s if timeout_s is None else timeout_s,
            )
            if not raw:
                raise RolloutAPIError("server closed the connection without a response")
            response = json.loads(raw)
            if response.get("id") != request_id:
                raise RolloutAPIError("response id does not match request id")
            if not response.get("ok"):
                error = response.get("error", {})
                raise RolloutAPIError(
                    f"{error.get('type', 'APIError')}: {error.get('message', '')}"
                )
            return response.get("result")
        finally:
            writer.close()
            await writer.wait_closed()

    async def health(self) -> dict[str, Any]:
        return await self._request("health")

    async def create_games(
        self, games: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return await self._request("create_games", {"games": list(games)})

    async def poll_policy_turns(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]:
        return await self._request(
            "poll_policy_turns",
            {"max_items": max_items, "timeout_ms": timeout_ms},
            timeout_s=self.timeout_s + timeout_ms / 1000.0,
        )

    async def submit_actions(
        self, actions: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return await self._request("submit_actions", {"actions": list(actions)})

    async def poll_results(
        self, max_items: int = 64, timeout_ms: int = 1000
    ) -> list[dict[str, Any]]:
        return await self._request(
            "poll_results",
            {"max_items": max_items, "timeout_ms": timeout_ms},
            timeout_s=self.timeout_s + timeout_ms / 1000.0,
        )

    async def get_games(
        self, game_ids: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        payload = {} if game_ids is None else {"game_ids": list(game_ids)}
        return await self._request("get_games", payload)

    async def cancel_games(self, game_ids: Iterable[str]) -> list[dict[str, Any]]:
        return await self._request("cancel_games", {"game_ids": list(game_ids)})

    async def release_games(self, game_ids: Iterable[str]) -> list[dict[str, Any]]:
        return await self._request("release_games", {"game_ids": list(game_ids)})
