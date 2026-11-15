"""Stateless move queries over the same bounded opponent pool used by rollouts.

Full-game clients own adjudication. No horizon evaluation, CP conversion, or
automatic draw/forfeit decision is performed by this endpoint.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import chess

from .config import SearchLimit


async def play_positions(
    pool: Any, positions: Sequence[Mapping[str, Any]], *, max_batch_size: int
) -> list[dict[str, Any]]:
    if len(positions) > max_batch_size:
        raise ValueError("position batch exceeds max_batch_size")
    requests = []
    ids = set()
    # Validate the entire batch before submitting any searches.
    for position in positions:
        if not isinstance(position, Mapping):
            raise ValueError("each position must be a mapping")
        request_id = position.get("id")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            raise ValueError("position IDs must be unique nonempty strings")
        ids.add(request_id)
        board = chess.Board(position.get("start_fen", chess.STARTING_FEN))
        moves = position.get("moves_uci", [])
        if not isinstance(moves, (list, tuple)) or not all(
            isinstance(u, str) for u in moves
        ):
            raise ValueError("moves_uci must be a list of UCI strings")
        for uci in moves:
            board.push_uci(uci)
        values = position.get("limit")
        if values is not None and not isinstance(values, Mapping):
            raise ValueError("limit must be a search-limit mapping")
        limit = None if values is None else SearchLimit(**values)
        requests.append((request_id, board, limit))

    async def execute(
        request_id: str, board: chess.Board, limit: SearchLimit | None
    ) -> dict[str, Any]:
        try:
            move = await pool.play(board, limit)
            return {
                "id": request_id,
                "move_uci": None if move is None else move.uci(),
                "error": None,
            }
        except Exception as exc:
            return {
                "id": request_id,
                "move_uci": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    return list(await asyncio.gather(*(execute(*request) for request in requests)))
