"""Stateless position queries must never change rollout semantics."""

import unittest

import chess
import chess.engine

from rl_stockfish_service.position_play import play_positions
from rl_stockfish_service.tests.test_coordinator import FakePool


class PositionPlayTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_limits_and_no_evaluation(self):
        class Pool(FakePool):
            async def play(self, board, limit=None):
                self.history = [m.uci() for m in board.move_stack]
                self.limit = limit
                return await super().play(board, limit)

        pool = Pool()
        result = await play_positions(
            pool,
            [
                {
                    "id": "a",
                    "moves_uci": ["e2e4", "e7e5"],
                    "limit": {"depth": 2, "nodes": 100},
                }
            ],
            max_batch_size=2,
        )
        self.assertEqual(pool.history, ["e2e4", "e7e5"])
        self.assertEqual(pool.limit.depth, 2)
        self.assertEqual(pool.eval_calls, 0)
        self.assertEqual(result[0]["id"], "a")
        self.assertIsNone(result[0]["error"])

    async def test_batch_validation_is_atomic_and_size_bounded(self):
        pool = FakePool()
        for positions in (
            [{"id": "a"}, {"id": "a"}],
            [{"id": "a"}, {"id": "b", "moves_uci": ["e2e5"]}],
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        ):
            with self.assertRaises(ValueError):
                await play_positions(pool, positions, max_batch_size=2)
            self.assertEqual(pool.play_calls, 0)

    async def test_per_search_error_not_mistaken_for_game_result(self):
        class Broken(FakePool):
            async def play(self, *args):
                raise chess.engine.EngineTerminatedError("engine exited")

        result = await play_positions(Broken(), [{"id": "a"}], max_batch_size=1)
        self.assertEqual(result[0]["error"]["type"], "EngineTerminatedError")
        self.assertNotIn("result", result[0])
