from __future__ import annotations

import os
import asyncio
import unittest

import chess

from rl_stockfish_service.config import EnginePoolConfig, SearchLimit
from rl_stockfish_service.engine_pool import StockfishPool


class StockfishIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_node_cold_cache_queries_match_across_workers(self):
        executable = os.environ.get("RL_STOCKFISH_BIN")
        if not executable:
            self.skipTest("set RL_STOCKFISH_BIN for real-engine tests")
        pool = StockfishPool(EnginePoolConfig(
            executable=executable, workers=2, threads_per_engine=1,
            hash_mb_per_engine=16, default_limit=SearchLimit(nodes=2000),
            clear_hash_before_search=True,
        ), name="cold-cache")
        await pool.start()
        try:
            moves = await asyncio.gather(*(pool.play(chess.Board()) for _ in range(4)))
            self.assertEqual(len(set(moves)), 1)
            metrics = pool.metrics()
            self.assertEqual(len(set(metrics["worker_pids"].values())), 2)
            self.assertGreaterEqual(metrics["searches"]["play"]["nodes_mean"], 2000)
        finally:
            await pool.close()
    async def test_real_engine_can_play_and_evaluate(self) -> None:
        executable = os.environ.get("RL_STOCKFISH_BIN")
        if not executable:
            self.skipTest("set RL_STOCKFISH_BIN to run the real-engine integration test")
        pool = StockfishPool(
            EnginePoolConfig(
                executable=executable,
                workers=1,
                threads_per_engine=1,
                hash_mb_per_engine=16,
                queue_capacity=4,
                default_limit=SearchLimit(depth=1),
            ),
            name="integration",
        )
        await pool.start()
        try:
            board = chess.Board()
            move = await pool.play(board)
            self.assertIn(move, board.legal_moves)
            board.push(move)
            evaluation = await pool.evaluate(
                board,
                chess.WHITE,
                mate_score_cp=100_000,
            )
            self.assertIsInstance(evaluation.centipawns, int)
            metrics = pool.metrics()
            self.assertEqual(len(metrics["worker_pids"]), 1)
            self.assertGreater(metrics["searches"]["play"]["nodes_mean"], 0)
            self.assertGreater(metrics["searches"]["evaluate"]["nodes_mean"], 0)
            self.assertEqual(metrics["searches"]["play"]["count"], 1)
        finally:
            await pool.close()


if __name__ == "__main__":
    unittest.main()
