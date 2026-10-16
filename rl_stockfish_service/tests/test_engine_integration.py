from __future__ import annotations

import os
import unittest

import chess

from rl_stockfish_service.config import EnginePoolConfig, SearchLimit
from rl_stockfish_service.engine_pool import StockfishPool


class StockfishIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
        finally:
            await pool.close()


if __name__ == "__main__":
    unittest.main()
