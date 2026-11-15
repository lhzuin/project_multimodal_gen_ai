from __future__ import annotations

import unittest
from typing import Any

import chess

from rl_stockfish_service.config import ServiceConfig
from rl_stockfish_service.coordinator import RolloutCoordinator
from rl_stockfish_service.protocol import EngineEvaluation, GameSpec, PolicyColor


class FakePool:
    def __init__(self, evaluation_cp: int = 123) -> None:
        self.evaluation_cp = evaluation_cp
        self.play_calls = 0
        self.eval_calls = 0

    async def play(self, board: chess.Board, limit: Any = None) -> chess.Move:
        self.play_calls += 1
        return next(iter(board.legal_moves))

    async def evaluate(
        self,
        board: chess.Board,
        perspective: chess.Color,
        *,
        mate_score_cp: int,
        limit: Any = None,
    ) -> EngineEvaluation:
        self.eval_calls += 1
        return EngineEvaluation(self.evaluation_cp, None, 10, 100)

    def metrics(self) -> dict[str, Any]:
        return {"workers": 1, "queue_size": 0}


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.opponent = FakePool()
        self.evaluator = FakePool()
        self.coordinator = RolloutCoordinator(
            opponent_pool=self.opponent,
            evaluator_pool=self.evaluator,
            config=ServiceConfig(),
        )

    async def test_white_policy_reaches_horizon_and_is_evaluated(self) -> None:
        await self.coordinator.create_games(
            [
                GameSpec(
                    game_id="white-game",
                    policy_color=PolicyColor.WHITE,
                    horizon_plies=2,
                )
            ]
        )
        turns = await self.coordinator.poll_policy_turns(8, 1.0)
        self.assertEqual(len(turns), 1)
        acknowledgements = await self.coordinator.submit_actions(
            [{"request_id": turns[0]["request_id"], "move_uci": "e2e4"}]
        )
        self.assertTrue(acknowledgements[0]["accepted"])

        results = await self.coordinator.poll_results(8, 1.0)
        self.assertEqual(results[0]["termination"], "horizon")
        self.assertEqual(results[0]["reward"]["centipawns"], 123)
        self.assertEqual(results[0]["reward"]["perspective"], "policy")
        self.assertEqual(results[0]["rollout_plies"], 2)
        self.assertEqual(self.opponent.play_calls, 1)
        self.assertEqual(self.evaluator.eval_calls, 1)

    async def test_black_policy_gets_observation_after_stockfish_move(self) -> None:
        await self.coordinator.create_games(
            [
                GameSpec(
                    game_id="black-game",
                    policy_color=PolicyColor.BLACK,
                    horizon_plies=2,
                )
            ]
        )
        turns = await self.coordinator.poll_policy_turns(8, 1.0)
        self.assertEqual(turns[0]["rollout_ply"], 1)
        move = turns[0]["legal_moves_uci"][0]
        await self.coordinator.submit_actions(
            [{"request_id": turns[0]["request_id"], "move_uci": move}]
        )
        result = (await self.coordinator.poll_results(8, 1.0))[0]
        self.assertEqual(result["policy_color"], "black")
        self.assertEqual(result["rollout_plies"], 2)

    async def test_illegal_policy_move_finishes_with_penalty(self) -> None:
        await self.coordinator.create_games(
            [GameSpec(game_id="illegal", policy_color=PolicyColor.WHITE, horizon_plies=4)]
        )
        turn = (await self.coordinator.poll_policy_turns(1, 1.0))[0]
        ack = (
            await self.coordinator.submit_actions(
                [{"request_id": turn["request_id"], "move_uci": "e2e5"}]
            )
        )[0]
        self.assertFalse(ack["accepted"])
        result = (await self.coordinator.poll_results(1, 1.0))[0]
        self.assertEqual(result["termination"], "illegal_policy_move")
        self.assertEqual(result["reward"]["centipawns"], -100_000)

    async def test_cancelled_observation_is_not_returned(self) -> None:
        await self.coordinator.create_games(
            [GameSpec(game_id="cancelled", policy_color=PolicyColor.WHITE)]
        )
        await self.coordinator.wait_background()
        await self.coordinator.cancel_games(["cancelled"])
        self.assertEqual(await self.coordinator.poll_policy_turns(1, 0), [])

    async def test_duplicate_batch_is_atomic(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate game_id"):
            await self.coordinator.create_games(
                [GameSpec(game_id="same"), GameSpec(game_id="same")]
            )
        self.assertEqual(self.coordinator.get_games(), [])


if __name__ == "__main__":
    unittest.main()
