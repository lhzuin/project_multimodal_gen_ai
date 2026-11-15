"""Deterministic differential tests against the original tournament game loop."""

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import chess
import chess.engine

from rl_stockfish_service.position_play import play_positions
from rl_stockfish_service import (
    AsyncRolloutClient,
    RolloutCoordinator,
    RolloutTCPServer,
    ServiceConfig,
)
from utils import tournament_runner as legacy
from utils.async_tournament.engines import ModelPool, OpponentClient
from utils.async_tournament.runner import (
    build_parser,
    pool_config,
    run_games,
)


class DeterministicEngine:
    def __init__(self, fail=None):
        self.fail = fail
        self.closed = False
        self.busy = False
        self.lock = threading.Lock()

    def play(self, board, limit):
        with self.lock:
            if self.busy:
                raise AssertionError("overlapping calls to a single UCI model")
            self.busy = True
        try:
            time.sleep(0.001)
            if self.fail:
                raise self.fail
            return chess.engine.PlayResult(next(iter(board.legal_moves)), None)
        finally:
            self.busy = False

    def quit(self):
        self.closed = True

    close = quit


class FakeServer:
    def __init__(self, config, failure=None, engine_failure=None):
        self.config = config
        self.failure = failure
        self.engine_failure = engine_failure
        self.active = self.max_active = self.calls = 0
        self.semaphore = asyncio.Semaphore(config.workers)
        self.requests = []

    async def health(self):
        return {
            "api_version": "1",
            "capabilities": ["play_positions"],
            "max_batch_size": 512,
            "opponent_pool": {"configuration": self.config.public_dict()},
        }

    async def play_positions(self, positions):
        if self.failure:
            raise self.failure
        self.requests.extend(positions)
        return await play_positions(self, positions, max_batch_size=512)

    def metrics(self):
        return {"configuration": self.config.public_dict()}

    async def play(self, board, limit):
        async with self.semaphore:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.active, self.max_active)
            try:
                await asyncio.sleep(0.003)
                if self.engine_failure:
                    raise self.engine_failure
                return next(iter(board.legal_moves))
            finally:
                self.active -= 1


class AsyncTournamentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.suite = Path(self.tmp.name) / "openings.json"
        self.args = build_parser().parse_args(
            [
                "--openings_json",
                str(self.suite),
                "--stockfish",
                "unused-stockfish",
                "--model_type",
                "encoder",
                "--ckpt",
                "unused-checkpoint",
                "--max_plies",
                "8",
                "--max_concurrent_games",
                "4",
                "--stockfish_workers",
                "2",
                "--model_workers",
                "1",
            ]
        )

    async def compare(self, entries, *, model_failure=None, opponent_failure=None):
        self.suite.write_text(json.dumps(entries))
        _, _, selected = legacy.select_openings(self.args)
        expected = [
            legacy.play_single_game(
                opening_id=legacy._opening_id(entry, index),
                opening_entry=entry,
                model_eng=DeterministicEngine(model_failure),
                opp_eng=DeterministicEngine(opponent_failure),
                model_is_white=white,
                limit=legacy._limit_from_args(self.args),
                max_plies=self.args.max_plies,
            )
            for index, entry in selected
            for white in (True, False)
        ]
        server = FakeServer(pool_config(self.args), engine_failure=opponent_failure)
        models = []

        def launch():
            model = DeterministicEngine(model_failure)
            models.append(model)
            return model

        summary, actual, execution = await asyncio.wait_for(
            run_games(self.args, server, launch_model=launch), 5
        )
        self.assertEqual([asdict(g) for g in actual], [asdict(g) for g in expected])
        original = legacy.build_summary(
            self.args, len(entries), len(selected), self.args.pct, expected
        )
        self.assertEqual(asdict(summary), asdict(original))
        self.assertTrue(all(model.closed for model in models))
        return summary, actual, server

    async def test_paired_games_histories_caps_and_parallel_searches(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        entries = [
            {"id": "prefix", "fen": board.fen(), "uci_prefix": ["e2e4", "e7e5"]},
            {"id": "start", "fen": chess.STARTING_FEN},
        ]
        summary, games, server = await self.compare(entries)
        self.assertEqual(summary.n_games, 4)
        self.assertEqual(summary.draws, 4)  # Unfinished '*' games score 0.5.
        self.assertGreater(server.max_active, 1)
        self.assertLessEqual(server.max_active, self.args.stockfish_workers)
        self.assertTrue(
            any(p["moves_uci"][:2] == ["e2e4", "e7e5"] for p in server.requests)
        )

    async def test_mates_draws_invalid_fen_and_truncated_prefix(self):
        board = chess.Board()
        for u in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            board.push_uci(u)
        await self.compare(
            [
                {"id": "mate", "fen": board.fen()},
                {"id": "draw", "fen": "8/8/8/8/8/6k1/8/4K3 w - - 0 42"},
                {"id": "invalid", "fen": "not a FEN"},
                {
                    "id": "prefix",
                    "fen": chess.STARTING_FEN,
                    "uci_prefix": ["e2e4", "invalid"],
                },
            ]
        )

    async def test_model_errors_are_forfeits_not_server_failures(self):
        summary, games, _ = await self.compare(
            [{"id": "a", "fen": chess.STARTING_FEN}],
            model_failure=chess.engine.EngineTerminatedError("model exited"),
        )
        self.assertEqual(summary.losses, 2)
        self.assertTrue(all(g.termination == "engine_terminated" for g in games))

    async def test_network_failure_aborts_instead_of_counting_stockfish_losses(self):
        self.suite.write_text(json.dumps([{"fen": chess.STARTING_FEN}]))
        server = FakeServer(
            pool_config(self.args), failure=ConnectionError("server lost")
        )
        with self.assertRaisesRegex(RuntimeError, "communication failed"):
            await asyncio.wait_for(
                run_games(self.args, server, launch_model=DeterministicEngine), 5
            )

    async def test_max_concurrency_one_does_not_deadlock_shared_executors(self):
        self.args.max_concurrent_games = 1
        await self.compare([{"fen": chess.STARTING_FEN}])

    async def test_more_waiting_games_than_default_executor_threads(self):
        self.args.max_concurrent_games = 40
        self.args.max_plies = 2
        await self.compare([{"fen": chess.STARTING_FEN} for _ in range(20)])

    async def test_stockfish_search_errors_keep_original_forfeit_rule(self):
        summary, games, _ = await self.compare(
            [{"fen": chess.STARTING_FEN}],
            opponent_failure=chess.engine.EngineTerminatedError("Stockfish exited"),
        )
        self.assertEqual(summary.wins, 2)
        self.assertTrue(all(g.termination == "engine_terminated" for g in games))

    async def test_two_model_workers_and_seeded_subset(self):
        self.args.model_workers = 2
        self.args.pct = 0.5
        self.args.seed = 7
        summary, games, _ = await self.compare(
            [{"id": str(i), "fen": chess.STARTING_FEN} for i in range(6)]
        )
        self.assertEqual(summary.n_openings_sampled, 3)
        self.assertEqual(summary.n_games, 6)

    async def test_cancellation_unwinds_waiting_game_threads(self):
        entered = asyncio.Event()

        class SlowServer(FakeServer):
            async def play(self, board, limit):
                entered.set()
                await asyncio.Future()

        self.suite.write_text(json.dumps([{"fen": chess.STARTING_FEN}]))
        server = SlowServer(pool_config(self.args))
        task = asyncio.create_task(
            run_games(self.args, server, launch_model=DeterministicEngine)
        )
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    async def test_server_configuration_mismatch_rejected_before_model_launch(self):
        self.suite.write_text(json.dumps([{"fen": chess.STARTING_FEN}]))
        server = FakeServer(pool_config(self.args))
        self.args.threads = 2
        with self.assertRaisesRegex(ValueError, "threads_per_engine"):
            await run_games(
                self.args, server, launch_model=lambda: self.fail("must not launch")
            )

    async def test_position_transport_retains_uncapturable_en_passant_square(self):
        server = FakeServer(pool_config(self.args))
        client = OpponentClient(server, 4)
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        try:
            await client.play(board, legacy._limit_from_args(self.args))
        finally:
            await client.close()
        self.assertEqual(server.requests[0]["start_fen"].split()[3], "e3")

    async def test_server_cache_policy_mismatch_is_rejected(self):
        self.suite.write_text(json.dumps([{"fen": chess.STARTING_FEN}]))
        server = FakeServer(pool_config(self.args))
        self.args.clear_hash_before_search = True
        with self.assertRaisesRegex(ValueError, "clear_hash_before_search"):
            await run_games(self.args, server, launch_model=lambda: self.fail("must not launch"))

    async def test_tournament_through_real_tcp_contract(self):
        self.suite.write_text(json.dumps([{"fen": chess.STARTING_FEN}]))
        pool = FakeServer(pool_config(self.args))
        coordinator = RolloutCoordinator(
            opponent_pool=pool, evaluator_pool=pool, config=ServiceConfig()
        )
        server = RolloutTCPServer(coordinator, port=0)
        self.addAsyncCleanup(server.close)
        self.addAsyncCleanup(coordinator.shutdown)
        try:
            await server.start()
        except PermissionError:
            self.skipTest("runtime sandbox forbids loopback sockets")
        client = AsyncRolloutClient(port=server.sockets[0].getsockname()[1])
        summary, games, execution = await run_games(
            self.args, client, launch_model=DeterministicEngine
        )
        self.assertEqual(summary.n_games, 2)
        self.assertEqual(coordinator.get_games(), [])
        self.assertEqual([g.model_color for g in games], ["white", "black"])

    async def test_model_startup_cancellation_reaps_eventual_subprocess(self):
        entered, release = threading.Event(), threading.Event()
        created = []

        def launch():
            entered.set()
            release.wait(2)
            engine = DeterministicEngine()
            created.append(engine)
            return engine

        pool = ModelPool(launch, 1)
        task = asyncio.create_task(pool.__aenter__())
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(created[0].closed)


class ParserTests(unittest.TestCase):
    def test_all_legacy_flags_are_available_with_original_defaults(self):
        old = {action.dest: action for action in legacy.build_parser()._actions}
        new = {action.dest: action for action in build_parser()._actions}
        for name, action in old.items():
            self.assertIn(name, new)
            if name != "model_uci_py":
                self.assertEqual(action.default, new[name].default)
