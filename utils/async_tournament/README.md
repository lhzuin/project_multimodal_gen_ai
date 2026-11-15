# Async full-game tournament runner

Run `utils/tournament_runner_async.py` (or `python -m utils.tournament_runner_async`)
from the repository root. It accepts every original `tournament_runner.py` flag,
plus explicit concurrency controls. The original sequential runner still works.
The one default-path correction is `--model_uci_py=utils/tournament_uci_engine.py`;
the old default pointed at a missing `tournament_uci_engine_patched.py` file.

```bash
.venv/bin/python -m utils.tournament_runner_async \
  --openings_json utils/opening_suite_large.json --pct 0.01 --seed 0 \
  --stockfish /path/to/stockfish \
  --limit_strength --uci_elo 2100 \
  --movetime_ms 100 --threads 1 --hash_mb 64 \
  --model_type encoder --version 2 \
  --ckpt checkpoints/rloo_encoder_balanced_v1_lichess_mixed.pt \
  --temperature 0.8 --topk 5 --greedy \
  --max_concurrent_games 8 --stockfish_workers 3 --model_workers 1 \
  --out_dir tournament_out/rloo_encoder_async_2100 --write_pgn
```

This starts an owned, in-process service adapter: Stockfish still runs as
independent CPU UCI subprocesses, but no local TCP serialization is necessary.
The server's same bounded queue and `play_positions` contract are used in either
local or remote mode. No evaluator subprocess is launched/queried for tournaments.

## Parallelism and resource budgets

| Flag | Default | Meaning |
|---|---:|---|
| `--max_concurrent_games` | 4 | Maximum simultaneous game loops/boards |
| `--stockfish_workers` | 2 | Independent Stockfish opponent processes |
| `--model_workers` | 1 | Independent complete UCI model replicas |
| `--stockfish_rpc_batch_size` | 16 | Maximum positions in one service request |
| `--queue_capacity` | 128 | Local opponent search queue capacity |
| `--threads` | 1 | Threads **per Stockfish process**, not total |
| `--hash_mb` | 256 | Hash memory **per Stockfish process**, not total |

Start with one model worker on a single GPU/MPS device. More workers each load
another model and can increase memory usage substantially without increasing
accelerator throughput. Increasing game concurrency overlaps CPU Stockfish work
with the existing model's UCI inference; it does not require model replication.
Approximate opponent CPU budget is workers × threads; hash memory is workers ×
hash_mb. Keep both within the machine's capacity, especially for time-limited runs.

Inference deliberately uses the **original UCI wrapper**, including its actual
`sample_moves` implementation, temperature, top-k, greedy behavior, promotions,
and decoder context/cropping. It does not substitute the RL legal-distribution
adapter or reinterpret the tournament sampling policy. Each UCI process receives
only one search at a time. Model inference is not tensor-batched across games;
this avoids silently changing the model's sampling implementation.

Game loops run in their own bounded executor, separate from the threads used by
async UCI workers. This prevents a deadlock where waiting games occupy all the
threads needed to complete engine searches. Histories, not just current FENs,
are sent to both engines, retaining repetition and decoder context. Concurrent
Stockfish requests are coalesced where available, with per-position response IDs.

## Scientific behavior retained

Both runners share the exact opening sampler, `play_single_game`, summary builder
and output writer. No second game adjudication implementation is introduced.

- Same seeded opening shuffle, percentage clamping and ceiling/minimum-one rule.
- Same paired colors, with output order opening / model white / model black,
  regardless of completion order.
- Same prefix-over-FEN precedence, malformed-prefix truncation, starting-position
  behavior and decoder history.
- Same `claim_draw=True` and full-game checkmate/draw behavior.
- Same cap on **new** plies after the opening prefix; `plies` remains the original
  absolute board ply count.
- Same engine-error and illegal-move forfeits. Failed searches are not retried
  inside the Stockfish pool (`restart_attempts=0`).
- Same score `(wins + 0.5 * draws) / games`; unresolved `"*"` results at the cap
  still count as 0.5, exactly as before. They are not asserted to be actual draws.
- No centipawn/horizon adjudication, resignation threshold or RL reward conversion.
- Same `summary.json`, `games.jsonl`, and optional `games.pgn` schemas/headers.

`execution.json` is an additional audit sidecar: concurrency, wall time, server
settings, and final pool statistics/engine identities. Existing consumers can
continue reading the original summary and game files unchanged.

**Same protocol is not a promise of identical sampled games.** As before, `--seed`
selects openings; it does not seed model sampling or Stockfish randomness.
Concurrent model RNG consumption, separate engine transposition tables, and
time-limited searches under CPU contention can change trajectories. Record
parallelism, use the same settings for model comparisons, avoid oversubscription,
and prefer node/depth budgets when comparing search effort. The original code
forwards time/depth/nodes together, so use `--movetime_ms 0 --nodes ...` or
`--movetime_ms 0 --depth ...` when you want no wall-time stop condition.
Keep original sampling flags fixed; adding `--greedy` changes an experiment unless
the reference run was also greedy.

The service fails closed on unsupported UCI options instead of silently ignoring
the requested strength. Server connection/protocol failures abort without writing
a new final score; they are not counted as Stockfish forfeits. Genuine search
errors still follow the original side-to-move-forfeit behavior. Use a distinct
`--out_dir` for each run: output filenames retain the original overwrite behavior.

## Remote service

Start a dedicated server with exactly matching opponent options:

```bash
.venv/bin/python -m rl_stockfish_service \
  --stockfish /path/to/stockfish \
  --play-workers 3 --play-threads 1 --play-hash-mb 64 \
  --play-movetime-ms 100 --limit-strength --uci-elo 2100 \
  --restart-attempts 0
```

Then add `--server_mode remote --server_host 127.0.0.1 --server_port 8765`
to the tournament command. `--stockfish_workers`, `--threads`, `--hash_mb`, strength,
search limits and retry policy must match the server's published settings.
The tournament cannot reconfigure remote engines silently. In remote mode
`--stockfish` records the requested binary path; the actual server and engine
identities are separately recorded in `execution.json`. Use
`--server_timeout_s` for long search budgets/queues. The local queue-capacity flag
does not reconfigure an already running remote server.

This is an additive API v1 capability. Old servers without `play_positions` and
published opponent settings are rejected before the model is launched.

## Verification

```bash
.venv/bin/python -m unittest tests.test_async_tournament \
  rl_stockfish_service.tests.test_position_play -v
```

Deterministic differential tests compare complete games and summaries with the
original loop, including terminal positions, caps, errors and invalid prefixes.
They also check bounded search overlap, UCI exclusivity and infrastructure failure.
An opt-in real encoder/UCI/Stockfish CLI + PGN smoke test uses only temporary output:

```bash
RL_STOCKFISH_BIN=/path/to/stockfish \
ASYNC_TOURNAMENT_REAL_MODELS=1 \
  .venv/bin/python -m unittest tests.test_async_tournament_integration -v
```
