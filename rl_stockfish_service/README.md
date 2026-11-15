# Asynchronous Stockfish rollout service

This folder is the infrastructure boundary for future reinforcement learning.
It runs many policy-versus-Stockfish rollouts concurrently and evaluates each
position after a configured number of newly played plies. It intentionally does
**not** implement GRPO, PPO, policy loading, or optimization yet.

The policy stays in the training process. The service sends it batches of
positions and legal moves; the trainer samples actions and sends them back.
This is important: a single GPU forward pass can serve many simultaneous games,
and the service never holds a stale copy of weights that are changing during RL.

The service also supports full-game evaluation via
[`utils/tournament_runner_async.py`](../utils/async_tournament/README.md). That
client retains the original tournament's UCI policy and game/scoring rules,
using the stateless `play_positions` endpoint rather than RL horizon rewards.

### Stateless opponent queries (additive API v1 capability)

`health.capabilities` advertises `play_positions`, `health.max_batch_size` gives
the request bound, and each pool's health contains its public configuration and
launched engine identities. `AsyncRolloutClient.play_positions(positions)` accepts
a list of `{id, start_fen, moves_uci, limit}` records. History moves are replayed
from `start_fen`; `limit` is an optional `{movetime_ms, depth, nodes}` mapping.
The entire batch is validated before searches are submitted. Each response has
`id`, `move_uci` and `error` (`null` on success, otherwise `{type,message}`).
No game IDs are allocated, no board is advanced for the caller, and no CP or
result is inferred. The endpoint uses the same bounded opponent queue and leaves
rollout collection unchanged. Independent requests can coexist, but use a
dedicated server/resource budget for controlled tournament measurements.
`--restart-attempts 0` disables failed-search retries for tournament forfeits;
the original RL service default remains one retry.

## Architecture

```text
 RL trainer / policy GPU
   |  poll_policy_turns (batch)       submit_actions (batch)
   v                                      |
 JSON-lines API                           |
   |                                      v
 bounded policy queue <- rollout coordinator -> bounded result queue
                              |                       ^
                              v                       |
                 opponent work queue        evaluator work queue
                    |  |  |                    |  |
                 Stockfish processes       Stockfish processes
```

- Each engine worker exclusively owns one UCI subprocess. There is no shared
  engine lock and no unsafe concurrent access to `SimpleEngine`.
- Blocking UCI calls run outside the asyncio event loop.
- Bounded queues apply backpressure rather than allowing memory to grow without
  limit.
- Opponent and evaluator pools are independent. A slow final analysis therefore
  does not consume every opponent slot.
- Engine processes are restarted after recoverable UCI failures.
- The transport is separate from the coordinator, so a future gRPC, HTTP, Redis,
  or cluster adapter can retain the same rollout contract.

## Start the service

From the repository root:

```bash
python -m rl_stockfish_service \
  --stockfish /path/to/stockfish \
  --play-workers 4 \
  --eval-workers 2 \
  --play-threads 1 \
  --eval-threads 1 \
  --play-movetime-ms 50 \
  --eval-movetime-ms 150
```

Then run the included end-to-end example in a second terminal:

```bash
python -m rl_stockfish_service.example_random_policy \
  --games 16 --horizon-plies 32 --group-size 4
```

For a strength-limited opponent, add (the evaluator remains unrestricted):

```bash
--limit-strength --uci-elo 1700
```

`--skill-level 0..20` is also available. To use depth or node limits without a
time limit, pass `--play-movetime-ms 0` or `--eval-movetime-ms 0`, then the
corresponding `--*-depth` or `--*-nodes` argument.

## Parallelism and resource budget

The principal controls are:

- `--play-workers`: simultaneous Stockfish opponent requests.
- `--eval-workers`: simultaneous terminal/horizon evaluations.
- `--play-threads`, `--eval-threads`: threads used inside each engine process.
- `--max-batch-size`: largest policy/result batch returned by one poll.
- `--queue-capacity`: maximum buffered policy, result, and engine work.

Approximate maximum Stockfish CPU demand is:

```text
play_workers * play_threads + eval_workers * eval_threads
```

Start with one thread per engine and a worker total no larger than the physical
core count. Independent one-thread engines usually give better rollout
throughput than one heavily threaded engine. Approximate engine hash memory is:

```text
play_workers * play_hash_mb + eval_workers * eval_hash_mb
```

This is in addition to model, Python, and OS memory. Conservative values matter
on the machine that previously ran out of memory.

## Version 1 wire contract

The server listens on loopback by default and accepts newline-delimited JSON over
TCP. Every request is one object:

```json
{"id":"client-generated-id","op":"operation","payload":{}}
```

Every response has the same `id` and is either:

```json
{"id":"...","ok":true,"result":{}}
```

or:

```json
{"id":"...","ok":false,"error":{"type":"ValueError","message":"..."}}
```

Use `AsyncRolloutClient` instead of constructing these messages manually.

### Operations

| Operation | Payload | Result |
|---|---|---|
| `health` | `{}` | queue, game, and worker metrics; `api_version: "1"` |
| `create_games` | `{"games": [GameSpec, ...]}` | created game summaries |
| `poll_policy_turns` | `{"max_items":64,"timeout_ms":1000}` | policy observations |
| `submit_actions` | `{"actions":[Action, ...]}` | per-action acknowledgements |
| `poll_results` | `{"max_items":64,"timeout_ms":1000}` | completed rollout results |
| `get_games` | `{}` or `{"game_ids":[...]}` | current game summaries |
| `cancel_games` | `{"game_ids":[...]}` | cancellation acknowledgements |
| `release_games` | `{"game_ids":[...]}` | removes terminal games from memory |

`GameSpec` fields:

```json
{
  "game_id": "optional-client-id",
  "start_fen": "optional FEN; standard initial position by default",
  "opening_moves_uci": ["e2e4", "e7e5"],
  "policy_color": "white | black | random",
  "horizon_plies": 32,
  "max_plies": 400,
  "metadata": {"group_id": "prompt-7", "sample_index": 0}
}
```

`horizon_plies` counts moves by both sides after the supplied opening. Metadata
is opaque to the service and is copied to observations and results. For GRPO,
create several games with the same starting state/opening and the same
`group_id`, but distinct game IDs or `sample_index` values.

A policy observation contains `request_id`, `game_id`, FEN, relative UCI move
history, legal UCI actions, policy color, ply counters, horizon, and metadata.
Submit exactly one sampled legal move for each request:

```json
{"request_id":"from-observation","move_uci":"g1f3"}
```

Request IDs are single-use. Duplicate or late actions are rejected. An illegal
action terminates that rollout with a negative mate score, which makes failures
visible instead of silently changing the policy distribution.

`submit_actions` acknowledges valid actions as soon as they are queued. The
following Stockfish move or final evaluation continues asynchronously; observe
progress through the next policy poll, result poll, or `get_games`.

## Reward meaning

At checkmate or a claimed draw, the service emits a terminal value. At the
horizon or `max_plies`, the evaluator pool analyses the final board. Every score
is from the **policy player's perspective**:

- positive centipawns favor the policy;
- negative centipawns favor Stockfish;
- mate is converted to `mate_score_cp` (default `100000`);
- `reward.centipawns` is the raw value;
- `reward.tanh = tanh(centipawns / reward_scale_cp)` is a bounded convenience
  value, not a prescribed RL reward.

Keeping the raw score lets the future trainer choose clipping, whitening,
advantages, terminal bonuses, or group-relative normalization without changing
the rollout service.

## Trainer integration sketch

```python
from rl_stockfish_service import AsyncRolloutClient

client = AsyncRolloutClient()
turns = await client.poll_policy_turns(max_items=128, timeout_ms=1000)

# One model call: encode FEN/history, mask to legal_moves_uci, then sample.
moves = policy.sample_batch(turns)
await client.submit_actions([
    {"request_id": turn["request_id"], "move_uci": move}
    for turn, move in zip(turns, moves, strict=True)
])

results = await client.poll_results(max_items=128, timeout_ms=0)
```

The eventual trainer should run policy polling/action submission and result
collection as separate async tasks so GPU inference, opponent moves, and final
evaluations overlap.

## Operational boundaries

- State and queues are in memory. A process restart loses active rollouts.
- Bind to `127.0.0.1` unless authentication/TLS is added; the protocol is not an
  internet-facing security boundary.
- Exactly-once action delivery is enforced only within one process lifetime.
- Finished games remain queryable until `release_games` is called.
- Policy inference latency is controlled by the trainer's batching strategy.
- Version 1 assumes one logical trainer consumes the policy and result queues.
  Multiple async tasks inside that trainer are supported; unrelated consumers
  would compete for work and should use a future broker/session adapter.

These are suitable semantics for a single-node first RL implementation. A later
distributed deployment should add durable rollout IDs, idempotency persistence,
authentication, telemetry, and a real message broker without changing the
policy observation/action schema.

## Search-work diagnostics

Each pool publishes `worker_pids` and cumulative `searches` statistics for play
and evaluation, including counts, node totals/means, mean depth and reported
engine/search wall times. These reveal the work done under wall-time budgets;
nominal milliseconds alone do not fix search effort under CPU contention.
Workers own independent Stockfish subprocesses, so their native searches are
not serialized by Python's GIL.

`EnginePoolConfig.clear_hash_before_search` defaults to `false`, preserving the
existing cache behavior. Set it explicitly to `true` to issue `Clear Hash`
before every query. Combined with a fixed-node limit and one thread per engine,
this removes previous-query cache history from controlled comparisons. It
changes effective search conditions and may cost throughput; it does not make
limited-strength random move selection or asynchronous policy sampling fully
deterministic. New controlled runs must be compared with baselines using the
same node and cache settings.

The proposed v5 config uses 30,000 opponent nodes, 100,000 evaluator nodes and
cold caches, rather than a wall-time opponent budget. This is an experiment,
not a calibrated Stockfish Elo setting or a guarantee of stronger training.

## Tests

The regular test suite uses fake deterministic engine pools and requires no
Stockfish process:

```bash
python -m unittest discover -s rl_stockfish_service/tests -v
```

An optional real-engine smoke test is enabled with:

```bash
RL_STOCKFISH_BIN=/path/to/stockfish \
  python -m unittest rl_stockfish_service.tests.test_engine_integration -v
```
