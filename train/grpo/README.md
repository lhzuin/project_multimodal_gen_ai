# Chess GRPO

The first compute-constrained RL experiment now uses
[RLOO](../rloo/README.md), which documents the method comparison and decision.
GRPO remains available as a baseline. Shared implementation now lives in
`train/rl/`; the original `train.grpo` imports and launch commands remain supported.

`GRPOTrainer` fine-tunes an encoder or decoder by playing grouped, sampled
rollouts against the Stockfish service. The default horizon is **10 new plies
total**, including both players, or fewer if the game ends. Final policy-side
centipawns supply the rollout score. The training class has no Hydra or W&B
dependency; the launcher supports optional W&B and default local JSONL logging.
See the [shared instrumentation guide](../rl/README.md).

The puzzle/key-position dataset is deliberately an interface only. Selection,
database storage, and conversion from your large PGN databases remain upstream.
The small included prompt manifest tests the interface; it is not a puzzle corpus.

## Layout and algorithm boundaries

```text
PositionDatabase -> PositionPrompt -> RolloutCollector -> Trajectory
                                         |                  |
                                  Stockfish v1 API           v
                                         |          group_training_items
                                  PolicyAdapter              |
                                  (sample/replay)            v
                                              clipped_grpo_loss -> optimizer
```

| Module | Responsibility |
|---|---|
| `contracts.py` | Dataset, server, policy and trajectory interfaces |
| `config.py` | Validated dataclass configuration |
| `policies.py` | Strict checkpoint loading; differentiable encoder/decoder distributions |
| `rollouts.py` | Concurrent environment collection; immutable behavior statistics |
| `objective.py` | Group advantages, clipping and reference KL |
| `trainer.py` | Effective batches, gradient accumulation, checkpoints and resume |
| `server.py` | Owned local server lifecycle with the same API contract |
| `launch.py` | Optional argparse/YAML launcher |

To implement another RL algorithm, reuse prompts, policy distributions,
trajectories and the collector, then replace advantage construction and update.
`GRPOTrainer.training_items()` is an explicit advantage extension point; `update()`
contains the optimizer/objective boundary.

## The training objective

This follows outcome-supervised GRPO from
[DeepSeekMath, section 4.1](https://arxiv.org/html/2402.03300v3#S4.SS1):

1. Choose one policy color per selected entry and sample `G` trajectories from
   the same position with that color. By default `randomize_policy_color=true`
   chooses a random color once per group. Set it to `false` in the `grpo` YAML
   section, or use `GRPOConfig(randomize_policy_color=False)`, to play the side
   to move at the selected position after its opening history. This ensures the
   policy makes the first rollout move, which is useful for targeted puzzles.
2. Freeze the current weights during collection. Store each policy action's
   behavior log probability; no second old-policy model is needed.
3. Compute `A_i = (CP_i - group_mean) / (population_std + epsilon)`.
4. Apply the clipped policy-ratio objective to every **policy** action, with its
   trajectory's advantage. Average actions within a trajectory, trajectories
   within a group, and groups within the effective training batch.
5. Add `kl_beta * KL(current || reference)` over the legal action distribution.
   The reference stays fixed at the original checkpoint. Since chess has few
   legal moves, the implementation calculates exact categorical KL rather than
   the paper's sampled approximation.

The loss for one policy action is:

```text
ratio = exp(log_prob_current - log_prob_behavior)
loss = -min(ratio*A, clip(ratio, 1-epsilon, 1+epsilon)*A) + beta*KL
```

Raw centipawns are used, including the server's configured mate score. Groups
with identical scores have zero policy advantages; the KL term can still act.
An engine failure drops its **entire group** so a missing sample does not distort
group normalization. Groups with a trajectory containing no policy actions are
also dropped. Metrics report both dropped and zero-variance groups.

### Policy-only loss masking

The collector stores decisions only when the server requests a policy action.
This is a sparse loss mask: Stockfish moves and existing prompt moves have no
target log probability and no policy loss. Decoder histories still contain them
as causal context. Backpropagation through their input embeddings/attention is
expected; this teaches the policy how to respond to context without training it
to reproduce the opponent's decisions.

### Sampling consistency

Temperature applies identically to sampling, replay and the reference. Sampling
uses the full legal distribution. Greedy and top-k generation are not used:
changing/truncating action support between collection and replay invalidates the
GRPO ratio and complicates KL.

Models stay in `eval()` during both sampling and gradient replay, disabling
dropout without disabling gradients. A different dropout mask on replay would
change the likelihood ratio even before weights were updated.

For encoders, pair probability is combined with a sampled conditional promotion
probability. This lets the promotion head learn and avoids counting the same
from/to square four times in the legal softmax.

### Intermediate scores

Version 1 uses final CP. It already produces one differentiable loss term per
policy move. Intermediate CP is not interchangeable with a per-token advantage:
it requires a shaping rule and adds Stockfish searches. If process supervision
is added later, define a server extension for evaluation events with ply and
perspective, then build advantages from future normalized step rewards or
potential differences. The current server contract remains version 1.

## Dataset contract

Provide a finite indexable object implementing:

```python
def __len__(self) -> int: ...
def __getitem__(self, index: int) -> PositionPrompt | Mapping: ...
```

Each item represents **one already selected moment**, not an entire unsliced
game or a supervised target tensor:

```python
from train.grpo import PositionPrompt

prompt = PositionPrompt(
    entry_id="game-123-ply-42",
    # Defaults to chess.STARTING_FEN.
    moves_uci=("e2e4", "e7e5", "g1f3", "b8c6"),
    metadata={"source_game": 123, "keyness": 0.9},
)
```

The board obtained by replaying `moves_uci` from `start_fen` is the selected
starting moment. That history is sent as the server's `opening_moves_uci`;
the 10-ply horizon starts **after** this history.

- Encoder items may instead set `start_fen` to the selected board and leave
  `moves_uci` empty. Actual history is preferable when repetition matters.
- Decoder items must provide real history from `chess.STARTING_FEN`. The existing
  decoder has no board-state embedding; using an arbitrary puzzle FEN with an
  empty history would silently misrepresent its state. Such items are rejected.
- Items must be legal and nonterminal. Metadata is preserved alongside the
  trainer's `entry_id`, `group_id` and `sample_index`.
- Snapshot, length and indexing must stay fixed for checkpoint resume. Same-length
  database changes cannot be detected automatically, so keep the snapshot ID in
  your upstream dataset configuration.

## Python usage with an existing server

```python
import asyncio
from rl_stockfish_service import AsyncRolloutClient
from train.grpo import GRPOConfig, GRPOTrainer

async def main(database):
    trainer = GRPOTrainer(
        database=database,
        server=AsyncRolloutClient("127.0.0.1", 8765),
        checkpoint_path="checkpoints/chess_llm_decoder_balanced_v1.best.pt",
        model_type="decoder",
        model_kwargs={
            "tokenizer_path": "tokenizers/chess_uci_vocab.json",
            "n_layers": 12,
            "max_seq_len": 256,
        },
        config=GRPOConfig(
            training_batch_size=4,
            rollouts_per_entry=8,
            max_concurrent_games=16,
            sampling_batch_size=8,
            micro_batch_size=4,
            ply_depth=10,
            temperature=0.8,
        ),
    )
    await trainer.fit(epochs=3, checkpoint_path="checkpoints/grpo.latest.pt")
    trainer.export_weights("checkpoints/grpo.final.pt")
```

The server must be dedicated to this trainer: version 1 has shared destructive
poll queues, not per-session routing. Many games and async calls inside the
trainer are supported. Another trainer or the random-policy example must not
consume the same queues at the same time.

Pass the exact model constructor configuration used for supervised training.
Checkpoint loading is strict. Attention head count is not recoverable from
weight shapes. Existing encoder ViT weights are retained for compatible exports
but frozen; decoder auxiliary piece-head weights are also frozen.

## Building an owned environment

```python
from rl_stockfish_service import EnginePoolConfig, SearchLimit
from train.grpo import LocalStockfishServer

opponent = EnginePoolConfig(
    executable="/path/to/stockfish", workers=4, threads_per_engine=1,
    hash_mb_per_engine=64, default_limit=SearchLimit(movetime_ms=50),
)
evaluator = EnginePoolConfig(
    executable="/path/to/stockfish", workers=2, threads_per_engine=1,
    hash_mb_per_engine=64, default_limit=SearchLimit(movetime_ms=150),
)
async with LocalStockfishServer(opponent, evaluator) as server:
    trainer = GRPOTrainer(database=database, server=server, ...)
    await trainer.fit(...)
```

The local adapter preserves the network contract while avoiding serialization.
It owns the processes; an external client leaves server lifecycle with its caller.

## Batch sizing and memory

| Setting | Meaning |
|---|---|
| `training_batch_size` | Dataset entries per effective optimizer batch |
| `rollouts_per_entry` | Samples per group; at least 2 |
| `max_concurrent_games` | Maximum live games; later games start as slots free |
| `sampling_batch_size` | Maximum policy positions in one inference call |
| `micro_batch_size` | Maximum policy decisions in one backward call |
| `update_epochs` | Optimizer passes over the same collected batch |
| `ply_depth` | Newly played plies, both players, default 10 |

A batch of 4 entries × 8 rollouts × 10 plies creates about 160 policy decisions,
but it need not run 32 games or backpropagate 160 decisions simultaneously.
Gradient accumulation preserves the same trajectory/group weights regardless of
micro-batch size, including the final short batch. Rollout storage contains CPU
observations and scalar/vector statistics, never retained autograd graphs.

Keep Stockfish queue capacities at least as large as your chosen live-game limit.
The collector drains results before requesting more policy work. Accelerator
inference runs in one worker thread so an embedded server's event loop continues
to advance games while the model is busy. Policy optimization begins only after
the batch's collection has finished, keeping the samples on-policy.

Start conservatively on your Mac: `micro_batch_size=1`, sampling batches 2–4,
and 4–8 live games. A positive KL coefficient holds one frozen reference model;
set `kl_beta=0` to omit its memory/computation entirely. Full-precision AdamW is
used; mixed precision and distributed optimization are future extensions.

## Launch without Hydra

Activate the project environment, start a dedicated Stockfish server, then run:

```bash
source .venv/bin/activate
python -m train.grpo --config configs/rl/examples/grpo_encoder.yaml
# Or:
python -m train.grpo --config configs/rl/examples/grpo_decoder.yaml
```

The included configs use small synthetic opening prompts and your v1 checkpoints
for integration trials. Replace the database manifest with your selected puzzle
entries before meaningful training. A custom database adapter can be supplied as:

```json
{"database": {"factory": "my_package.my_dataset:open_database", "kwargs": {"path": "..."}}}
```

To construct Stockfish from the launcher instead, configure:

```json
{
  "server": {
    "mode": "local",
    "opponent": {
      "executable": "/path/to/stockfish",
      "workers": 4,
      "default_limit": {"movetime_ms": 50}
    },
    "evaluator": {
      "executable": "/path/to/stockfish",
      "workers": 2,
      "default_limit": {"movetime_ms": 150}
    }
  }
}
```

## Checkpoints and interruption

`save_checkpoint()` atomically stores policy/reference weights, optimizer state,
config, dataset traversal cursor, shuffle/color RNG and action-sampling RNG.
The reference remains the original policy even when resuming an updated model.
`export_weights()` produces a plain state dict for existing tournament loaders.

Set `resume_path` in the YAML launcher or call `trainer.resume(path)` after
constructing the same trainer. Resume allows a different device, micro-batch
size, sampling batch size and live-game limit. Other training settings must match.
Different batching or Stockfish timing can change exact stochastic trajectories;
fixed RNG state does not make the external engine completely deterministic.

Checkpoints are saved at completed effective-batch boundaries. Active rollouts
are in memory: after a crash, restart from the last saved boundary and recollect
the uncommitted batch. `max_updates` is an absolute optimizer-step target and must
align with `update_epochs`. A lost network connection can lose destructively
polled observations; abort, clean up, and recollect instead of continuing an
incomplete group.

## Verification

```bash
.venv/bin/python -m unittest discover -s train/grpo/tests -v
.venv/bin/python -m unittest discover -s rl_stockfish_service/tests -v
```

Optional actual-checkpoint/real-Stockfish tests:

```bash
RL_STOCKFISH_BIN=/Users/luiszuin/Downloads/stockfish/stockfish-macos-m1-apple-silicon \
GRPO_REAL_MODELS=1 \
  .venv/bin/python -m unittest train.grpo.tests.test_integration -v
```

Tests cover numerical clipping and KL, equal trajectory weights, zero-variance
and failed groups, encoder coordinates/promotions, decoder padding/crops and
gradients, rollout concurrency and color grouping, policy-only decisions,
cancellation under backpressure, gradient accumulation equivalence, reference
freezing, checkpoint resume, and both original models with a real engine.
