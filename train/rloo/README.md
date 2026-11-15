# RLOO for chess policy fine-tuning

We choose RLOO for the first compute-constrained RL experiment: our pretrained
policy plays about five moves in a 10-ply rollout, we can sample several
continuations from the same position, and Stockfish supplies the final score.
RLOO provides a group baseline without a learned critic or per-position reward
standardization. This is a design choice, not evidence that it improves chess Elo
more than GRPO; that requires held-out tournaments.

## Alternatives and tradeoffs

| Method | Advantage | Limitation for this project |
|---|---|---|
| [GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1) | Critic-free grouped sampling; clipped updates allow repeated optimization passes | Standardizing each group's rewards removes differences in reward scale across positions, potentially promoting engine noise; length averaging changes the episodic objective |
| [RLOO](https://aclanthology.org/2024.acl-long.662/) | Simple leave-one-out baseline; no critic or group standard deviation; naturally handles trajectory rewards | Still needs several continuations and sensible reward scaling; this implementation makes one fresh on-policy update per batch |
| [PPO](https://arxiv.org/abs/1707.06347) | A learned value baseline can help on long games; supports multiple update epochs | Adds critic targets and tuning. A shared value head is possible, but Stockfish's value is not automatically our policy's expected return |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Queries expert moves on states the learner actually visits, addressing supervised-training distribution shift | Optimizes imitation rather than our rollout reward; needs expert-action queries beyond the current final-CP contract |
| [AggreVaTe](https://arxiv.org/abs/1406.5979) | Uses expert action costs, distinguishing harmless alternatives from blunders | More expert analysis and implementation work; Stockfish searches provide approximate expert costs |

RLOO is close to centered GRPO: its advantage is `G/(G-1)` times the
group-centered return. The important change here is removing per-group standard
deviation scaling, using trajectory REINFORCE, and recollecting after every
optimizer update. The [Dr. GRPO analysis](https://arxiv.org/abs/2503.20783) also
questions the original normalization choices. Those papers study language tasks;
our chess recommendation is an inference from the setup, not a published chess
comparison. If improving playing strength were the only goal, expert imitation
would remain a strong alternative.

## Implemented objective

For each position, choose one policy color for the whole group and collect
`G >= 2` independent sampled continuations with fixed policy weights:

```text
reward_i = tanh(final_policy_CP_i / reward_scale_cp)
sampled_KL_i = sum_policy_moves(log pi_behavior(action|state) - log pi_reference(action|state))
return_i = reward_i - kl_beta * sampled_KL_i
advantage_i = return_i - mean(return_j for j != i)
loss = -mean_groups(mean_rollouts(advantage_i * sum_policy_moves(log pi_current(action|state))))
```

The KL cost is part of the return before computing the baseline. This follows a
KL-regularized trajectory REINFORCE formulation; it is not GRPO's auxiliary exact
categorical KL loss. A sampled KL can be negative on an individual trajectory.
The fixed reference is initialized from the original checkpoint and saved on resume.
Setting `kl_beta=0` removes the reference model entirely.

Only policy decisions are loss targets. Stockfish moves and existing prompt moves
remain decoder context. Policy action log probabilities are **summed**, not
divided by each trajectory's length, including when early mates shorten a game.
Micro-batches accumulate that same group/trajectory mean before one optimizer step.
There are no PPO ratios, PPO ratio clipping, critic, or replay epochs. Global
gradient-norm clipping still applies after accumulation (default limit 1.0).
Eval mode disables
dropout during sampling and gradient replay, while gradients remain enabled.

Default bounded rewards prevent the server's large mate CP constant from
dominating an update. `reward_mode="linear"` uses `CP/reward_scale_cp` instead.
Reward transformation is deliberate and independent of the server's convenience
`reward.tanh` field. Failed/incomplete groups are dropped and reported in metrics.
An early mate before the policy acts still supplies a return to the group's
baseline, but contributes no policy loss. Equal regularized returns
produce zero advantages; no division by a small variance occurs.

## Run

For the curated Lichess dataset, follow the [dataset builder guide](../../utils/dataset_rl/README.md).
The original experiment config is `configs/rl/rloo_encoder_lichess_mixed_v1.yaml`
with the selected data, conservative training settings, and the policy moving first.

With a dedicated Stockfish v1 server running:

```bash
.venv/bin/python -m train.rloo --config configs/rl/examples/rloo_encoder.yaml
.venv/bin/python -m train.rloo --config configs/rl/examples/rloo_decoder.yaml
```

These are small integration examples reusing the existing synthetic opening
manifest. Supply your selected puzzle/key-position database before meaningful
training. Use a `database.manifest` or `database.factory` as described in the
[dataset contract](../grpo/README.md#dataset-contract). Decoder prompts require
real history from the standard starting position. Specify constructor settings
matching your checkpoint; the example v1 decoder has 12 layers. Point
`checkpoint_path` and `model_kwargs` at v3 when training that model.

Color randomization is optional in both RLOO and GRPO. The default
`randomize_policy_color=true` chooses a random color once per group. Set
`randomize_policy_color: false` inside the `rloo` YAML section (or use
`RLOOConfig(randomize_policy_color=False)`) for targeted puzzles: the policy plays
the side to move **after** replaying the prompt's history and therefore makes the
first rollout move. Every sample in the group keeps that same color.

Python usage:

```python
from rl_stockfish_service import AsyncRolloutClient
from train.rloo import RLOOConfig, RLOOTrainer

trainer = RLOOTrainer(
    database=database, server=AsyncRolloutClient(),
    checkpoint_path="checkpoints/chess_llm_decoder_balanced_v1.best.pt",
    model_type="decoder",
    model_kwargs={"tokenizer_path": "tokenizers/chess_uci_vocab.json", "n_layers": 12},
    config=RLOOConfig(ply_depth=10, temperature=0.8, micro_batch_size=1),
)
await trainer.fit(epochs=1, checkpoint_path="checkpoints/rloo.latest.pt")
```

`training_batch_size` controls entries per optimizer batch; `rollouts_per_entry`
controls group size; `max_concurrent_games`, `sampling_batch_size`, and
`micro_batch_size` independently control environment and accelerator memory.
`server.mode="local"` builds an owned engine pool using the same configuration
and API contract as the [GRPO launcher](../grpo/README.md#building-an-owned-environment).

Shared contracts, adapters, collection, lifecycle, and launch helpers now live in
`train/rl/`. `train/rloo/` owns its objective and optimizer loop; `train/grpo/`
retains its original algorithm and compatible imports. Resume checkpoints have
distinct algorithm formats: use original/exported weights to initialize another
algorithm, not its optimizer checkpoint. RLOO resumes restore model/reference,
optimizer, RNGs and dataset cursor at completed batch boundaries. Use the same
database snapshot; in-flight rollouts are recollected after a crash.

Test with `.venv/bin/python -m unittest discover -s train/rloo/tests -v`.
Optional original-model tests require `RL_STOCKFISH_BIN` and `RLOO_REAL_MODELS=1`.
Judge the experiment by held-out full games in both colors, not training CP alone.

## Persistent metrics and held-out checks

Both launchers now save append-only JSONL by default alongside the output
checkpoint (`run.latest.pt` -> `run.latest.metrics.jsonl`), even with an older
config. The classes still only expose the `on_metrics` callback: logging belongs
to the launcher, not the algorithm. See the [shared logging guide](../rl/README.md)
for optional W&B, resume behavior, and metric definitions.

The new `configs/rl/rloo_encoder_lichess_mixed_v2.yaml` points to the same data
and original checkpoint, saves JSONL, and uses distinct output filenames so it
does not overwrite the completed first experiment. It preserves LR, KL weight,
temperature, and clipping. It checkpoints every 10 batches (plus normal end),
reducing full-checkpoint writes; a crash may lose up to 9 completed batches.
This is a fresh-run config, not a request to rerun or continue the first run.

A read-only paired comparison against the original checkpoint:

```bash
.venv/bin/python -m train.rl.evaluate \
  --config configs/rl/rloo_encoder_lichess_mixed_v1.yaml \
  --database processed_games/rl/lichess_mixed_v1/validation.json \
  --candidate checkpoints/rloo_encoder_balanced_v1_lichess_mixed.pt \
  --positions 200 --rollouts 4 --seed 314159 \
  --output analysis/rloo_lichess_mixed_v1/heldout_200_seed314159.json
```

This compares bounded terminal rewards on identical held-out starting positions,
not Elo. Repeat with different seeds and use the untouched test split only after
choosing settings. The evaluator never updates models or writes checkpoints.
