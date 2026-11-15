# Shared RL instrumentation

GRPO and RLOO share contracts, rollout collection, checkpoint lifecycle, and
launcher logging. Their objectives and update loops remain separate. No training
class imports W&B. W&B is a lazy, optional launcher integration.

## Logging configuration

Experiment configs live in [`configs/rl/`](../../configs/rl/README.md). Both
launchers and the evaluator require YAML (`.yaml`/`.yml`), loaded safely without
Hydra composition. Dataset manifests and metric journals remain JSON/JSONL.

Add a top-level `logging` section to either launcher's YAML config:

```yaml
logging:
  stdout: true
  jsonl_path: logs/my_experiment.metrics.jsonl
  fsync: false
  wandb:
    enabled: true
    project: chess_vision
    name: my_experiment
    mode: online
    offline_fallback: true
    init_timeout_s: 30
```

W&B is disabled unless explicitly enabled. `mode` accepts online, offline, or
disabled; omitting it respects `WANDB_MODE`. Authenticate using your existing
W&B account/environment, not credentials embedded in YAML. The implementation
uses the SDK's [run lifecycle and custom metric axis](https://docs.wandb.ai/ref/python/experiments/run/).
No models or source files are uploaded automatically, only config and metrics.

Without a logging section, stdout remains enabled and a local journal is created
beside `output_checkpoint`, replacing `.pt` with `.metrics.jsonl`. Set
`jsonl_path` to null to disable it. Set `stdout=false` for file-only logging.
Every JSONL metric record is flushed immediately. Optional `fsync=true` also
requests filesystem synchronization, at extra I/O cost. Local journal errors
stop training rather than silently lose the requested experiment record.

W&B initialization can fall back to offline mode. A later W&B log error warns,
finishes/disables that sink, and leaves stdout/JSONL active. Missing W&B also
warns and leaves local logging active. Offline directories can be uploaded later
with `wandb sync <run-directory>`; see [W&B's network guidance](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues).
No automatic background reconnection or offline-to-online merge is promised.

### Resume and provenance

Checkpoints store the logical logging `run_id`; old checkpoints without it still
load and start a new logging identity. Online W&B resumes that ID using
`resume="allow"`. Each process also gets its own `session_id`. Local journals
append rather than truncate. `<jsonl_path>.sessions.jsonl` records each session's
full launch config, timestamp, algorithm and resume status. Keep credentials out
of launch configs. Offline W&B sessions are separate syncable SDK directories;
the local journal retains the logical identity across them.

Checkpointing and logging are not one atomic transaction. A crash can cause a
replayed batch to appear twice, or a checkpointed batch to be absent from logs.
Reconcile records using `(run_id, completed_batches)`, retaining the appropriate
session. This is best-effort observability, not exactly-once
event storage. W&B uses `completed_batches` as a custom chart axis, independent
of its own history counter. Older completed runs cannot recover missing metrics
from checkpoints; saved terminal output can be imported as an explicitly partial
history, never as fabricated intermediate batches.

## Diagnostic definitions

- `grad_norm` remains the pre-clipping global L2 norm for compatibility.
- `grad_norm_pre_clip` makes that meaning explicit.
- `grad_clip_scale` is the common coefficient applied after all micro-batches.
- `grad_norm_post_clip_estimate` is the pre-norm times that coefficient.
- `grad_clipped` is 1 when clipping occurs and 0 otherwise. GRPO averages these
  diagnostics over its update epochs; RLOO has one update per collected batch.
- `rollout_seconds`, `update_seconds`, `batch_seconds` exclude checkpoint and
  logging overhead; `session_elapsed_seconds` includes overall session wall time.
- `epoch_progress` and `target_batches` describe the absolute fit target.

Clipping the accumulated vector preserves its direction and does not zero it.
With Adam, a 0.1 gradient rescaling is not equivalent to reducing the learning
rate by 10x. Do not increase the clipping threshold merely to make clipping rare.
RLOO's sampled KL is trajectory-summed over policy decisions, not a per-token
exact KL; individual samples and batch means can be negative.

RLOO additionally reports raw reward variation separately from KL-regularized
returns: `reward_zero_variance_groups`, `reward_low_spread_groups` (raw reward
range below 0.02), `mean_reward_group_spread`, `mean_reward_group_std` (population
standard deviation within each group, then averaged), and
`reward_saturation_fraction` (fraction with absolute tanh reward above 0.99;
zero for linear rewards). `mean_abs_advantage` averages over trajectories, and
`mean_abs_kl_penalty` averages the absolute trajectory KL cost. These diagnostics
do not alter the loss. Historical journals cannot reconstruct these new metrics.

RLOO now measures `parameter_update_norm` after the optimizer step and
`relative_parameter_update_norm` relative to the pre-step norm of parameters
with gradients. This is actual weight displacement, distinct from the gradient
norm. It temporarily copies active trainable weights; frozen visual/reference
weights are not copied. Optimizer behavior is unchanged. Historical journals
cannot recover a complete update-norm history from the last optimizer state.

When the server publishes search telemetry, training records cumulative
`server_opponent_nodes_mean_cumulative`, `server_opponent_depth_mean_cumulative`
and the corresponding evaluator fields. These are cumulative server means,
not values specific to the latest training batch. Search node budgets and cache
policy should be kept fixed when comparing experiments.

Set top-level `export_every_batches: 50` alongside `export_weights` to retain
intermediate policies as `<stem>.step-000050.pt`, etc. The final named export
remains launcher-owned and full resume checkpoints keep their existing cadence.
These extra weight exports permit validation-based checkpoint selection; they
do not automatically run evaluation or select a winner.

## Read-only held-out evaluation

`python -m train.rl.evaluate --help` compares exported candidate weights with the
original checkpoint on the same seeded subset of a supplied held-out manifest.
It refuses the configured training manifest and existing output files, requires
all rollout groups to complete, forces the policy to move first, and never
creates an optimizer or writes checkpoints. Sampling temperature, horizon,
opponent, evaluator and CP scaling come from the experiment config.

It reports per-position reward means, paired differences, standard error, and a
descriptive normal-approximation interval. When `metadata.game_id` is present,
uncertainty clusters all anchors from the same source game together. Without
game IDs, each position is its own cluster, preserving the earlier calculation.
Individual correlated rollouts never count as independent observations. The
summary needs at least two clusters. Small samples, repeated
testing on validation, and stochastic engine choices limit inference. A fixed
seed controls selection/policy sampling but cannot make concurrent UCI engines,
time-limited searches, or limited-strength opponent choices fully deterministic.
This is a short-horizon diagnostic, not calibrated Elo or proof of full-game gain.
