# RL configurations

Run `python -m train.rloo --config <file>` or `python -m train.grpo --config <file>`
from the repository root. Files use plain YAML with duplicate-key validation:
there is no Hydra composition, interpolation or `--config-name`.

## Main recipes

| Recipe | Purpose |
|---|---|
| `rloo_encoder_lichess_mixed_final.yaml` | Encoder with early-game anchors, 20-ply horizon, four rollouts per anchor, two epochs |
| `rloo_decoder_lichess_mixed_final.yaml` | Original-v12 decoder control, 20-ply horizon, four rollouts per anchor, two epochs |
| `rloo_decoder_lichess_mixed_v5_horizon8.yaml` | Decoder with 8-ply horizon, eight rollouts per anchor, six epochs |
| `examples/` | Small encoder/decoder GRPO and RLOO integration configurations |

Versioned v1-v5 files retain the distinct learning-rate, horizon, reward, engine
and batch choices used in earlier experiments. They are reproducibility recipes,
not a chronological to-do list or a claim of playing strength. Result tables live
under `results/`. Explicit continuation/resume recipes retain their saved run
paths; inspect them before launching.

## Runtime inputs

Prepare `database.manifest` and the supervised `checkpoint_path` first. Decoder
prompts require legal full histories from their starting FEN. Production recipes
can own a local Stockfish service; examples use a separately started remote one.
Put Stockfish on `PATH`, set `STOCKFISH_EXECUTABLE`, or replace `executable` with
its path. Scientific search/batch settings are unchanged by executable relocation.
Historical-v1 regression checks use normalized portable executable names.

Paths are relative to the working directory. Use new output/checkpoint/log paths
for a fresh experiment. A restart uses an explicit `resume_path` and preserves
algorithm settings, the frozen reference and absolute epoch/update targets.
`save_every_batches` controls full checkpoints; `export_every_batches` retains
numbered inference exports. Final `export_weights` remains separate.

Logging is optional under `logging`; W&B is disabled unless enabled. See
[shared RL lifecycle and metrics](../../train/rl/README.md),
[RLOO](../../train/rloo/README.md), [GRPO](../../train/grpo/README.md), and
[dataset construction](../../utils/dataset_rl/README.md).
