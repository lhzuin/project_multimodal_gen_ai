# Chessformer

Small transformer models for human chess move prediction and play. The project
compares a board-based encoder with a causal decoder over annotated move histories,
with supervised learning, balanced supervision, Stockfish distillation and RL.
A separate vision transformer recognizes pieces on rendered boards.

The code includes training, UCI play, paired tournaments, held-out move prediction,
board-state probes and corpus audits. The vision model is evaluated separately;
the main policy experiments use explicit boards or recorded move histories.

## Start here

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Install a PyTorch build
suited to your hardware. Stockfish is a separate UCI executable: put it on `PATH`,
or set `STOCKFISH_EXECUTABLE` for RL configurations that use `executable: stockfish`.
Tournament commands accept its path through `--stockfish`.

Tests use small generated games and models. Tests requiring Stockfish or trained
weights are opt-in; see [development](docs/development.md). A fresh clone contains
source, configurations, the tokenizer, piece sprites and compact saved results.
Prepare the corpora and supply checkpoints before training or model evaluation.

## Project layout

| Directory | Responsibility |
|---|---|
| `dataset/` | PGN datasets, collators, board rendering and frozen balancing-plan reader |
| `models/` | Vision, encoder and decoder architectures; checkpoint construction |
| `train/` | Supervised and distillation trainers; shared training utilities |
| `train/rl/` | Policies, rollouts, checkpoint lifecycle, logging and held-out RL evaluation |
| `train/rloo/`, `train/grpo/` | Algorithm-specific objectives and updates |
| `rl_stockfish_service/` | Bounded Stockfish workers and local/TCP rollout API |
| `analysis/` | Move prediction, decoder-state probes and corpus diagnostics |
| `utils/` | Data preparation, UCI engines, tournaments and plotting |
| `configs/` | Training recipes and analysis configurations |
| `docs/` | Architecture, runtime contracts and usage guides |
| `results/` | Curated small tables, figures and provenance for saved experiments |

Large artifacts are local: corpora in `processed_games/`, weights in `checkpoints/`,
evaluation databases in `outputs/`, tournaments in `tournament_out/`, and journals
in `logs/` or `wandb/`. These folders are excluded from Git.

## Prepare data

Build a merged TWIC corpus, random-access index and sampled board positions:

```bash
python -m utils.prepare_training_files --project-root . --twic-start 1484 --twic-end 1633
python -m dataset.sanity_test
```

The sequence trainers consume PGNs and indices directly. Vision training also
uses a sampled-position JSON. Configure the actual paths in the selected YAML:
different recipes use different corpus sizes. See [data preparation](docs/data.md)
and [event-category corpora](docs/dataset_by_category.md).

## Train

Supervised trainers use Hydra; `--config-name` selects a recipe:

```bash
python -m train.train_vit_v2 --config-name chess_classifier_vit_v2_stats
python -m train.train_encoder_seq --config-name chess_encoder_seq_v2
python -m train.train_chessdecoder --config-name chess_decoder_player_v12
```

Balanced variants use the same trainers and an offline supervision plan:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1
python -m train.train_chessdecoder --config-name chess_decoder_player_balanced_v1
python -m train.train_chessdecoder_board --config-name chess_decoder_player_balanced_board_v1
```

The auxiliary-board variant adds reconstruction targets during training and
uses the same move policy at inference. Guides:
[training and checkpoints](docs/training.md),
[balanced supervision](docs/balanced_training_integration.md),
[auxiliary board objective](docs/decoder_board_auxiliary.md).

RL launchers use plain YAML through `--config`, independently of Hydra:

```bash
python -m train.rloo --config configs/rl/rloo_decoder_lichess_mixed_v5_horizon8.yaml
```

Prepare the selected-position dataset and supervised checkpoint first. See
[RL configurations](configs/rl/README.md), [RLOO](train/rloo/README.md),
[GRPO](train/grpo/README.md), and the [Stockfish service](rl_stockfish_service/README.md).

## Evaluate and play

Create an opening suite, then run paired full games:

```bash
python -m utils.generate_opening_book --pgn processed_games/games_large.pgn \
  --out utils/opening_suite_large.json --full_moves 8
python -m utils.tournament_runner_async \
  --openings_json utils/opening_suite_large.json --pct 0.004 --seed 0 \
  --stockfish /path/to/stockfish --limit_strength --uci_elo 1500 --movetime_ms 10 \
  --model_type decoder --ckpt checkpoints/chess_llm_decoder_v12.pt \
  --tokenizer_path tokenizers/chess_uci_vocab.json --n_layers 12 --greedy \
  --out_dir tournament_out/decoder_v12_1500 --write_pgn
```

The sequential runner is `utils.tournament_runner`. Both share the same game
loop and adjudication. See [asynchronous tournaments](utils/async_tournament/README.md)
for concurrency budgets and protocol details. Register `run_uci.sh` or
`run_uci.ps1` in a chess GUI, or invoke `python -m utils.uci_engine --help` directly.

Offline evaluation covers paired checkpoint predictions, decoder-state probes,
post-RL human-move agreement and time-control policy comparisons. The
[evaluation guide](docs/evaluation.md) maps each command to its inputs and outputs.

## Maintain and extend

Read [the architecture](docs/architecture.md) for model/data contracts and
[development](docs/development.md) for tests, submission rules and contribution
conventions. Existing recipe names and checkpoint parameter names are retained so
older experiments can still be loaded. Intermediate YAMLs record experiment
settings; the [config index](configs/README.md) identifies the main entry points.

Saved aggregate outputs live in [results](results/README.md). Full run artifacts
remain local and can be regenerated with the corresponding evaluation modules.
