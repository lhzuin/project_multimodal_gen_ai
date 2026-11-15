# Data preparation

Generated corpora and caches stay outside Git. Preserve source identifiers,
filters, seeds and checksums in their generated manifests. Source sprites and
the UCI vocabulary are part of the repository.

## TWIC corpora

```bash
python -m utils.prepare_training_files --project-root . --twic-start 1484 --twic-end 1633
python -m dataset.sanity_test
```

This writes `processed_games/games.pgn`, its byte-offset index, and a sampled
position JSON in `dataset/`. The vision dataset needs sampled positions; sequence
and decoder datasets read PGNs directly. Recipes using `games_large.pgn` or
`games_xl.pgn` need the corresponding corpus, not a renamed index from another
file. Byte offsets are valid only for the exact PGN bytes that were indexed.

For an existing corpus, use `--pgn-path`. `--skip-download` limits preparation
to local inputs. `--force-rebuild-index` and `--force-rebuild-samples` explicitly
replace rebuildable caches. See each command's `--help` for output names.

[Category preparation](dataset_by_category.md) supports both streamed TWIC
collection and filtering an existing PGN. Event-derived categories are labels
from the PGN Event header, not a verified measure of every game's time control.
The category builder records excluded variants, missing ratings and download failures.

## Balanced supervision

`python -m utils.balanced_training_plan --help` describes the offline planning
stages. A frozen plan contains selected game windows, reservoir offsets and a
bit-packed per-ply loss mask. Store plans under `processed_games/balanced/`.
Both training/validation datasets must refer to the same plan before the seeded
item split. See [the runtime contract](balanced_training_integration.md).

## RL positions

`utils.dataset_rl` builds selected-position manifests from cached immutable
Lichess Parquet shards. It validates complete legal prefixes, separates games
across partitions, enforces position uniqueness, excludes configured opening
positions and applies frozen-policy confidence filtering.

```bash
python -m utils.dataset_rl.build_dataset --help
python -m utils.dataset_rl.verify_dataset --help
python -m utils.dataset_rl.augment_early --help
```

`augment_early` retains the original split rows and adds color-balanced ordinary
early prefixes from their existing source games. Shared prefixes spanning
partitions are excluded. The default additions are 4,000 training and 400 per
held-out split, between plies 12 and 23. Quotas must be nonnegative and even.
The command fails before writing if a quota cannot be filled.

```bash
python -m utils.dataset_rl.augment_early \
  --source processed_games/rl/lichess_mixed_v1 \
  --output processed_games/rl/lichess_mixed_final \
  --exclusion-openings utils/opening_suite_large.json
```

Pass `--exclusion-games tournament_out/<run>/games.jsonl` when effective recorded
starting positions also need exclusion. Exclusions cover specified positions;
they do not establish exclusion of every continuation or whole tournament game.
See the [RL dataset guide](../utils/dataset_rl/README.md) for selection details.

## Distillation

`utils.build_stockfish_dataset` builds sharded SQLite teacher data from sampled
PGN positions. `utils.merge_stockfish_runs` merges compatible shards, and
`train.distillation_encoder` consumes the result. Use each command's `--help`;
configure Stockfish search budget and worker count explicitly before generation.

## Local artifacts

| Path | Contents |
|---|---|
| `processed_games/` | PGNs, indices, balancing plans and RL split manifests |
| `twic_zips/` | Optional cached source archives |
| `distill_db/`, `merged_distill/` | Offline teacher databases |
| `checkpoints/` | Supervised/RL weights and resume state |
| `outputs/` | Evaluation databases, activation shards and generated tables |
| `tournament_out/` | Full games and tournament summaries |
| `logs/`, `wandb/` | Runtime journals |

Some older recipes expect root-level corpus caches. These remain ignored locally;
new recipes should use the directories above. A fresh clone regenerates them.
