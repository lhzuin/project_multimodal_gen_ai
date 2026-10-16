# Mixed Lichess starting positions for RLOO

This package builds a frozen, auditable dataset of starting moments, not offline
RL trajectories. Fresh policy/Stockfish continuations and rewards are generated
by the existing RLOO trainer. The original `test_puzzle_dataset.py` is preserved.

## Build and verify

From the repository root, with the joined Lichess shard already cached:

```bash
.venv/bin/python -m utils.dataset_rl.build_dataset
.venv/bin/python -m utils.dataset_rl.verify_dataset --smoke-train
```

The default destination is `processed_games/rl/lichess_mixed_v1/`. Existing output
is never overwritten. A build fails rather than silently relaxing thresholds or
publishing incomplete quotas. Publication uses a temporary sibling directory.

The default source is the cached first shard of
[`Lichess/chess-puzzles-with-games`](https://huggingface.co/datasets/Lichess/chess-puzzles-with-games),
pinned to revision `3510a01f320eaaab7b7775ebc73afcfe0b33b4c2`.
No network access or implicit download of all fourteen shards is required. To use
different already downloaded files, pass a YAML build configuration:

```yaml
local_parquet:
  - /path/to/shard.parquet
output_dir: processed_games/rl/my_dataset
train_size: 10000
validation_size: 1000
test_size: 1000
```

```bash
.venv/bin/python -m utils.dataset_rl.build_dataset --config my_build_config.yaml
```

Unspecified fields keep the defaults in `config.py`. Split sizes must be divisible
by 20 to support exact mixture and color quotas. A larger source pool can be
supplied by adding cached shards; no new Stockfish annotation sweep is necessary.

## Decisions and limitations

- Every split contains exactly **50% tactical middlegame puzzles, 30% quiet
  original-game middlegames, and 20% endgame puzzles**. Each bucket is 50/50 by the
  side to move. Labels are operational categories, not guaranteed tactical purity.
- Both players have Elo ≥1800. Puzzle groups use ratings 1200–2400, deviation
  ≤85, popularity ≥70, ≥100 attempts, and lines of 4–10 UCI plies including the
  trigger. Mate-in-one tasks are excluded. Upstream endgame themes determine the
  endgame bucket. Puzzle ratings are not interchangeable with playing Elo.
- Quiet states are sampled from complete original games attached to puzzle rows,
  not from the supplied solution. They precede 75% of the game, lie at ply 20–100,
  and are separated from the puzzle anchor by at least 11 plies. The prior move
  is quiet, there is no current check, and enough material/legal choices remain.
  Existing annotations must show a modest, stable evaluation and no last-move
  judgment. See the generated report for exact predicates.
- Draws are retained: excluding all draws would bias the general-play bucket
  against defensive and balanced play. The suggested 80% final-quarter bias is
  not used because the quiet bucket needs middlegame coverage. The existing local
  PGNs remain an alternative source; this version uses the cached joined dataset
  to reuse available engine annotations and avoid evaluating every position.
- The frozen original balanced-v1 encoder rejects positions whose maximum legal
  move probability exceeds 0.95 at temperature 1.0. This is an uncertainty gate,
  not proof that the position is learnable or that the model's preferred move is
  correct. RL sampling uses a separate temperature of 0.8.
- Source rows are chosen using bounded seeded hash reservoirs, not the first N
  matching rows. Parsing, model calls, and manifests are bounded; full Parquet
  shards and the large opening suite are never converted wholesale to Python.
- At most one position per game is retained. Normalized game IDs and hashes of
  normalized original SAN histories guard duplicates. Game hashes determine
  splits before selection; globally unique board/turn/castling/en-passant keys
  prevent repeated anchors across them. Actual histories and draw counters stay
  intact in the prompts.
- Overlap with supervised pretraining is explicitly allowed and not claimed
  absent. Online continuation does not prove later positions are unseen either.
  RL holdouts are separate from RL training, but they cannot be described as
  necessarily unseen during pretraining. Exact tournament-suite starting boards
  are excluded; full tournament source-game exclusion is not claimed.
- Quiet states come from puzzle-linked Lichess games, a source-selection bias.
  Inherited annotations have varying/unspecified engine budgets. They are only
  curation heuristics; final RL rewards come from fresh Stockfish analysis.

## Avoiding answer leakage

The [official puzzle specification](https://database.lichess.org/#puzzles) defines
`FEN` as the board BEFORE the opponent trigger, `Moves[0]` as the trigger, and
`Moves[1]` as the first solution response. The lowercase `moves` field in this
joined dataset contains the original SAN game, while `movetext` includes the
original PGN and annotations. These are not interchangeable with uppercase `Moves`.

For a puzzle, reconstruct original SAN history up to the supplied FEN; verify it,
verify/apply the original trigger, and stop. Validate the remaining puzzle line
only for legality. No supplied solution, human response after the anchor, future
evaluation, game result, or full-game movetext is passed to the model. Quiet
prompts likewise contain only their observed original history.

The runtime files use only the established `PositionPrompt` fields. Provenance
and selection statistics are written separately in `audit.jsonl`, never loaded
as training examples. The verifier independently replays all selected source
histories and checks exact trigger boundaries for all accepted puzzles.

## Artifacts and training

| File | Purpose |
|---|---|
| `train.json` | 10,000 starting positions for online RL |
| `validation.json`, `test.json` | 1,000 positions each; not loaded by training config |
| `audit.jsonl` | Source row, game identity, alignment, confidence and selection details |
| `manifest.json`, `build_config.json` | Counts, source/checkpoint/artifact hashes and reproducibility settings |
| `REPORT.md` | Actual TG2-ready methods, counts and limitations |
| `verification.json` | Replay/contract/disjointness checks and optional smoke-test status |
| `training_config.yaml` (new builds) | Generated RLOO recipe pointing to `train.json`; experiment presets live under `configs/rl/` |

For the existing `lichess_mixed_v1` dataset, the original v1 and updated v2
experiment configs have been moved to `configs/rl/`. The old experiment JSON
files were removed after checking identical parsed settings. The dataset manifest
references and verifies the relocated v1 YAML, retaining the original JSON hash
as provenance. Dataset JSON/JSONL files are unchanged.

```bash
.venv/bin/python -m train.rloo \
  --config configs/rl/rloo_encoder_lichess_mixed_v2.yaml
```

This config starts owned Stockfish pools, uses balanced encoder v1, 8 entries per
update, 4 rollouts per entry, a 10-ply horizon, temperature 0.8, LR 5e-6,
micro-batch 1, bounded CP rewards and reference KL. `randomize_policy_color=false`
ensures the model always makes the first move. One epoch is the initial experiment.
The training loop does not automatically select a best checkpoint on validation;
full-game evaluation remains a separate step.

`--smoke-train` performs one real optimizer update using temporary checkpoint
paths, never overwriting experiment outputs. It validates integration, not Elo gain.

Tests:

```bash
.venv/bin/python -m unittest discover -s utils/dataset_rl/tests -v
```
