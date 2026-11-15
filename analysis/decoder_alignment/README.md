# Decoder human-move agreement after post-training

This evaluator compares the completed RLOO v5 decoder with the saved original-v12
predictions on the **same 44,049 TWIC games / 4,060,389 recorded human decisions**
used for the original/balanced checkpoint comparison. No download or new sample
selection is needed. The source PGN, index, tokenizer, checkpoints, and baseline
CSV are read-only inputs.

Run from the project root:

```bash
python -u -m analysis.decoder_alignment --stage all --resume --device auto
```

The default candidate is
`checkpoints/rloo_decoder_v12_lichess_mixed_v5_horizon8.pt` under the project root.
Inference uses this checkout's decoder implementation and checks its compatibility with saved baseline predictions. The command uses CUDA when available, otherwise
MPS, otherwise CPU. Run this evaluation separately from timed tournaments.

## What is reused

1. The completed `outputs/twic_1662_1667/manifest.json` must identify the same
   corpus, index, tokenizer, and original-v12 checkpoint by SHA-256.
2. The evaluator streams `per_move.csv` once and imports only `decoder_baseline`
   rows. It does not rerun the encoder, balanced decoder, or full original decoder.
3. One game per represented TWIC issue checks the original model's saved greedy
   decisions, top-5 results, and likelihoods against current inference. These six
   compatibility games are not a new evaluation sample.
4. Every candidate game's positions are checked against the saved baseline:
   game/ply IDs, complete FEN, human move, legal count, metadata, and context.
   All candidate inference uses the original PGN histories, not generated moves.

The full CSV checksum is checked when opening or resuming the run. Preparation
can therefore spend some time reading the existing large file before progress
appears. Imported baseline games are retained across interruptions; a resumed
partial import rescans the CSV while skipping already committed games.

## Metrics and interpretation

The main comparison reports complete legal UCI top-1 agreement, top-5 agreement,
and legal-move NLL overall and in the original six absolute-ply bands. Inputs
contain up to 256 preceding-history tokens, including the same moved-piece and
turn annotations as the earlier evaluator. Greedy decisions and legal top-5
use the original scoring function. NLL normalizes over every legal complete
move at temperature one, without sampling top-k truncation.

One causal forward pass scores all prefixes through ply 255. Beyond that point,
each decision gets its own truncated 256-token window, with positional indices
and turn alignment restarted exactly as before. `--batch-size` (default 32)
controls these late-window batches.

Primary averages weight positions equally, matching the original experiment.
Whole-game bootstrap replicates resample paired candidate/baseline sufficient
sums, preserving every decision from each selected game. They estimate the
ratio of sampled totals, not an unweighted average of per-game changes. The
default is 2,000 replicates with seed 42. Accuracy changes above zero favor the
candidate; NLL changes below zero favor it. Equal-game averages and a nonforced
view are supplementary. Disagreement, beneficial switches (baseline incorrect,
candidate correct), and harmful switches show what produces the accuracy change.

An increase would show that stronger tournament play accompanies better human
move prediction on these recorded positions. A decrease would quantify a
trade-off between the two objectives. A confidence interval spanning zero leaves
the direction unresolved. These measurements concern agreement with the sampled
human population, not individual style or behavior on generated trajectories.
Intervals condition on the checkpoints and corpus; they do not cover training
seeds or all event/player/opening dependence.

## Outputs and progress

Results go to `outputs/twic_1662_1667_decoder_rloo_v5_alignment/` in the project root:

| File | Purpose |
|---|---|
| `evaluation.log` | Timestamped progress and ETA, appended across sessions. |
| `progress.json` | Current stage and counts of imported/completed games and positions. |
| `results.sqlite` | Baseline and candidate per-position predictions, stored transactionally. |
| `manifest.json` | Input/code hashes, selected IDs, and exact comparison settings. |
| `summary.csv` | Baseline/candidate metrics, including equal-game diagnostics. |
| `paired_deltas.csv` | Paired changes and confidence intervals; candidate minus baseline. |
| `summary.json` | Machine-readable results and the compatibility-check receipt. |
| `REPORT.md` | Concise results and interpretation. |

From the project root, follow the log with:

```bash
tail -f outputs/twic_1662_1667_decoder_rloo_v5_alignment/evaluation.log
```

## Recovery and separate stages

Repeat the same command with `--resume` after interruption. Each baseline or
candidate game is committed as a unit; an interrupted candidate game is rerun,
and complete games are skipped. An OS lock prevents two evaluators from sharing
one output directory. Changes to data, checkpoint, tokenizer, inference code,
or inference runtime require a new directory, preventing mixed scientific runs.

`--stage prepare` imports the saved baseline without model inference.
`--stage evaluate` prepares if needed and evaluates missing candidate games.
`--stage summarize` rebuilds tables/report from the completed database without
loading models. Bootstrap count/seed may change when summarizing.
`--stage export` optionally writes a joined `per_move.csv`; the SQLite database
already contains the individual predictions, so this extra large export is not
needed for the report.

A smoke test must use a separate directory:

```bash
python -u -m analysis.decoder_alignment \
  --output-dir outputs/decoder_alignment_smoke \
  --max-games 8 --bootstrap 100 --device auto --resume
```

Run this command from the project root. Its outputs are explicitly marked as a
restricted smoke test and cannot be resumed as a full-corpus run. For another
v12-family decoder checkpoint, supply `--checkpoint`, `--candidate-id`, and a
fresh `--output-dir`; explicit relative paths are relative to the working directory.

## Verification

From the project root:

```bash
python -m unittest \
  tests.test_decoder_alignment \
  tests.test_ply_comparison_evaluate \
  tests.test_ply_comparison_aggregate \
  tests.test_ply_comparison_resume
```

Tests cover prefix/cache equivalence and the 256-token boundary, promotion and
position matching, variable-length game bootstrap weighting, interrupted game
rollback, partial-run recovery, immutable checkpoint identities, baseline reuse,
output protection, and concurrent-run rejection.
