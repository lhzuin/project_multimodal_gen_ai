# TWIC 1662–1667 checkpoint comparison: implementation record

## What was implemented

- `twic_category_builder.py` now has `--all-games-in-range`. It downloads every
  issue in the inclusive range, applies one shared filter, removes duplicate
  complete games, tags each retained PGN with `TWICIssue`, and writes a PGN,
  offset index, and per-issue report. The report includes archive SHA-256 hashes
  and skip counts.
- `audit_corpus.py` checks the indexed PGN's game dates and time-control
  headers and writes a JSON audit with hashes of the PGN and index.
- `evaluate.py` loads the four specified checkpoints strictly and scores the
  same recorded positions with all four models. It writes one CSV row per
  model and decision, summaries, a provenance manifest, and plots. The
  optional `--sample-games-per-issue` selects a reproducible exploratory
  subset by SHA-256 rank within each issue.
- `aggregate.py` computes decision-weighted and equal-game-weighted summaries
  and 95% intervals for balanced-minus-baseline differences by resampling
  whole games 2,000 times with a fixed seed.
- `plot_results.py` validates the summary CSVs and produces four SVG charts
  and a run-specific Markdown README.

## Frozen corpus

The owner confirmed TWIC 1662–1667 is disjoint from all four training runs.
All six archives were processed successfully. The resulting corpus has **44,049
games and 4,060,389 plies**. The builder removed 520 duplicate complete games.
All 44,049 retained PGN `Date` values parse as 2026 dates; the dates of some
games predate their TWIC issue. The issue number records provenance.

The project's `classic` category is inferred from event names. The audit found
**21,309 games (48.4%) with explicit 180- or 300-second controls** in this
category. Consequently this is a mixed-speed cohort. This limitation is
recorded in the corpus audit and figure README.

## Metrics

The target is the complete recorded UCI move, including promotion. All moves
are scored from the board **before** that recorded move, using its true prior
history. For each model and absolute-ply band `[0,20)`, `[20,40)`, `[40,60)`,
`[60,80)`, `[80,100)`, and `[100,∞)`, the primary metric is top-1 exact-move
accuracy. Secondary metrics are legal top-5 accuracy, mean legal-move negative
log likelihood, and legal perplexity `exp(mean NLL)`. The encoder CSV also
reports `(from,to)` correctness. Relative game-progress quintiles are provided
as a secondary summary. The main comparison is balanced minus baseline within
each architecture on paired decisions. Confidence intervals resample games,
because moves from the same game are dependent.

The decoder's probability is normalized across legal UCI tokens. The encoder
normalizes across legal `(from,to)` pairs and uses a conditional distribution
over legal promotion pieces. Its greedy move matches the model's promotion
fallback. Thus the NLL is comparable as a legal complete-move quantity; it is
not full-vocabulary language-model perplexity.

Agreement with one human move does not establish move quality, chess strength,
or decoder state drift. A test of internal-state recoverability, model-action
trajectories, or move-value loss requires the separate experiments described
in [the plan](twic_disjoint_checkpoint_comparison_plan.md).

## Reproduction

Run from the repository root with the project `.venv`:

```powershell
.venv/Scripts/python.exe -m utils.dataset_by_category.twic_category_builder `
  --all-games-in-range --categories classic `
  --start-week 1667 --lower-week 1662 --min-elo 1600 `
  --output-dir processed_games/eval --prefix twic_1662_1667
.venv/Scripts/python.exe -m analysis.ply_comparison.audit_corpus
.venv/Scripts/python.exe -m analysis.ply_comparison.evaluate `
  --output-dir outputs/twic_1662_1667
```

The exact field definitions, output paths, chart regeneration command, and
sample command are in [the method](twic_disjoint_comparison_method.md).
The builder refuses to replace existing frozen outputs by default; the
commands above describe a fresh build. The already generated PGN and audit
can be used directly for inference.

## Validation

The four checkpoint state dictionaries loaded with strict matching. Tests
cover the beginning of a game, an initial FEN and promotion, the 255/256-token
boundary and turn alignment, equivalence of a causal decoder pass to separate
prefixes, legal scoring, paired aggregation, corpus auditing, builder
deduplication, and chart input validation. An end-to-end CPU sample generated
all CSVs, a manifest, and the four SVGs. All 15 focused unit tests passed;
the sampled output has 24 absolute-band summary rows, 12 paired-difference
rows, and 24,228 per-move model rows (6,057 for each checkpoint).

The evaluator reports live progress every 100 processed games, including the
percentage complete, accepted and excluded counts, processing rate, and ETA.

## Exploratory run on this machine

The installed PyTorch build uses CPU here. The completed end-to-end run scored
10 hash-ranked games per issue: **60 games and 6,057 decisions**. Its output is
in `outputs/twic_1662_1667_sample_10/`. The manifest marks
`complete=true` and `full_corpus=false`; this is a pipeline check and an
exploratory result, not the full-corpus estimate.

In the first band `[0,20)`, the balanced decoder's top-1 difference was
**+0.25 percentage points** (95% game-bootstrap interval −1.17 to +1.67),
while its legal NLL was **0.0313 higher** (interval +0.0116 to +0.0534), which
is worse. At `[80,100)`, decoder top-1 was **+3.58 percentage points**
(interval +0.94 to +6.27) and legal NLL was **0.0862 lower** (interval −0.1307
to −0.0374). These values suggest an early probability penalty and later gains
in this small sample; early exact-move accuracy does not show a clear decline.
They cannot establish why a chess rollout did or did not improve. The full
44,049-game evaluation has not been run on this CPU-only environment.

The paired table, all bands, and figures are in the sampled run's
`figures/README.md`. Re-run the full command above when appropriate compute is
available; the evaluator will label that output `full_corpus=true`.
