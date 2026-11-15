# Classical, rapid and blitz policy comparison

This package implements the held-out category-policy comparison. It samples games independently of model outcomes, evaluates all three frozen encoders on identical pre-move boards, and reports complete-move agreement, human-move NLL and greedy policy disagreement. It reuses the existing TWIC downloader, Event classifier, PGN offsets, board renderer and encoder class.

## Run on this project

Run from the project root. Inputs and generated outputs resolve against that checkout.

First download the complete candidate range once:

```bash
python -u -m utils.dataset_by_category.twic_category_builder \
  --all-games-in-range \
  --categories blitz rapid classic \
  --start-week 1667 \
  --lower-week 1662 \
  --min-elo 1600 \
  --output-dir processed_games/eval/twic_1662_1667_categories \
  --prefix twic_1662_1667_categories
```

Then launch the complete analysis:

```bash
python -u -m analysis.time_control_comparison \
  --config configs/time_control_comparison_twic_1662_1667.yaml \
  --stage all \
  --resume \
  --device auto
```

`auto` selects CUDA, then Apple MPS, then CPU. Use `--device cpu` or `--device cuda` explicitly when appropriate.

The default checkpoint paths are the validation-selected `.best.pt` exports for balanced classical v1, rapid v1 and blitz v1. Strict state-dict loading enforces the common encoder architecture. Use fixed selected weights for the experiment; their content hashes are checked at startup, on resume and after inference.

## Data selection

- All six TWIC issues must have been processed successfully. The preparer checks the builder report, configured filters, source counts, indexes and content hashes. The previous classic-only corpus cannot substitute for this three-category download.
- Classification is exactly `get_event_category`: case-insensitive substring matching, blitz before rapid, all other Event names classic. `TimeControl` is retained for cohort description and does not override training labels.
- Both player ratings must be numeric and at least 1600. As in the training builder, every nonempty Variant header is excluded. Empty, malformed, illegal/null-move and invalid-initial-board games are rejected. Standard chess games with an explicit initial FEN are supported.
- Canonical identity uses normalized initial FEN and the complete UCI mainline, independent of player/event/date headers. Sampling removes duplicates across the candidates; identities appearing with conflicting categories are excluded from every category.
- Within each category, games are sorted by SHA-256 of `seed:canonical_identity`, then the first 1,000 are retained without replacement. This gives a deterministic hash-ranked sample across the whole requested range, without artificially equalizing issues, ratings or events. Selection seed is 42.
- The PGN snapshot and selection manifest are frozen before inference. The full game is the sampling unit. Every valid recorded pre-move position is scored; no training loss mask or subsequent game moves are exposed to the policy.
- `corpus_audit.json` records actual eligible counts and exclusions. The default requires all three 1,000-game quotas. Insufficient supply raises an error before inference; it does not draw older weeks or duplicate games. An explicit `--allow-shortfall` accepts a smaller collection and records its actual size, if that later becomes the chosen protocol. A category with zero eligible games always fails.
- Recent TWIC source separation is the default overlap basis, recorded in the manifest. Optional `exclude_pgns` in the YAML scans supplied complete training/selection PGNs and removes canonical-game matches before selection. Scanning entire source PGNs is conservative if only some games contributed to training.

The existing downloader protects prior outputs and records network failures. It does not resume an interrupted download in place. If acquisition is incomplete, rerun that downloader with `--overwrite` against this dedicated **candidate corpus** directory before freezing the sample. Completed inference is resumed through the analysis command, not the downloader. Changing frozen inputs requires a new analysis output directory.

## Inference and metrics

The encoder receives perfect board piece IDs ordered rank 8 to rank 1, side to move, four castling flags and the en-passant square. It receives the board before the human target move. Training windows do not imply that the deployed board encoder consumes a history of 64 positions. All policies receive the same board tensors; models are frozen, in evaluation/inference mode, with float32 forward passes.

Greedy top-1 matches `ChessEncoderPlayerV2.sample_moves`: choose the highest-scoring legal origin/destination pair, resolve tied pairs by ascending flattened index, then select the promotion head's preferred class. The deployed class-zero promotion fallback is retained. This hierarchical decision can differ from the joint complete-move argmax; the evaluator does not silently change it.

For NLL, each distinct legal square pair appears once in a stable log-softmax. A promotion pair distributes its mass among legal promotion choices by a conditional log-softmax over knight, bishop, rook and queen classes; the no-promotion class is excluded from that conditional distribution. The resulting complete legal-move probabilities sum to one. No temperature scaling, top-k truncation or probability floor is applied. Scoring uses float64 CPU log-space arithmetic after float32 inference. Invalid targets and nonfinite legal logits are errors. Supplementary top-5 uses the joint complete-move probability ranking.

Primary agreement and NLL are position-weighted within each test category. Disagreement uses the same greedy predictions; forced positions are included in the primary analysis and separately excluded in a supplementary table. CSVs also retain ply-band, color and promotion strata, plus equal-game summaries. Category-macro results weight the three category scores equally.

Uncertainty uses 2,000 percentile-bootstrap replicates, resampling complete source games within each category. Every policy and contrast shares the same sampled game indices. Category-macro and own-versus-classical specialization interactions combine independent within-category resamples. Secondary strata resample contributing games. Intervals condition on the frozen checkpoints and do not capture training-seed variation or all event/player dependence. Secondary comparisons are exploratory, without a multiple-comparison correction.

## Resume and stages

Results are saved in `results.sqlite`. A transaction commits all positions and all three policies for one game together. Ctrl-C or process termination can lose only the current unfinished game; repeating `--stage all --resume` reuses committed games. An OS lock prevents two processes from writing the same run concurrently. SQLite uses foreign keys, WAL journaling and full synchronization.

Resume checks sample, checkpoint and inference-code hashes, batch size and the Python/PyTorch/NumPy/python-chess/device identity. A changed identity fails rather than mixing different inference attempts. Changing bootstrap settings does not require repeating inference; regenerate summaries. Runtime/device changes require a separate inference run under the current strict identity policy.

Available stages:

| Stage | Work |
|---|---|
| `prepare` | Validate the complete candidate collection and freeze the selected games. No checkpoint loading. |
| `evaluate` | Prepare/validate selection, resume inference, export per-move results when complete. |
| `summarize` | Regenerate all CSV summaries, game metadata, Markdown tables and SVG figures from the completed database. No checkpoints or accelerator needed. |
| `plot` | Regenerate Markdown tables and SVG matrices from completed summary CSVs. |
| `all` | Run the above sequence. |

For a separate small smoke run after downloading, append:

```text
--output-dir outputs/twic_1662_1667_time_controls_smoke --games-per-category 2
```

`--max-new-games 1` pauses after one newly committed game and can be used to check recovery in that separate smoke run. Remove it when resuming to finish. Reports for the complete sample are withheld until all selected games are present.

## Outputs

The configured result folder is the main project's `outputs/twic_1662_1667_time_controls/`:

| Artifact | Purpose |
|---|---|
| `README.md` | Prediction and disagreement matrices, paired contrasts and interpretation scope. |
| `results.sqlite` | Authoritative per-position predictions and atomic game-completion state. |
| `per_move.csv` | Predictions for every policy/position, full UCI targets, human/predicted log-probabilities, FEN, legal count, color, ply and promotion indicators. |
| `games.csv`, `corpus_profile.json` | Source issues, events, players/ratings, dates, time controls, game lengths and forced-position counts. |
| `prediction_summary.csv` | Agreement/NLL and supplementary top-5, entropy, pair accuracy; intervals and equal-game estimates. |
| `paired_deltas.csv` | Direct paired policy contrasts within each test category and stratum. |
| `disagreement.csv` | All three policy pairs, including primary rapid–classic/blitz–classic and non-forced-position results. |
| `category_macro.csv` | Equal-category overall agreement and NLL. |
| `specialization.csv` | Own-versus-classical-category difference in each category policy's advantage over the classical policy. |
| `selection_manifest.json`, `selected_games.pgn` | Immutable sample and canonical identities, original headers, source hashes, offsets and selection settings. |
| `run_manifest.json`, `summary_manifest.json` | Inference and aggregation provenance, settings and completion state. |
| `figures/*.svg` | Dependency-free accuracy, NLL and disagreement matrices. |

CSV rates are fractions, not percentages. NLL is nats per complete move. The Markdown report converts agreement/disagreement to percentages and agreement differences to percentage points. Per-move CSVs and SQLite remain available for later rating matching, event-level sensitivity analysis or alternative figures without rerunning the models.

## Maintenance and validation

`corpus.py` owns selection; `inference.py` owns model scoring; `storage.py` owns atomic persistence; `aggregate.py` owns statistical estimates; `report.py` owns presentation; `__main__.py` only coordinates stages. Shared configuration and artifact helpers live in `common.py`. Figures use SVG directly and need no additional plotting installation.

Run the focused regression suite from the project root:

```bash
python -m unittest tests.test_time_control_comparison
```

The tests cover promotions, greedy parity, ties, legal probability normalization, underflow, forced moves, castling/en-passant, pre-move inputs, frozen sampling, shortfalls, overlap exclusion, transactional rollback, interruption/resume, code/checkpoint changes, paired weighting/bootstrap and the complete CLI/reporting path. No tournament is launched by this package.

## Overlap audit and sensitivity

`python -m analysis.time_control_comparison.overlap --help` checks selected test
identities against saved policy reservoirs and reconstructs training/selection
roles from saved configurations. It verifies checkpoint hashes, plan arrays and
actual source PGN records. Supply full local artifacts; it writes a new audit.

`python -m analysis.time_control_comparison.sensitivity --help` excludes the
union of confirmed reservoir matches from saved predictions using read-only
SQLite input, then regenerates summaries in a new directory. The supplementary
matched-size view uses `--matched-games-per-category` (default 481).
