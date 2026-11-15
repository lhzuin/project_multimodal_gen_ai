# Evaluation interfaces

Run commands from the project root. Use separate output directories for distinct
checkpoints, protocols and smoke tests. Saved manifests and databases retain the
original run identity; reorganizing source does not rewrite old measurements.

| Measurement | Entry point | Required inputs |
|---|---|---|
| Paired full games | `utils.tournament_runner` / `utils.tournament_runner_async` | Opening suite, policy checkpoint, Stockfish |
| Approximate tournament Elo | `utils.elo_estimator` | Grouped W/D/L and calibrated opponent ratings |
| Human-move agreement by ply | `analysis.ply_comparison.evaluate` | Indexed held-out TWIC corpus, builder report, four checkpoints |
| Decoder-state recoverability | `analysis.state_probe.pipeline` | Held-out corpus, frozen decoders, completed prediction comparison, reference corpus |
| Post-training human agreement | `analysis.decoder_alignment` | Completed baseline comparison and candidate decoder |
| Category policy agreement | `analysis.time_control_comparison` | Category corpora and three encoder checkpoints |
| Held-out RL reward | `train.rl.evaluate` | Disjoint selected-position manifest, reference/candidate weights, engines |
| Corpus repetition | `analysis.state_repetition.analyze_pgn_repetition` | PGN corpus |

## Full-game evaluation

The runners sample openings with a seeded shuffle and play paired colors. They
share game termination, caps, scoring, JSONL and PGN serialization. Use the same
search budgets, sampling flags and worker settings within a comparison.

```bash
python -m utils.tournament_runner --help
python -m utils.tournament_runner_async --help
python -m utils.elo_estimator --summary tournament_out/run_1500/summary.json \
  --summary tournament_out/run_1600/summary.json
```

Alternatively, `--matches grouped_matches.json` accepts rows of
`[opponent_elo, wins, draws, losses]`. Draws count as half a point. Elo is an
approximate logistic fit; the reported information-based interval does not
account for every opening/engine dependence. All-win/all-loss data have no
finite estimate. The tournament's historical max-plies score mapping is retained;
a capped unresolved game is not evidence of a board-rule draw.

See [the async runner](../utils/async_tournament/README.md) for resource budgets,
remote service validation and the limits of reproducibility under concurrency.

## Held-out prediction and recovery

```bash
python -m analysis.ply_comparison.evaluate --help
python -m analysis.ply_comparison.audit_corpus --help
python -m analysis.ply_comparison.resume --help
```

The reference protocol evaluates TWIC 1662-1667, numeric ratings >=1600 and the
builder's classic category. Checkpoints live in `checkpoints/`; the evaluator's
`CHECKPOINTS` map identifies the four fixed model exports. Every valid recorded
pre-move position is paired across models. Outputs include per-move predictions,
absolute/relative ply summaries, game-cluster bootstrap contrasts and provenance.

Recovery imports only verified complete games from frozen CSVs. Pair each
`--resume-csv` with its source proof using `--resume-source`; a truncated final
game can be discarded, while corrupt/interior incomplete records fail. Review
code-identity changes explicitly or start a fresh run. Outputs are not silently
merged after source changes.

## State probes

```bash
python -m analysis.state_probe.pipeline --config configs/state_probe_twic_1662_1667.yaml \
  --stage all --resume --device auto
```

The [probe guide](../analysis/state_probe/README.md) documents deterministic game
splits, context/layer captures, readout training, human/self-play cross-replay and
reference-position novelty. Decoder weights remain frozen. Recoverability from a
readout is distinct from causal use by the deployed move head.

## Candidate decoder and category policies

```bash
python -m analysis.decoder_alignment --resume --device auto
python -m analysis.time_control_comparison \
  --config configs/time_control_comparison_twic_1662_1667.yaml --resume --device auto
```

[Decoder alignment](../analysis/decoder_alignment/README.md) reuses saved baseline
rows and checks baseline-inference compatibility before evaluating the candidate.
[Category comparison](../analysis/time_control_comparison/README.md) freezes one
sample before inference, evaluates every policy on identical boards and commits
complete games transactionally in SQLite. Summary/plot stages reuse saved rows.

The category package also exposes conservative overlap checks:

```bash
python -m analysis.time_control_comparison.overlap --help
python -m analysis.time_control_comparison.sensitivity --help
```

The audit verifies saved balancing-plan fingerprints against PGNs and
reconstructs training/selection roles. Sensitivity excludes the union of confirmed
reservoir matches from the existing prediction database, retaining that database
read-only. `--matched-games-per-category` controls the supplementary matched-size
view. Use a new output directory. Historical compact tables are under `results/`;
full selections, plan files and databases are required to rerun the audit.

## Plotting

```bash
python -m utils.plots.elo_evolution --input results/training/encoder_evolution.xlsx
python -m utils.plots.plot_vit --input outputs/<run>/vision_history.csv
python -m utils.plots.plot_prob_distributions --help
```

Plot commands write to generated output folders and show windows only when
requested. The checkpoint/probe evaluators generate lightweight SVGs directly.
