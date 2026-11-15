# Frozen decoder-state probes

The decoder weights remain frozen while readouts predict board-state labels
from selected context windows and layers. Run from the project root:

```bash
python -m analysis.state_probe.pipeline --config configs/state_probe_twic_1662_1667.yaml --stage all --resume --device auto
```

The config names the indexed corpus, baseline/balanced checkpoints, completed
prediction-comparison output and human-reference PGN. All relative paths resolve
against this checkout. Stage `evaluate` requires the full completed comparison.

Stages: `index-reference`, `extract-teacher`, `train-probes`, `generate-selfplay`,
`evaluate`, `plot`, or `all`. Repeating with `--resume` reuses matching shards and
checkpoints. Use a new output directory after changing material input conditions.

## Experiment structure

### Frozen corpus and split

The implementation uses the disjoint TWIC 1662–1667 corpus already built for the checkpoint comparison. Games receive deterministic, issue-stratified splits from the SHA-256 hash of the complete indexed PGN record:

- 70% training pool;
- 15% validation pool;
- 15% held-out test pool.

The CPU-first run selects 100 training, 20 validation, and 50 test games per issue. Selection is an independent hash ranking inside each issue and split. `splits.csv` retains every corpus game's assignment, so the selected sample is auditable.

### Teacher-forced history

For each position before a recorded human move, both frozen decoder checkpoints receive the true preceding PGN history. Linear probes are trained separately for each checkpoint and representation:

- final policy representation with 32, 64, 128, and 256 token windows;
- transformer block outputs after layers 3, 6, and 9 with 256 tokens;
- the normalized, turn-adjusted layer-12 representation consumed by the policy heads.

The probes predict:

- occupancy on 64 squares;
- one of 12 piece identities on occupied squares;
- side to move;
- four castling rights;
- recorded en-passant square, including none.

Decoder weights are frozen. Activations are standardized from the training split. Training positions are sampled with inverse ply-band frequency so long games do not make middle and late positions dominate. Three seeds, 42, 43, and 44, are trained with AdamW and validation-loss early stopping. A one-hidden-layer MLP is also trained for the final 256-token state as a sensitivity check.

### Decoder self-play and cross-replay

Each held-out opening begins with eight human plies. The baseline or balanced decoder then controls both colors and selects the highest-scoring legal move deterministically for at most 120 generated plies.

Every generated history is replayed through both frozen decoders:

| Trajectory source | Baseline representation | Balanced representation |
|---|---:|---:|
| Human PGN | yes | yes |
| Baseline self-play | yes | yes |
| Balanced self-play | yes | yes |

The final 256-token linear probe trained on human histories is reused without fitting on self-play positions. This design separates checkpoint representation quality on the same history from the difficulty of states selected by each policy.

### Common-reference novelty

`index-reference` scans `twic_large_5x_classic.pgn` and constructs a Bloom filter with a target false-positive rate of `1e-4`. Position fingerprints contain piece placement, side to move, cleaned castling rights, and the legal en-passant square, hashed with BLAKE2b-128.

An absent position is definitely absent from the indexed reference. A present result can be a Bloom false positive. The metric is named **common human-reference novelty** because the exact original decoder-v12 training PGN is unavailable. It must not be interpreted as exact checkpoint training membership.

## Metrics

The primary metrics are:

- **Board NLL per square:** occupancy NLL for every square plus piece-identity NLL for occupied squares, divided by 64.
- **Occupied-state NLL:** positive occupancy NLL plus piece-identity NLL, averaged over occupied squares.
- **Normalized board Hamming error:** fraction of squares where the reconstructed 13-class state differs from the true state.

Secondary metrics are occupancy precision, recall, and F1; occupied-square piece accuracy; 13-class square accuracy; exact-board accuracy; side-to-move accuracy; exact castling-right accuracy; and en-passant accuracy. Empty-board and per-square-majority Hamming baselines are recorded.

Teacher-history checkpoint differences use balanced minus baseline on identical positions. Confidence intervals resample complete games 2,000 times. On-policy rows retain opening IDs so comparisons can resample complete openings. Results use the established absolute-ply bands and generated-ply bands `[0,10)`, `[10,20)`, `[20,40)`, `[40,80)`, and `[80,120)`.

Policy linkage joins the probe rows to the completed comparison's legal top-1 and legal-move NLL values by game, ply, and checkpoint. `policy_error_association.csv` reports these policy metrics by within-model probe-error quartile. `policy_probe_spearman.csv` reports game-level Spearman association with game-cluster bootstrap intervals. `seed_board_nll_std` records the per-position variation across the three probe seeds.

The final 256-token linear probe is streamed across the complete 15% held-out pool. Layer, context-window, and MLP sensitivity analyses use the 300 selected test games, which bounds the expensive diagnostic matrix while keeping the primary drift estimate on the full split.


## Artifacts and validation

Outputs under the configured `paths.output_dir` include deterministic splits,
compressed activation shards and metadata, per-seed readout weights, self-play
histories, per-position metrics, paired game-cluster summaries and SVG figures.
Completed stages record manifests. Large artifacts are excluded from Git.

```bash
python -m unittest tests.test_state_probe tests.test_state_probe_selfplay_inference -v
```

Readout accuracy measures recoverable information under the chosen readout,
not proof that the move head causally uses it. Reference novelty is measured
against the named PGN, not exact membership in an unavailable training corpus.
