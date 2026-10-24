# Decoder State Drift and On-Policy Trajectory Analysis

## Research question

The balanced data distribution improved the encoder but did not clearly improve the decoder. This experiment tests whether that difference comes from state representation, from the trajectories selected by the policy, or from the move head after the state is already represented.

Two mechanisms motivate the analysis:

1. Early mistakes can move evaluation into positions unlike the human trajectories seen during training. This affects either architecture.
2. The decoder reconstructs the current position from a finite autoregressive move history. Errors in its chosen moves change all later inputs, so self-play can create a growing trajectory shift.

The decoder does not carry a recurrent hidden vector between turns. It recomputes hidden states from the visible move and moved-piece tokens. Here, **state drift** means declining recoverability of the board from that token history, or movement into positions whose states are harder to recover and less common in a human reference corpus.

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
- legal en-passant square, including none.

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

## Running the pipeline

The full checkpoint comparison must finish before `evaluate`, because that stage verifies that its manifest contains `complete=true` and `full_corpus=true`.

Run every stage on the best available device:

```powershell
.venv\Scripts\python.exe -m analysis.state_probe.pipeline `
  --config configs/state_probe_twic_1662_1667.yaml `
  --stage all `
  --resume
```

Force CPU execution:

```powershell
.venv\Scripts\python.exe -m analysis.state_probe.pipeline `
  --config configs/state_probe_twic_1662_1667.yaml `
  --stage all `
  --resume `
  --device cpu
```

Run one stage at a time:

```powershell
$stages = @(
  "index-reference",
  "extract-teacher",
  "train-probes",
  "generate-selfplay",
  "evaluate",
  "plot"
)
foreach ($stage in $stages) {
  .venv\Scripts\python.exe -m analysis.state_probe.pipeline `
    --config configs/state_probe_twic_1662_1667.yaml `
    --stage $stage `
    --resume
}
```

Every long stage prints completed work, percentage, processing rate, and ETA. `--resume` reuses matching split files, activation shards, probe checkpoints, complete rollouts, evaluation CSVs, and novelty-index checkpoints. Configuration and input hashes are stored in manifests. Change the output directory when changing an experimental condition materially.

## Expected outputs

The default output root is `outputs/twic_1662_1667_state_probe/`.

| Output | Meaning |
|---|---|
| `experiment_manifest.json` | Runtime, input hashes, configuration hash, device, and completed stages. |
| `splits.csv`, `split_manifest.json` | Full deterministic split and selected CPU-first sample. |
| `novelty/common_reference.bloom`, `.json` | Bloom bits, source hash, size, hash count, insertions, and estimated false-positive rate. |
| `activations/*.npz`, `meta_*.csv` | Float16 per-game hidden-state shards and matching board labels and provenance. |
| `probes/<model>/<condition>/*.pt` | Standardization values, best probe weights, seed, validation score, and epoch history. |
| `probe_baselines.json` | Empty-board and per-square-majority Hamming baselines. |
| `selfplay_trajectories.jsonl` | Opening prefix, matched human continuation, generated moves, and pre-move FENs. |
| `teacher_per_position.csv` | Full 15% test-pool final-256 linear-probe and linked policy metrics. |
| `teacher_diagnostics_per_position.csv` | Layer, context-window, MLP, and selected-test primary rows. |
| `selfplay_per_position.csv` | Cross-replayed on-policy probe metrics and novelty. |
| `bucket_summary.csv` | Mean representation metrics by trajectory, checkpoint, condition, and band. |
| `paired_deltas.csv` | Paired teacher-history checkpoint differences and game-cluster intervals. |
| `cross_replay_interactions.csv` | Both representations evaluated on each policy's generated histories. |
| `drift_interactions.csv` | Late-minus-early changes and checkpoint difference-in-differences. |
| `trajectory_gaps.csv` | Self-play-minus-human gaps in matching absolute-ply bands. |
| `novelty_summary.csv` | Common-reference novelty by trajectory and band. |
| `policy_error_association.csv` | Policy top-1 and NLL by state-probe error quartile. |
| `policy_probe_spearman.csv` | Game-level Spearman association and cluster-bootstrap interval. |
| `figures/*.svg` | Layer, context, drift, cross-replay, novelty, and policy-association figures. |
| `README.md` | Generated result report with metric definitions and interpretation rules. |

Activation NPZ files are compressed, written through temporary files, and atomically renamed. The matching CSV has the game, issue, split, ply, FEN, board labels, and metadata necessary to reproduce every probe target.

## Interpretation rules

- **Human-history degradation:** probe error grows with ply while both checkpoints see the same real histories.
- **Trajectory shift:** novelty and probe error grow after self-play leaves the human continuation.
- **State selection:** both representations perform worse on histories generated by one policy.
- **Checkpoint representation deficit:** one checkpoint's probe is worse when both checkpoints encode the same histories.
- **Move-head or calibration issue:** board information remains stable or improves, while human-move accuracy and legal NLL do not.

Probe results establish recoverability by the selected readout. They do not establish that the decoder's move head causally uses the recovered features.
