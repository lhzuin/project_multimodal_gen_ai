# TG2 balanced-training runtime integration

This document describes the runtime contract between the offline balancing planner and the existing supervised encoder-sequence / causal-decoder training pipelines.

The design goal is deliberately narrow: **change the distribution of supervised human chess decisions while preserving the existing models, optimizers, context limits, batching rules, and training entry points.** Legacy configs remain valid and keep their previous behavior.

## 1. High-level contract

The offline planner (`python -m utils.balanced_training_plan`) emits a frozen directory containing:

- `plan.json`: format/profile metadata and source/budget information;
- `selected_games.npy`: one row per dataset item, including exact PGN byte offsets and the planned human-ply window;
- `game_ply_offsets.npy`: global prefix sums used to locate a `(game_id, ply)` bit;
- `loss_mask.bin`: one bit per valid reservoir ply; `1` means that the human move contributes supervised loss;
- `report.json` / `report.md`: audit information.

At runtime, `dataset.balanced_training_plan.BalancedTrainingPlan` is the single reader for this contract. It validates the format/profile, lazily memory-maps the artifacts, checks the reservoir file size when the report provides it, and reconstructs local loss masks without loading the full plan into RAM.

The heavy statistical logic stays in preprocessing. The training code only consumes the frozen decisions.

## 2. Configuration switch and backward compatibility

Balanced training is enabled **only** when a dataset config contains:

```yaml
balanced_plan_dir: ${hydra:runtime.cwd}/processed_games/balanced/<plan_directory>
```

If `balanced_plan_dir` is absent or `null`, both datasets use their legacy code paths:

- `ChessGameSequenceDataset` loads the ordinary PGN index and samples its random window as before;
- `PGNMoveDataset` loads the ordinary PGN index and samples token crops as before;
- collators receive no `loss_mask`, so loss construction remains unchanged.

No separate trainer executable is required. This keeps old `chess_encoder_seq_v2.yaml`, `chess_decoder_player_v12.yaml`, and other existing configs usable without modification.

## 3. Encoder-sequence semantics

For a planned encoder item, `selected_games.npy` specifies:

```text
[pgn_start, pgn_end)
[window_start, window_stop)
```

The dataset reads exactly those PGN bytes and returns exactly that human-ply window. The plan's local loss mask has one Boolean value per returned position.

`CollateGameSequences` pads this as a separate `[B,T]` Boolean tensor named `loss_mask`. It is intentionally distinct from `attn_mask`.

The encoder trainer uses

```text
valid = padding/sequence validity
        AND valid move indices
        AND loss_mask (when present)
        AND legal-target validity
```

before cross entropy and accuracy calculation.

Thus a zero plan bit removes only that position's supervised contribution. The board state can still be present in the batch, and no attention/padding semantics are overloaded.

## 4. Decoder semantics and the off-by-one contract

The tokenizer emits

```text
[BOS, move_0, move_1, ..., move_(n-1), EOS]
```

Suppose the planner selects human target plies `[a,b)`. To predict move `a`, the causal model needs one predecessor/context token. Therefore the balanced decoder dataset returns full-token slice

```text
[a : b + 1]
```

which has exactly `(b-a)+1` tokens.

Examples:

```text
planned plies [0,3) -> [BOS, move_0, move_1, move_2]
planned plies [5,8) -> [move_4, move_5, move_6, move_7]
```

After standard next-token shifting, label position `j` predicts planned human ply `a+j`. The plan's `L=b-a` per-ply bits are therefore converted to a decoder label mask of length `L+1`:

```text
[ply_mask_0, ..., ply_mask_(L-1), False]
```

The final `False` prevents an artificial PAD/EOS-like target from being introduced by the planned crop.

`CollateLLMMoves` applies this mask only to `labels` and `piece_labels`:

```text
loss bit = 0 -> labels = -100, piece_labels = -100
```

It does **not** change `input_ids`, `piece_input_ids`, or `attention_mask`. Consequently, moves whose own prediction is not supervised remain visible as causal history for later supervised moves. This is the key requirement for preserving sequential state/tactical reasoning.

The balanced decoder config keeps `block_size=256`. Plans generated with `--max-window-plies 255` fit exactly: at most 255 human targets plus one predecessor/context token.

## 5. Fresh 5% validation split

The project deliberately continues using the existing automatic split:

```yaml
data:
  val_fraction: 0.05
```

No historical validation PGN is required and no `--exclude-pgn` is required for this protocol. The trainer shuffles the **selected planned items** using the configured seed and assigns 5% to validation exactly as it does for legacy datasets.

This means TG1 and TG2 validation games are not guaranteed to be identical. The intended final comparison remains online Elo/tournament play; validation continues to serve the existing training/early-stopping machinery.

Both `dataset_train` and `dataset_val` in the balanced configs point to the same frozen plan so the split is disjoint at the selected-item level.

## 6. Current balanced configs

### Encoder

`configs/chess_encoder_seq_balanced_v1.yaml` is based on `chess_encoder_seq_v2.yaml` and changes only the data source / planned-window interface plus the experiment name.

Expected artifacts:

```text
processed_games/by_category/twic_large_5x_classic.pgn
processed_games/balanced/encoder_large_classic_l050/
```

The supplied encoder report selected exactly 186,242 items and 11,139,816 supervised plies. With the unchanged 5% split this produces 176,930 training items, exactly matching the historical game-count split. With batch size 2 and gradient accumulation 4, a complete epoch has 88,465 micro-batches and 22,116 full optimizer updates (the existing trainer intentionally does not step a final partial accumulation).

### Decoder

`configs/chess_decoder_player_balanced_v1.yaml` is based on `chess_decoder_player_v12.yaml` and expects the plan generated from:

```text
processed_games/by_category/twic_all_classic.pgn
processed_games/balanced/decoder_xxl_classic_l050/
```

The preprocessing command used `--max-window-plies 255`, matching the unchanged 256-token model/data context.

With `--target-games 1352454`, the automatic 5% split produces 1,284,831 training items. With batch size 32 and gradient accumulation 1, this is 40,151 optimizer steps per full epoch. Compared with a nominal 1,352,515-item XXL corpus (40,153 steps/epoch after the same split), the difference is only two steps per epoch (~0.005%).

## 7. Commands

Run from the repository root.

Balanced encoder:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1
```

Balanced decoder:

```bash
python -m train.train_chessdecoder --config-name chess_decoder_player_balanced_v1
```

Legacy behavior remains available with the existing defaults:

```bash
python -m train.train_encoder_seq
python -m train.train_chessdecoder
```

or explicitly:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_v2
python -m train.train_chessdecoder --config-name chess_decoder_player_v12
```

## 8. Fail-fast checks

Balanced mode raises an error rather than silently training on misaligned data when any of the following occurs:

- wrong plan format/profile;
- reservoir byte size differs from the source used to construct the plan (when available in `report.json`);
- a planned PGN record no longer parses;
- the planned ply count differs from the parsed game;
- planned encoder window exceeds `max_seq_len`;
- planned decoder window exceeds `block_size`;
- local bit-mask active count disagrees with `selected_games.npy`;
- decoder token count and label-mask length disagree;
- a collate batch mixes planned and legacy items;
- collate `max_len` would truncate a planned decoder item.

Failing loudly is preferable here because an off-by-one error could otherwise produce a plausible-looking but scientifically invalid run.

## 9. Performance characteristics

The runtime integration is intentionally light:

- no state-frequency table is loaded during training;
- no large reservoir JSON index is loaded in balanced mode;
- `selected_games.npy`, `game_ply_offsets.npy`, and `loss_mask.bin` are memory-mapped lazily;
- each item performs only a small vectorized bit lookup for its planned window;
- PGN parsing/model input construction remains the dominant per-item work.

The planner's statistical choices are therefore paid once offline rather than repeatedly inside DataLoader workers.

## 10. Tests before launching a long run

Run:

```bash
pytest -q tests/test_balanced_training_integration.py
```

The tests cover the frozen-plan reader, bit-packed mask recovery, decoder ply/token alignment, encoder mask collation, decoder label masking, causal-context preservation, mixed-mode failure, and explicit legacy-collator compatibility.

Then run the project's normal sanity checks. For a final GPU run, it is also worth fetching one batch from each balanced config and confirming:

```text
sum(loss_mask) > 0
encoder: loss_mask.shape == attn_mask.shape
decoder: masked labels are -100 while input_ids remain populated
```
