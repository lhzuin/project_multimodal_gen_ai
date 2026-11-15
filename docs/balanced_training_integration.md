# Balanced supervision runtime contract

The offline planner changes which human decisions contribute to supervised loss.
Model architectures, context sizes, optimizers and ordinary trainer entry points
remain unchanged. Existing datasets work without a plan.

## Artifacts and configuration

`python -m utils.balanced_training_plan --help` builds frozen plans containing:

| Artifact | Role |
|---|---|
| `plan.json` | Format, profile, reservoir and budget metadata |
| `selected_games.npy` | PGN byte ranges and planned human-ply windows |
| `game_ply_offsets.npy` | Prefix sums locating game/ply bits |
| `loss_mask.bin` | One supervision bit per reservoir ply |
| `report.json`, `report.md` | Source and budget audit |

`dataset.balanced_training_plan.BalancedTrainingPlan` validates the contract and
memory-maps the arrays. Add `balanced_plan_dir` to both dataset sections of a
supervised config to enable it. Missing/null leaves the ordinary dataset path.
The encoder/decoder profile and maximum planned window must match the recipe.

## Encoder alignment

A selected row specifies a PGN byte range and `[window_start, window_stop)`.
The sequence dataset returns those positions and one loss bit per position.
`CollateGameSequences` pads a separate Boolean `[B,T]` `loss_mask`. The trainer
intersects it with sequence validity, target validity and legal-target validity
before loss and accuracy reductions. It is not an attention mask.

## Decoder alignment

The full token sequence is `[BOS, move_0, ..., move_(n-1), EOS]`. A human-target
window `[a,b)` returns full-token slice `[a:b+1]`, retaining the predecessor needed
to predict move `a`. For example, `[0,3)` returns BOS plus moves 0-2; `[5,8)`
returns moves 4-7.

The `b-a` supervision bits become `b-a+1` label-mask entries: the bits followed
by `False`. After next-token shifting, label position `j` predicts human ply
`a+j`. The final false bit prevents an artificial target at the crop boundary.

`CollateLLMMoves` sets inactive move/piece labels to `-100`. It preserves input
tokens, piece tokens and attention masks: an unsupervised move remains causal
context for later predictions. A 255-ply planned window plus its predecessor
fits the 256-token decoder context.

## Split and execution

Both dataset configs refer to the same plan. The existing seeded item split
assigns `data.val_fraction` (normally 0.05) to validation. This is disjoint by
selected item, not a promise that historical and balanced validation games match.

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1
python -m train.train_chessdecoder --config-name chess_decoder_player_balanced_v1
python -m unittest tests.test_balanced_training_integration -v
```

Tests exercise actual plan files, crop alignment, padding, masked losses and
unchanged unplanned behavior. Keep the plan and source PGN together: replacing
or editing the PGN invalidates its stored byte offsets.
