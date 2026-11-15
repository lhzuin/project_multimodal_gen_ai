# Auxiliary decoder board reconstruction

`ChessDecoderBoardPlayer` extends the ordinary causal decoder with a linear
`board_head`. The model inputs and move policy are shared. Targets describe the
board after the input move, which is the board before the next target move;
BOS targets the game's initial board. Ground-truth boards are never model inputs.

## Targets and masks

`PGNMoveBoardDataset` replays the full PGN prefix before labeling a crop, including
custom starting FENs, castling, en passant and promotions. It returns unshifted
`board_labels`; EOS/padding has no target. `CollateLLMBoardMoves` aligns these
with next-move labels and applies the same supervision mask. Output shape is
`[B,T,64,13]`, with rank 8-to-1 square order and the renderer's piece vocabulary.

## Objective

`DecoderBoardObjective` composes mean cross entropies:

```text
current_raw = CE_move + piece_loss_weight * CE_piece
current = current_raw / (log(move_vocab_size) + piece_loss_weight * log(piece_vocab_size))
board = CE_board / log(13)
loss = (1 - board_loss_weight) * current + board_loss_weight * board
```

Default `board_loss_weight` is 0.30; `normalization=none` disables the fixed
uniform-prediction scales for ablations. This is a fixed scale convention, not
a guarantee of equal gradient magnitudes. All-ignored targets return a
differentiable zero. Reductions run in float32 under mixed precision.

## Training and inference

```bash
python -m train.train_chessdecoder_board --config-name chess_decoder_player_balanced_board_v1
python -m train.train_chessdecoder_board checkpoint_load_path=auto
python -m unittest tests.test_decoder_board_auxiliary tests.test_tournament_decoder_loading -v
```

The recipe inherits balanced-v1 settings and starts from random weights by default.
Every optimized stage must include `board_head`. `BoardTrainer` inherits the
ordinary decoder's optimizer, validation, checkpoint and interruption lifecycle.
It reports raw/scaled component losses, per-square accuracy, occupied-square
accuracy and exact-board accuracy alongside ordinary move metrics.

In evaluation mode the board head is skipped unless `return_board_logits=True`,
as used for auxiliary validation. Both decoder variants share `forward_features`
and support `capture_layers` for probes. Parameter names and policy state-dict
keys are unchanged. `inference_state_dict()` omits only `board_head.*`, producing
weights strictly loadable by the original decoder. The GUI and tournament UCI
loaders detect the auxiliary keys and load the appropriate class strictly.

Tests cover causal behavior, full-prefix crop targets, special moves, mask
propagation, normalization, gradients, checkpoint resume and exact policy-logit
parity when the auxiliary head is unused.
