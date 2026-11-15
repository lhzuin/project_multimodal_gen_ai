# Architecture

The pipeline separates corpus preparation, model inputs, optimization and
measurement. Run entry points as Python modules from the project root.

## Models and inputs

| Component | Main class | Input | Output |
|---|---|---|---|
| Vision | `ChessVisualClassifierTransformers` with the `ChessVit` backbone | Rendered board image | 13 piece/empty classes on each square |
| Board policy | `ChessEncoderPlayerV2` | 64 piece IDs, side, castling and en-passant metadata | Origin/destination scores and promotion logits |
| History policy | `ChessDecoderPlayer` | Causal UCI move tokens, moved-piece tokens and starting turn | Next-move and moved-piece logits |
| Auxiliary history policy | `ChessDecoderBoardPlayer` | Same history inputs | Policy outputs plus optional `[B,T,64,13]` reconstruction logits |

`ChessEncoderPlayer` and `train.train_chessformer` support the earlier board
policy checkpoints and distillation recipes. The sequence trainer is the current
supervised board-policy entry point. `models/chessformer.py` contains the shared
encoder body; `models/chess_llm.py` contains the causal decoder body.

Board targets use rank 8 to rank 1, file a to h. Classes are empty (0), white
pawn/knight/bishop/rook/queen/king (1-6), then black pieces (7-12). Move squares
use python-chess indexing (`a1=0`); convert explicitly when joining these spaces.
The encoder encodes white-to-move as 1; the decoder starting-turn annotation uses
white=0. These conventions are part of the existing checkpoint contract.

The decoder's turn embedding is added after its transformer backbone.
`forward_features` is shared by the ordinary policy, auxiliary head and probes.
`capture_layers` returns numbered block states; the final capture includes
normalization and turn embedding and is exactly the move-head input.

## Data and training

`dataset/offsets.py` indexes PGN byte ranges. Datasets read complete records
lazily; collators own padding and target alignment. The tokenizer vocabulary is
versioned in `tokenizers/chess_uci_vocab.json`.

- `ChessGameSampleDataset` supplies rendered sampled positions for vision.
- `ChessGameSequenceDataset` supplies board sequences for supervised policy training.
- `PGNMoveDataset` supplies causal token crops. `PGNMoveBoardDataset` adds targets
  by replaying the original PGN prefix, including its starting FEN.
- `BalancedTrainingPlan` reads offline decisions and supervision bits. It does
  no statistical replanning during training.

The supervised encoder and decoder trainers retain their own model-specific
batch objectives. `train/training_utils.py` owns device selection, safe logging,
RNG capture, checkpoint rotation and graceful interruption. `BoardTrainer`
inherits the decoder's lifecycle and overrides its loss/component reporting.

## Reinforcement learning

`train/rl/` owns the algorithm-independent contracts and lifecycle. A policy
adapter converts model outputs into probabilities over complete legal moves.
The collector requests multiple trajectories per selected source position and
keeps collection separate from synchronous optimization. The reference policy
remains frozen. GRPO and RLOO implement separate objectives and update loops in
their respective packages.

`rl_stockfish_service/` owns worker processes, bounded queues, game state and
reward evaluation. Each worker exclusively owns one UCI engine. The in-process
adapter in `train/rl/server.py` and the TCP client use the same rollout contract.
The stateless `play_positions` endpoint serves full-game tournaments; game
adjudication remains with the tournament runner.

## Evaluation boundaries

Tournament runners share opening selection, the full-game loop and output
formats. The async runner schedules independent games and serializes access to
each model process; it does not change the policy's deployed sampling rule.

Offline analysis is organized by task:

- `ply_comparison/`: paired predictions and game-cluster summaries.
- `state_probe/`: frozen activations, learned readouts, cross-replay and novelty.
- `decoder_alignment/`: candidate decoder predictions against saved baseline rows.
- `time_control_comparison/`: frozen category sampling, inference, SQLite storage,
  paired statistics, overlap audits and reporting.
- `state_repetition/`: streaming corpus repetition measurements.

Scientific run manifests hash inputs and relevant code. Source changes can
invalidate strict resume identities even when model behavior is unchanged.
Keep completed runs intact and use a new output directory after refactoring.
