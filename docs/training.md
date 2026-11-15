# Training and checkpoints

Run from the repository root. Supervised and distillation entry points use
Hydra. RL entry points use plain YAML and have their own resume lifecycle.

## Supervised recipes

| Task | Entry point | Main recipe |
|---|---|---|
| Vision, clean/augmented validation | `train.train_vit_v2` | `chess_classifier_vit_v2_stats` |
| Board policy | `train.train_encoder_seq` | `chess_encoder_seq_v2` |
| History policy | `train.train_chessdecoder` | `chess_decoder_player_v12` |
| Balanced board policy | `train.train_encoder_seq` | `chess_encoder_seq_balanced_v1` |
| Balanced history policy | `train.train_chessdecoder` | `chess_decoder_player_balanced_v1` |
| Auxiliary board reconstruction | `train.train_chessdecoder_board` | `chess_decoder_player_balanced_board_v1` |
| Offline teacher distillation | `train.distillation_encoder` | `distillation_encoder_v9` |

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1 device=cpu log=false
```

Inspect a composed config without starting training:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1 --cfg job
```

Check corpus paths, tokenizer, balancing plan, architecture and checkpoint names
before a run. Vision/distillation retain their earlier weight-checkpoint behavior;
the full-state resume contract below applies to the sequence encoder, decoder
and inherited auxiliary decoder trainer.

## Exact resume and warm starts

Those three trainers write:

- `<experiment>.latest.pt`: model, optimizer, scheduler, scaler, training cursor,
  early-stopping state, RNG states, resolved config and logging identity.
- `<experiment>.best.pt`: best model weights for evaluation.
- `<experiment>.<stage>.epoch-NNNN.pt`: rotating weight snapshots, controlled by
  `checkpoint_keep_last`.

```bash
python -m train.train_encoder_seq checkpoint_load_path=auto
python -m train.train_chessdecoder checkpoint_load_path=auto
```

A full-state checkpoint resumes progress; a legacy weight-only file is a warm
start. Match the architecture and recipe to the saved checkpoint. `SIGINT`,
`SIGTERM` and, where available, `SIGUSR1` request a stop at the next optimizer
boundary, followed by a resumable save. The auxiliary trainer inherits this
behavior. Do not manually clear progress flags in a shared checkpoint.

`device=auto` selects CUDA, then Apple MPS, then CPU. Supervised sequence trainers
bound W&B initialization retries and can fall back to offline logging. RL logging
is configured independently under `logging`; see [shared RL utilities](../train/rl/README.md).

## Reinforcement learning

```bash
python -m train.rloo --config configs/rl/rloo_decoder_lichess_mixed_v5_horizon8.yaml
python -m train.grpo --config configs/rl/examples/grpo_encoder.yaml
```

The example configs require a running remote service; production configs can own
a local one. Keep rollout horizon, reward scaling, reference checkpoint and
engine budgets fixed when comparing runs. Each process owns and closes its
workers. Full checkpoints preserve the frozen reference and RNG/data cursor;
`export_weights` writes inference-only state dictionaries. `resume_path` is an
explicit YAML field, not a Hydra override. Use fresh output/log paths for a new
experiment, and retain numbered exports when selecting on validation data.

The [RL config index](../configs/rl/README.md), [RLOO guide](../train/rloo/README.md)
and [GRPO guide](../train/grpo/README.md) document objective details.
