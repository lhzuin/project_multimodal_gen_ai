# Configuration index

Supervised training uses Hydra with `--config-name <stem>`. Run from the root:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_balanced_v1 --cfg job
```

| Family | Main recipe(s) | Entry point |
|---|---|---|
| Vision | `chess_classifier_vit_v2_stats` | `train.train_vit_v2` |
| Original sequence encoder | `chess_encoder_seq_v2` | `train.train_encoder_seq` |
| Balanced encoder | `chess_encoder_seq_balanced_v1` | `train.train_encoder_seq` |
| Category encoders | `chess_encoder_seq_balanced_{rapid,blitz}_v1` | `train.train_encoder_seq` |
| Original decoder | `chess_decoder_player_v12` | `train.train_chessdecoder` |
| Balanced decoder | `chess_decoder_player_balanced_v1` (v2/v3 are alternative recipes) | `train.train_chessdecoder` |
| Auxiliary decoder | `chess_decoder_player_balanced_board_v1` | `train.train_chessdecoder_board` |
| Distillation | `distillation_encoder_v9` | `train.distillation_encoder` |
| Earlier board policy | `chess_encoder_player_v3` | `train.train_chessformer` |

Numbered recipes preserve the architecture, data and optimization choices of
past experiments. They remain in this directory so existing Hydra invocations
and checkpoint configurations continue to resolve. Continuation recipes already
in history are retained as experiment records; current sequence/decoder trainers
resume through `checkpoint_load_path=auto` instead of requiring another file.

Before running, set the actual PGN/index, balancing plan, visual checkpoint and
output paths. A recipe is not a bundle of its generated data or weights.

`rl/` contains plain YAML, loaded with `--config`; Hydra defaults/interpolation do
not apply. See [the RL index](rl/README.md).

`state_probe_twic_1662_1667.yaml` configures frozen decoder probes;
`time_control_comparison_twic_1662_1667.yaml` configures category inference. Both
resolve relative artifact paths against the project root.
