# Multimodal Chess Project

This repository contains the full training, evaluation, and inference pipeline for the chess models developed in the project. It includes data preparation utilities, dataset code, model definitions, training entry points, tournament evaluation tools, UCI engines for GUI play, and plotting scripts used for analysis and report figures.

The project is organized around three main model families:

- **ViT** for board-state perception from rendered board images
- **Encoder** for next-move prediction from explicit board representations
- **Decoder** for next-move prediction from move history only

The training workflow relies on a PGN-based corpus, random-access game indexing, sampled position files for board-image datasets, and configuration-driven training scripts.

## Repository structure

The project is organized by function.

```text
project_multimodal_gen_ai/
├── checkpoints/        # trained model checkpoints
├── configs/            # training and evaluation configurations
├── dataset/            # datasets, collate functions, rendering, augmentations, sprites, sanity checks
├── models/             # model definitions
├── processed_games/    # merged PGNs, PGN indices, processed training corpora
├── tokenizers/         # tokenizer files for decoder models
├── train/              # training entry points
├── twic_zips/          # raw TWIC zip archives
├── utils/              # data-preparation, tournament, UCI, plotting, and helper scripts
├── outputs/            # training outputs and logs
├── tournament_out/     # tournament PGNs, summaries, and JSONL outputs
├── debug_plots*/       # analysis and debugging visualizations
└── README.md
```

### What is stored where

- `configs/` contains the experiment configurations used to train the final project models and intermediate variants.
- `dataset/` contains all dataset logic, board rendering, augmentations, PGN indexing helpers, collate functions, and the sampled positions JSON used by the board-image pipeline.
- `models/` contains the ViT, encoder, decoder, tokenizer, and related model components.
- `processed_games/` contains merged PGNs and their byte-offset indices, as well as larger processed corpora used by different training branches.
- `tokenizers/` contains the UCI tokenizer vocabulary used by the decoder.
- `train/` contains the main training entry points.
- `utils/` contains preparation scripts, tournament tools, opening-book generation, Stockfish-data helpers, plotting utilities, and UCI engines.
- `checkpoints/` stores trained checkpoints.
- `tournament_out/` stores tournament artifacts such as PGNs and summary files.
- `outputs/` stores training outputs and run directories.
- `twic_zips/` stores downloaded raw TWIC archives so they can be reused across runs.

## Development flow and best practices

### Run scripts as modules

This repository is structured as Python packages. The recommended way to run scripts is from the repository root with `python -m`.

Examples:

```bash
python -m utils.prepare_training_files --project-root .
python -m dataset.sanity_test
python -m train.train_vit_v2
```

This keeps imports consistent across the project and avoids path issues.

### Keep raw, processed, and generated artifacts separated

A practical workflow is:

- keep raw TWIC archives in `twic_zips/`
- keep merged and indexed PGNs in `processed_games/`
- keep sampled board positions in `dataset/`
- keep checkpoints in `checkpoints/`
- keep tournament outputs in `tournament_out/`
- keep analysis plots in dedicated plot folders

### Prefer configuration-driven runs

Training and evaluation settings should come from files in `configs/`. When a script supports config selection, prefer running it with the intended config explicitly. Otherwise, the script should use its default final project configuration.

### Reuse existing artifacts whenever possible

Before rebuilding large files, check whether the repository already contains the required data:

- existing TWIC zip archives in `twic_zips/`
- existing merged PGNs in `processed_games/`
- existing sampled position files in `dataset/`
- existing opening suites in `utils/`

This saves time and keeps experiments reproducible.

### Validate the data pipeline before training

Always run the sanity test after generating or modifying the training artifacts. This catches dataset, indexing, rendering, and collate issues before long training runs.

## Environment setup

Create a virtual environment from the repository root and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below assume they are run from the repository root inside the project environment.

## Core training artifacts

Three files are required before running the board-image and board-state training pipelines:

- `processed_games/games.pgn`
- `processed_games/games_index.json`
- `dataset/games_samples_*.json`

These are used as follows:

- the PGN stores the merged games
- the PGN index enables random access to individual games
- the sampled positions JSON fixes which positions are extracted from each game for board-based datasets

## Step 1 — Prepare the training files

Generate the main training artifacts with:

```bash
python -m utils.prepare_training_files \
  --project-root . \
  --twic-start 1484 \
  --twic-end 1633
```

By default, this will:

- reuse or create `processed_games/games.pgn`
- reuse or create `processed_games/games_index.json`
- reuse or create `dataset/games_samples_s42_r0.1_m32.json`

The preparation script is aligned with the repository layout:

- merged PGNs and byte offsets go into `processed_games/`
- the samples JSON goes into `dataset/`
- TWIC zip files are expected in `twic_zips/`

### What the preparation step does

1. reuses an existing merged PGN if present
2. otherwise reuses or downloads the requested TWIC zip files
3. merges the TWIC PGNs into one file
4. builds the random-access byte-offset index
5. instantiates `ChessGameSampleDataset` to create the sampled positions file

### Useful variants

Reuse an existing merged PGN directly:

```bash
python -m utils.prepare_training_files \
  --project-root . \
  --pgn-path processed_games/games_xl.pgn \
  --pgn-name games.pgn
```

Use existing TWIC zips only, without downloading anything new:

```bash
python -m utils.prepare_training_files \
  --project-root . \
  --twic-start 1484 \
  --twic-end 1633 \
  --skip-download \
  --force-rebuild-pgn
```

Force a rebuild of index and samples:

```bash
python -m utils.prepare_training_files \
  --project-root . \
  --force-rebuild-index \
  --force-rebuild-samples
```

## Step 2 — Sanity-check the dataset pipeline

After preparing the files, run:

```bash
python -m dataset.sanity_test
```

This verifies that the pipeline can:

- load `processed_games/games.pgn`
- load `processed_games/games_index.json`
- load `dataset/games_samples_s42_r0.1_m32.json`
- build at least one dataloader batch
- validate tensor shapes and value ranges
- render a few example positions

This step should be part of the standard workflow whenever the dataset, samples, or rendering code changes.

## Configs and default behavior

Training configs are stored in `configs/`.

Examples currently present in the repository include:

- `configs/chess_classifier_vit_v2.yaml`
- `configs/chess_encoder_seq_v2.yaml`
- `configs/chess_decoder_player_v12.yaml`
- `configs/distillation_encoder_v9.yaml`

The commands below are written with the final project configurations in mind. When supported by the training entry point, prefer passing the desired config explicitly.

## Step 3 — Train the ViT branch

The ViT branch is the visual front-end. It learns to recover board information from rendered board images under augmentation and style variation.

Run:

```bash
python -m train.train_vit_v2
```

If the training entry point supports explicit config selection:

```bash
python -m train.train_vit_v2 --config-name chess_classifier_vit_v2
```

## Step 4 — Train the supervised encoder

The supervised encoder is the main board-based policy model. It predicts the next move from an explicit board representation.

Run:

```bash
python -m train.train_encoder_seq
```

Recommended final config:

```bash
python -m train.train_encoder_seq --config-name chess_encoder_seq_v2
```

## Step 5 — Train the decoder

The decoder predicts the next move from move history only, using the UCI tokenizer and move-sequence modeling.

Run:

```bash
python -m train.train_chessdecoder
```

Recommended final config:

```bash
python -m train.train_chessdecoder --config-name chess_decoder_player_v12
```

## Step 6 — Train the distilled encoder

The distilled encoder uses offline teacher supervision from Stockfish top moves rather than only the human next move.

Run:

```bash
python -m train.distillation_encoder
```

Recommended final config:

```bash
python -m train.distillation_encoder --config-name distillation_encoder_v9
```

## Step 7 — Generate the opening suite

Tournament evaluation uses a curated opening suite. Generate it with:

```bash
python -m utils.generate_opening_book \
  --pgn processed_games/games_large.pgn \
  --out utils/opening_suite_large.json \
  --full_moves 8
```

This produces the JSON opening set consumed by the tournament runner.

## Step 8 — Evaluate with tournaments against Stockfish

Tournament evaluation is run with `utils.tournament_runner`, which launches the model through `utils/tournament_uci_engine.py` and plays from the generated opening suite.

### Decoder tournament example

```bash
python -m utils.tournament_runner \
  --openings_json utils/opening_suite_large.json \
  --pct 0.004 \
  --seed 0 \
  --stockfish /path/to/stockfish \
  --movetime_ms 10 \
  --limit_strength \
  --uci_elo 1500 \
  --model_uci_py utils/tournament_uci_engine.py \
  --model_type decoder \
  --ckpt checkpoints/chess_llm_decoder_v12.pt \
  --tokenizer_path tokenizers/chess_uci_vocab.json \
  --n_layers 12 \
  --write_pgn \
  --verbose \
  --greedy
```

### Encoder tournament example

```bash
python -m utils.tournament_runner \
  --openings_json utils/opening_suite_large.json \
  --pct 0.004 \
  --seed 0 \
  --stockfish /path/to/stockfish \
  --movetime_ms 10 \
  --limit_strength \
  --uci_elo 2000 \
  --model_uci_py utils/tournament_uci_engine.py \
  --model_type encoder \
  --version 2 \
  --ckpt checkpoints/chess_encoder_seq_v2_ep11.pt \
  --tokenizer_path tokenizers/chess_uci_vocab.json \
  --n_layers 8 \
  --write_pgn \
  --verbose \
  --greedy
```

### Notes on tournament evaluation

- `utils/tournament_uci_engine.py` is the UCI bridge used by the tournament runner.
- `--greedy` is the standard evaluation setting for the main tournament runs.
- `--write_pgn` stores the played games for later inspection.
- `--uci_elo` sets the calibrated Stockfish opponent level.

Tournament outputs are typically written to `tournament_out/`.

## Step 9 — Estimate Elo from tournament results

After running tournaments, use `utils.elo_estimator.py` to convert grouped match results against calibrated Stockfish levels into an Elo estimate with uncertainty.

Run:

```bash
python -m utils.elo_estimator
```

The script expects grouped results in the form:

```python
ENCODER_SEQ_MATCHES = [
    (2000, 386, 342, 384),
    (2100, 414, 312, 386),
    (2200, 252, 364, 496),
]
```

Each tuple is:

```text
(opponent_elo, wins, draws, losses)
```

The output includes:

- estimated Elo
- standard error
- 95% confidence interval
- compact summary values suitable for plots or poster text


## Step 10 — Play against the model in a GUI

`utils/uci_engine.py` is the UCI entry point intended for chess GUIs.

A practical workflow is to create a small shell script pointing to the project root and virtual environment, then register that script as a UCI engine in the GUI.

Example:

```bash
#!/bin/bash
cd "/absolute/path/to/project_multimodal_gen_ai" || exit 1

"/absolute/path/to/.venv/bin/python" -u -m utils.uci_engine \
  --model_type encoder \
  --ckpt "/absolute/path/to/project_multimodal_gen_ai/checkpoints/chess_encoder_seq_v2_ep12.pt" \
  --temperature 0.5 \
  --greedy \
  --n_layers 8 \
  --version 2 \
  --device mps
```

Typical usage:

- `utils.uci_engine` for manual play in a GUI
- `utils.tournament_uci_engine` for automated tournament play through `utils.tournament_runner`

The repository already includes helper launch scripts such as `run_uci.sh`, `run_uci.ps1`, and model-specific shell wrappers.

## Step 11 — Generate plots and report figures

The plots used in the report are generated with the Python scripts inside `utils/plots/`.

Example:

```bash
python -m utils.plots.plot_prob_distributions \
  --cfg configs/distillation_encoder_v9.yaml \
  --ckpt checkpoints/chess_encoder_seq_v1_07_03_FINAL.pt \
  --decoder_ckpt checkpoints/chess_llm_decoder_v11.pt \
  --decoder_tokenizer_path tokenizers/chess_uci_vocab.json \
  --device mps \
  --outdir debug_plots
```

Related analysis utilities are located in:

- `utils/plots/`
- `utils/elo_estimator.py`
- other helper scripts in `utils/`

Generated figures can be stored in dedicated output folders such as `debug_plots/`, `debug_plots_final/`, or similar analysis directories.

## Typical end-to-end workflow

A practical full workflow is:

1. create the virtual environment and install dependencies
2. prepare the PGN, index, and sampled positions files
3. run the sanity test
4. train the ViT branch
5. train the supervised encoder
6. train the decoder
7. optionally generate Stockfish supervision data and train the distilled encoder
8. generate the opening suite
9. run tournaments against calibrated Stockfish
10. estimate Elo from the tournament results
11. inspect PGNs, test the engine in a GUI, and generate final plots

## Main assets already present in the repository

The current repository already contains many reusable assets, including:

- multiple checkpoints in `checkpoints/`
- final and intermediate configs in `configs/`
- prepared corpora in `processed_games/`
- TWIC archives in `twic_zips/`
- tokenizer files in `tokenizers/`
- plotting outputs and debugging visualizations in the plot directories

This makes it possible to resume experiments, rerun evaluations, or extend the project without rebuilding everything from scratch.

## Notes on workflow discipline

For stable and reproducible experiments:

- keep commands run from the repository root
- avoid mixing ad hoc local paths directly into reusable scripts
- prefer config files over editing code for experiment changes
- rebuild only the artifacts that actually changed
- validate the dataset pipeline before long training runs
- keep tournament outputs, plots, and checkpoints organized by purpose

## Training and evaluation entry points

### Data preparation and validation
- `python -m utils.prepare_training_files`
- `python -m dataset.sanity_test`

### Training
- `python -m train.train_vit_v2`
- `python -m train.train_encoder_seq`
- `python -m train.train_chessdecoder`
- `python -m train.distillation_encoder`

### Evaluation and analysis
- `python -m utils.generate_opening_book`
- `python -m utils.tournament_runner`
- `python -m utils.elo_estimator`
- `python -m utils.plots.<plot_script>`

### UCI / GUI usage
- `python -m utils.uci_engine`
- `python -m utils.tournament_uci_engine`
