# Development and submission

Use Python 3.12, install `requirements.txt` and run modules from the root. Keep
architecture classes in `models/`, data/target construction in `dataset/`, and
algorithm-specific updates in their trainer packages. Add reusable analyses as
packages with a CLI, rather than as scripts named after a run date.

## Verification

Install `requirements-dev.txt` for static checks, then run `ruff check analysis dataset models train utils rl_stockfish_service tests scripts`. The check covers undefined names, unused imports and redefinitions.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -v
python scripts/check_submission.py
```

The tests generate small PGNs/models and use temporary outputs. They cover mask
alignment, legal scoring, checkpoint/RNG resume, RL gradients, transactional
analysis recovery and tournament protocol parity. The standard suite requires
neither downloaded data nor trained checkpoints. Loopback tests run when the
execution environment permits sockets.

Real-engine checks are opt-in:

```bash
RL_STOCKFISH_BIN=/path/to/stockfish python -m unittest rl_stockfish_service.tests.test_engine_integration -v
RL_STOCKFISH_BIN=/path/to/stockfish GRPO_REAL_MODELS=1 RLOO_REAL_MODELS=1 \
  ASYNC_TOURNAMENT_REAL_MODELS=1 python -m unittest \
  train.grpo.tests.test_integration train.rloo.tests.test_integration \
  tests.test_async_tournament_integration -v
```

The latter commands require the named encoder/decoder checkpoints in
`checkpoints/`. They perform short real updates and UCI games using temporary
files; they do not replace saved experiment exports. CI runs offline regression
tests and the real Stockfish service checks, without private weights.

## Changes that need care

- Preserve state-dict keys and tensor conventions when refactoring shared model
  code. Use exact policy-logit parity tests for feature-path changes.
- Keep input/context masks separate from supervision masks. Preserve move and
  moved-piece target shifts and custom starting FENs.
- Preserve the optimizer/reference/RNG/data cursor together for RL resume.
- Hash the actual source corpus and selected weights. A new source revision
  may intentionally require a new analysis output directory.
- Keep inference, storage, aggregation and report generation separate. Report
  uncertainty at the source-game level where positions share a game.

## What belongs in Git

Include source, tests, small recipes, code guides, tokenizer vocabulary, rendering
sprites and curated result summaries. Keep corpus downloads, generated indices,
training samples, model weights, databases, activation arrays, full predictions,
logs, environment folders and scratch experiments local. `.gitignore` names
those artifact locations; JSON/CSV are not globally ignored because small
reproducible inputs and summaries are useful source assets.

`scripts/check_submission.py` inspects indexed files (and staged updates), checks
file sizes and artifact paths, and verifies curated-result checksums. It does
not delete or upload anything. Run it after staging and before submission.
`git archive HEAD` exports the committed source tree without ignored local data.
The existing Git history still contains historical generated files; removing
those from the index keeps the final tree lean without rewriting shared history.

## Configurations and local artifacts

Keep existing recipe names for reproducibility. The [config index](../configs/README.md)
identifies the active entry points; versioned recipes retain distinct scientific
settings. Use normal resume fields instead of adding a YAML for each restart.
Never embed home-directory paths or credentials in shared code/configs.

Local cleanup backups live under `outputs/archive/`, alongside the retained full
run artifacts. They are not part of a submission. Saved results are documented
under [results](../results/README.md), separately from the code guides.
