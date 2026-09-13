# Building TWIC datasets by category

`utils.dataset_by_category.twic_category_builder` builds separate Blitz, Rapid,
and/or Classic corpora without retaining the downloaded weekly archives. It
downloads issues in descending order, reads PGNs directly from one temporary ZIP,
filters and writes complete games, deletes that ZIP, and continues until every
selected category reaches its ply target or the inclusive lower-week limit.

## Main decisions

- Output is controlled by `--output-dir` plus a shared `--prefix`. For example,
  prefix `twic_recent` creates `twic_recent_blitz.pgn` and
  `twic_recent_blitz_index.json`.
- Classification uses the PGN `Event` header. Names containing `blitz` are Blitz,
  names containing `rapid` are Rapid, and all remaining names are Classic.
- `--min-elo N` requires both `WhiteElo` and `BlackElo` to be numeric and at
  least `N`. Games with missing ratings are excluded when this option is active.
- Variant games are excluded by default whenever a non-empty `Variant` header is
  present. Use `--include-variant` to retain them.
- A whole game is always retained, so the final game can make a corpus exceed
  the requested number of plies.
- Mainlines and headers are retained; comments and side variations are omitted to
  reduce storage and match the current training datasets.
- Each PGN receives the random-access index needed by both the sequential encoder
  (`ChessGameSequenceDataset`) and decoder (`PGNMoveDataset`). A sampled-position
  JSON is intentionally not generated because neither pipeline consumes it.
- Existing outputs are protected unless `--overwrite` is supplied.

## Usage

From the project root, collect at least ten million plies for all three categories:

```bash
python -m utils.dataset_by_category.twic_category_builder 10000000 \
  --categories blitz rapid classic \
  --output-dir processed_games/by_category \
  --prefix twic_recent \
  --min-elo 2000
```

The defaults start at TWIC 1661 and move backwards through TWIC 920 inclusive.
Useful options include:

```text
--start-week 1661
--lower-week 920
--include-variant
--retries 4
--timeout 120
--backoff-seconds 1
--overwrite
```

The primary TWIC URL and browser-like request headers follow
`utils/twic_download.py`; the assets URL is tried as a fallback. Network and
archive failures are recorded and skipped so later weeks can still be processed.

The command writes `<prefix>_report.json` with detailed per-week counts and
`<prefix>_report.md` with a concise human-readable summary. If the lower bound is
reached before a target, both reports contain a warning and the command exits with
status 2. Download failures are always listed in the report.

To train, point the chosen encoder or decoder YAML's `pgn_path` and `index_path`
at one category's generated pair. The decoder still needs its existing tokenizer;
this command does not modify training configurations.
