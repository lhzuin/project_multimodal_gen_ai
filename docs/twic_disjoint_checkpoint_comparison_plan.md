# Plan: comparison by ply on recent TWIC games

Plan date: 21 October 2026. Scope: compare, without retraining, the checkpoints
`chess_encoder_seq_v1_07_03_FINAL.pt` and `chess_encoder_seq_balanced_v1.best.pt`,
and `chess_llm_decoder_v12.pt` and `chess_llm_decoder_balanced_v1_final.pt`.
All four files are in the project root. The primary comparisons are **within
each architecture**, on the same set of human moves.

## 1. Hypotheses and limits of inference

- H1: balanced minus original performs worse at early plies and better at later
  plies, particularly for the decoder.
- H2: the decoder policy depends more on move-history information and may find
  it increasingly difficult to recover the true state as the history grows or
  is truncated.
- H3: model-selected moves may lead to states outside the distribution of human
  games used for training.

Next-move evaluation on human games tests H1. It does not isolate H2 or H3:
the prefix is the true recorded history (teacher forcing), and the causal
decoder does not maintain a predicted board that it feeds back recurrently.
H2 requires a state probe or context ablation; H3 requires trajectories that
contain model actions. An observed difference also does not identify the
causal effect of redistribution, because the training corpora, filters,
windows, and compute actually consumed may have changed between runs.

## 2. Freeze the test data

The project owner confirmed that **TWIC issues 1662–1667 are disjoint from all
four training runs**. This range is fixed as the test set. The official TWIC
archive dates these issues from 14 September through 19 October 2026. Record
the owner's confirmation in the experiment manifest. The absence of some
training PGNs from this checkout does not block this evaluation.

1. Record in the manifest that the project owner confirmed the temporal
   separation of issues 1662–1667 from all four training runs. Save hashes of
   the files actually used for testing.
2. Add an `--all-games-in-range` mode to
   `utils/dataset_by_category/twic_category_builder.py`, with no stopping
   condition based on a ply target. Process every issue from 1667 down to
   1662 in full, retaining per-issue reports. Do not use an artificially huge
   target with exit status 2 as the final protocol.
3. Apply the same filters for all four models: valid PGN, standard variant,
   nonempty game, no null move, numeric Elo >= 1600 for both players, and the
   current `classic` category heuristic based on the `Event` header. Report
   the numbers of games and plies excluded for each reason. The residual
   `classic` category does not establish the actual time control; validate
   `TimeControl` when available or describe it as a category inferred from
   event names. Do not expand beyond the confirmed range without establishing
   a new provenance rule.
4. Detect duplicate games by initial FEN plus the mainline UCI sequence,
   including duplicates across issues, and remove them. Repeated *positions*
   across different games can remain; quantify them, especially early on.
5. Freeze the PGN, offset index, and manifest (issues and URLs, filters,
   counts, input and output hashes, code and tokenizer versions), along with
   an example table containing `game_id, ply_zero_based, FEN_before,
   UCI_target`. Use the same manifest for every inference run. Audit the PGN's
   internal `Date` field: the issue number establishes provenance but does
   not guarantee the date on which a game was played.

Planned command after adding `--all-games-in-range`:

```powershell
python -m utils.dataset_by_category.twic_category_builder `
  --all-games-in-range --categories classic `
  --start-week 1667 --lower-week 1662 --min-elo 1600 `
  --output-dir processed_games/eval --prefix twic_1662_1667
```

The existing builder already downloads and validates each issue's ZIP,
filters games, writes a PGN and index, and produces reports. Its current
default starts at issue 1661 and stops when it reaches a ply target, so it
needs the range mode to produce this complete test set.

## 3. Inference consistent with the checkpoints

Create `analysis/ply_comparison/evaluate.py` to consume the frozen manifest.
Do not use the YAML files' `dataset_val` or the balanced training masks.

- Instantiate each encoder as `ChessEncoderPlayerV2` with the parameters in
  its YAML: dimension 256, 8 layers, 4 heads, and `labels64` plus `turn`,
  `castling`, and `ep_square`. Obtain `labels64` from the board **before** the
  target move. Use `map_location=cpu`, load weights with `strict=True`, call
  `eval()` and `inference_mode()`, and verify the checkpoint hash and number
  of keys. The ViT weights are present, but the `labels64` path does not
  render images. Note that the encoder encodes White to move as `turn=1`.
- Instantiate each decoder as `ChessDecoderPlayer` with the parameters in
  its YAML: dimension 256, 12 layers, 4 heads, tokenizer
  `tokenizers/chess_uci_vocab.json`, context length 256, `piece_input_ids`
  constructed by replay, and `start_turn` aligned to the beginning of the
  window. The prefix for predicting ply `t` is
  `[BOS, move_0, ..., move_(t-1)]`; retain only the last 256 tokens. Do not
  include the target move, EOS, or a future piece label. Note that the
  decoder encodes White to move as `start_turn=0`.
- Reject examples whose recorded move is illegal, is absent from the UCI
  vocabulary, or follows a terminal position. Report the shared exclusions.
  Manually verify PGN, FEN, prefix, turn, and target alignment on a sample,
  especially around plies 255/256 and at promotions.
- For a metric comparable across architectures, normalize over **legal UCI
  moves only**. For the decoder, mask illegal and special tokens and
  renormalize. For the encoder, normalize legal `(from,to)` pair logits and,
  when a pair has several legal promotions, divide its probability among
  those promotions using the `promo_head`. This preserves the two-stage
  decision structure of `sample_moves()`. Adding `promo_logits` to all legal
  move logits without this conditioning, as the distribution plot script
  does, changes the relative probabilities of move pairs. Also retain the
  encoder's `(from,to)` accuracy to reproduce the historical metric. For
  top-1, separately reproduce the greedy decision in `sample_moves()`,
  including its fallback when the preferred promotion class does not
  correspond to a legal promotion. Use the explicit UCI distribution for NLL.

## 4. Metrics and analysis

Use **absolute ply before the target move** with these bands: `[0,20)`,
`[20,40)`, `[40,60)`, `[60,80)`, `[80,100)`, and `[100,+inf)`. These match the
balanced training plan's bands. For each band, report the number of games and
decisions, which must be shared within each pair of models. As a complement,
divide relative progress `t / n_plies` into quintiles, again showing sample
sizes. Do not call ply-number bands literal opening, middlegame, or endgame
phases, and do not mix absolute and relative segmentations.

The primary metric is top-1 accuracy for the **complete legal UCI move**
against the recorded move, plus the paired difference, balanced minus
original, in each band. Secondary metrics are legal top-5 (the greedy choice
followed by the four highest-probability remaining moves), mean
`-log p(human_move | legal_moves)`, and perplexity `exp(mean_NLL)`, plus
`(from,to)` and promotion accuracy where applicable. If full-vocabulary
decoder NLL is also calculated, report it separately with its definition;
do not compare its perplexity to legality-renormalized NLL without stating
the distinction. The recorded human move is only one possible good move, so
agreement with it is not a direct measure of chess strength.

Store one row per `(game_id, ply, model_id)` with top-1/top-5 correctness,
NLL, target probability, top-1 move, number of legal moves, visible context
length, Elo, TWIC issue, and category. Aggregate by band with **decision**
weights and, as a sensitivity analysis, equal **game** weights. Estimate 95%
confidence intervals for paired differences through a game-level bootstrap
(the same resampling for both checkpoints), a fixed seed, and 2,000
replicates. Plies from one game must not be treated as independent. Show
smoothed curves by ply as a secondary visualization, but use the predefined
bands for the primary analysis. Exploratory strata can include side to move,
Elo, issue, game length, and whether the context was truncated.

The descriptive check for H1 is the sign and confidence interval of the
difference in each band, with particular attention to the first band and the
early-versus-later interaction. If neither NLL nor paired accuracy supports
H1, the claim that early-move performance worsened lacks support from this
test. Do not fill that gap with an unsupported claim about drift.

## 5. Separate mechanistic tests

1. **Context and internal state (H2):** evaluate the same human prefix with
   controlled windows of 32, 64, 128, and 256 tokens, aligning `start_turn`.
   Separate cases in which the initial FEN can still be reconstructed. Train
   a small linear or MLP probe on **different games**, with the policy
   backbone frozen, to recover the board before the move from the last
   token's hidden state. Measure per-square, occupied-square, and exact-board
   accuracy by ply. Recoverability does not prove that the policy head uses
   the information causally.
2. **Trajectory shift (H3):** start from matched opening positions and
   alternate each checkpoint's greedy actions with the same opponent under
   the same budget. At each visited position, measure NLL/agreement against
   a reference when one exists, state novelty against an available training
   corpus, and, on a fixed sample, move-value loss under a fixed Stockfish
   configuration. Pair openings, color, and opponent, and record game lengths
   and visited states. Differences between trajectories include state
   selection and do not isolate accumulated internal error.
3. **Decision quality:** to test whether early mistakes cost more, sample
   human positions by band and evaluate the four models' top-1 moves with
   the same Stockfish settings (version, depth/time, Threads, Hash, and score
   perspective). Report centipawn loss or change in WDL by band. Agreement
   with a human move alone does not test the cost of a mistake.

## 6. Deliverables and acceptance criteria

Deliver the test manifest, a reproducible command, per-move results,
`bucket_summary.csv`, `relative_summary.csv`, `paired_deltas.csv`, and SVG figures for accuracy and
NLL (levels and paired differences with confidence intervals) produced by
`analysis/ply_comparison/plot_results.py`. Generate a `README.md` alongside
the figures explaining what was actually run, the metrics, sample sizes,
filters, and limitations. The CSV contract is defined in
`docs/twic_disjoint_comparison_method.md`. Save
hashes of the four checkpoints, tokenizer, PGN, and indexes, plus the Git
revision. Run alignment tests covering the start of a game, context-window
truncation, promotion, games with an initial FEN, turn encoding, legal masks,
and deduplication. Run a smoke test on a few games before full inference,
then reproduce a subset of results on CPU.

Run inference with the project's `.venv`, which contains the required
dependencies.

TWIC archive: https://theweekinchess.com/twic
