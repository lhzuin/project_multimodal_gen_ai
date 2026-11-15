"""Thesis-facing tables and dependency-free SVG summary matrices."""
from __future__ import annotations

from collections import Counter
import csv
from html import escape
import json

import numpy as np

from .common import CATEGORIES, atomic_json, atomic_text


def _read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _matrix_svg(path, title, rows, columns, values, *, percent=False):
    """Numeric labels remain authoritative; colors show within-panel magnitude."""
    width, height = 700, 155 + 85 * len(rows)
    finite = [value for row in values for value in row]
    low, high = min(finite), max(finite)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="32" font-family="sans-serif" font-size="20">{escape(title)}</text>',
             '<g font-family="sans-serif" font-size="15">']
    for j, name in enumerate(columns):
        parts.append(f'<text x="{245 + j * 145}" y="78" text-anchor="middle">{escape(name)}</text>')
    for i, (name, row) in enumerate(zip(rows, values)):
        y = 100 + i * 85
        parts.append(f'<text x="15" y="{y + 40}">{escape(name)}</text>')
        for j, value in enumerate(row):
            fraction = (value - low) / (high - low) if high > low else .5
            color = f'rgb({int(240 - 110 * fraction)},{int(246 - 65 * fraction)},250)'
            x = 175 + j * 145
            label = f"{100 * value:.2f}%" if percent else f"{value:.3f}"
            parts += [f'<rect x="{x}" y="{y}" width="140" height="75" rx="3" fill="{color}"/>',
                      f'<text x="{x + 70}" y="{y + 43}" text-anchor="middle">{label}</text>']
    parts += [f'<text x="15" y="{height - 15}" font-size="12">Columns: held-out test category. See CSV tables for paired game-bootstrap intervals.</text>', '</g></svg>']
    atomic_text(path, "\n".join(parts) + "\n")


def render(config):
    output = config.output_dir
    if not json.loads((output / "summary_manifest.json").read_text())["complete"]:
        raise ValueError("summary generation is incomplete; rerun --stage summarize")
    summary = _read(output / "prediction_summary.csv")
    disagreement = _read(output / "disagreement.csv")
    deltas = _read(output / "paired_deltas.csv")
    lookup = {(r["model"], r["test_category"]): r for r in summary if r["scope"] == "all"}
    pairs = {(r["model"], r["test_category"], r["scope"]): r for r in disagreement if r["reference"] == "classic"}
    lines = ["# Encoder policies across Event categories", "",
             "All policies use the same frozen games and positions within each column. Complete legal moves include promotion suffixes. NLL uses natural logarithms and an untempered legal distribution; top-1 follows deployed hierarchical greedy selection.", "",
             "## Human-move prediction", "",
             "Each cell is agreement (%) / NLL (nats per move).", "",
             "| Training category | Classical test | Rapid test | Blitz test |",
             "|---|---:|---:|---:|"]
    for model in CATEGORIES:
        cells = [f"{100 * float(lookup[model, c]['agreement']):.2f} / {float(lookup[model, c]['nll']):.4f}" for c in CATEGORIES]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    for field, title in (("n_games", "Games"), ("n_positions", "Positions")):
        lines.append(f"| {title} | " + " | ".join(lookup['classic', c][field] for c in CATEGORIES) + " |")
    for scope, heading in (("all", "Policy disagreement"), ("nonforced", "Disagreement with at least two legal moves")):
        lines += ["", f"## {heading}", "", "Percent disagreement [95% game-bootstrap interval].", "",
                  "| Policy pair | Classical test | Rapid test | Blitz test |", "|---|---:|---:|---:|"]
        for model in ("rapid", "blitz"):
            cells = []
            for category in CATEGORIES:
                row = pairs.get((model, category, scope))
                cells.append("No positions" if row is None else f"{100 * float(row['disagreement']):.2f} [{100 * float(row['ci_low']):.2f}, {100 * float(row['ci_high']):.2f}]")
            lines.append(f"| {model}–classic | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences from the classical policy", "",
              "Agreement differences are percentage points; NLL differences are nats. Positive agreement and negative NLL favor the category policy.", "",
              "| Policy | Test category | Δ agreement [95% CI] | Δ NLL [95% CI] |", "|---|---|---:|---:|"]
    for row in deltas:
        if row["scope"] != "all" or row["reference"] != "classic":
            continue
        a = [100 * float(row[k]) for k in ("delta_agreement", "delta_agreement_ci_low", "delta_agreement_ci_high")]
        n = [float(row[k]) for k in ("delta_nll", "delta_nll_ci_low", "delta_nll_ci_high")]
        lines.append(f"| {row['model']} | {row['test_category']} | {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}] | {n[0]:+.4f} [{n[1]:+.4f}, {n[2]:+.4f}] |")
    lines += ["", "## Corpus and artifacts", "",
              "Category labels reuse the training Event heuristic: blitz takes precedence over rapid; other Event names are classic. They are not certified clock settings. Both player ratings meet the configured minimum.", "",
              "- `selection_manifest.json`, `selected_games.pgn`: frozen sample, identities, headers, source hashes and selection rule.",
              "- `corpus_audit.json`, `corpus_profile.json`, `games.csv`: candidate counts, exclusions, realized quotas, events, issues, ratings and time-control metadata.",
              "- `results.sqlite`: authoritative transactional per-position predictions for all policies; one complete game is committed at a time.",
              "- `per_move.csv`: regenerated long-format export, with human/predicted moves, target log-probability, FEN, legal counts, color and promotion indicators.",
              "- `prediction_summary.csv`, `paired_deltas.csv`, `disagreement.csv`: position-weighted estimates and paired intervals, overall and by ply band, color, promotions and non-forced positions. Equal-game estimates are supplementary.",
              "- `category_macro.csv`: equal average across the three category scores, with stratified game-bootstrap intervals.",
              "- `specialization.csv`: own-versus-classical-category difference in the policy's advantage over the classical reference; agreement and NLL contrasts use independent within-category game resampling.",
              "- `run_manifest.json`, `summary_manifest.json`: checkpoints/code/runtime provenance, completion status, uncertainty settings and metric definitions.", "",
              "Intervals condition on the evaluated checkpoints and use paired complete-game resampling. Secondary strata resample games contributing positions to that stratum. Intervals do not include training-seed variability or account for all shared-event/player dependence. Secondary strata and specialization interactions are exploratory; there is no multiple-comparison adjustment. Compare policies within a test category before attributing differences to specialization. Category-macro scores weight categories equally; pooled positions would answer a different question.", "",
              "No tournament evaluation is included in this pipeline."]
    atomic_text(output / "README.md", "\n".join(lines) + "\n")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for metric, title, percent in (("agreement", "Complete-move human agreement", True), ("nll", "Human-move NLL (lower is better)", False)):
        values = [[float(lookup[m, c][metric]) for c in CATEGORIES] for m in CATEGORIES]
        _matrix_svg(figures / f"{metric}_matrix.svg", title, list(CATEGORIES), list(CATEGORIES), values, percent=percent)
    values = [[float(pairs[m, c, 'all']["disagreement"]) for c in CATEGORIES] for m in ("rapid", "blitz")]
    _matrix_svg(figures / "disagreement_matrix.svg", "Top-1 disagreement with classical policy", ["rapid–classic", "blitz–classic"], list(CATEGORIES), values, percent=True)
    manifest = json.loads((output / "selection_manifest.json").read_text())
    profile = {}
    for category in CATEGORIES:
        games = [g for g in manifest["games"] if g["category"] == category]
        ratings = [int(g["headers"][key]) for g in games for key in ("WhiteElo", "BlackElo")]
        profile[category] = {
            "games": len(games), "positions": sum(g["game_plies"] for g in games),
            "player_rating_min": min(ratings), "player_rating_median": float(np.median(ratings)), "player_rating_max": max(ratings),
            "events": dict(Counter(g["headers"].get("Event", "") for g in games)),
            "issues": dict(Counter(g["headers"]["TWICIssue"] for g in games)),
            "time_controls": dict(Counter(g["headers"].get("TimeControl", "missing") for g in games)),
            "dates": dict(Counter(g["headers"].get("Date", "missing") for g in games)),
            "forced_positions": int(lookup['classic', category]['n_positions']) - sum(
                int(r["n_positions"]) for r in summary if r["model"] == "classic" and r["test_category"] == category and r["scope"] == "nonforced"),
        }
    atomic_json(output / "corpus_profile.json", profile)
