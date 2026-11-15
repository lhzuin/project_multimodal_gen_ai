"""Plot the frozen TWIC comparison summaries without third-party packages.

The evaluator owns inference, aggregation and game-cluster bootstrap. This
script validates its two summary CSVs and renders publication-ready SVGs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path


BUCKETS = ("0_20", "20_40", "40_60", "60_80", "80_100", "100_plus")
BUCKET_LABELS = ("0–19", "20–39", "40–59", "60–79", "80–99", "100+")
MODELS = (
    "encoder_baseline",
    "encoder_balanced",
    "decoder_baseline",
    "decoder_balanced",
)
FAMILIES = ("encoder", "decoder")
COLORS = {
    "encoder_baseline": "#325b9b",
    "encoder_balanced": "#3b9d78",
    "decoder_baseline": "#c16a39",
    "decoder_balanced": "#9752a0",
    "encoder": "#3b9d78",
    "decoder": "#9752a0",
}
LABELS = {
    "encoder_baseline": "Encoder baseline",
    "encoder_balanced": "Encoder balanced",
    "decoder_baseline": "Decoder baseline",
    "decoder_balanced": "Decoder balanced",
    "encoder": "Encoder",
    "decoder": "Decoder",
}
SUMMARY_COLUMNS = {
    "model_id", "bucket", "n_games", "n_plies", "accuracy_top1",
    "accuracy_top5", "nll_legal",
}
DELTA_COLUMNS = {
    "family", "bucket", "n_games", "n_plies",
    "delta_accuracy_top1", "ci_low_accuracy_top1", "ci_high_accuracy_top1",
    "delta_nll_legal", "ci_low_nll_legal", "ci_high_nll_legal",
}


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def _int(row: dict[str, str], key: str) -> int:
    value = int(row[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive: {value}")
    return value


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite: {row[key]}")
    return value


def load_results(summary_path: Path, delta_path: Path):
    summary = {}
    for row in _read_csv(summary_path, SUMMARY_COLUMNS):
        key = (row["model_id"], row["bucket"])
        if key[0] not in MODELS or key[1] not in BUCKETS or key in summary:
            raise ValueError(f"unknown or duplicate model/bucket: {key}")
        n_games, n_plies = _int(row, "n_games"), _int(row, "n_plies")
        if n_games > n_plies:
            raise ValueError(f"more games than plies: {key}")
        top1, top5, nll = (_float(row, k) for k in
                           ("accuracy_top1", "accuracy_top5", "nll_legal"))
        if not (0 <= top1 <= top5 <= 1) or nll < 0:
            raise ValueError(f"metrics outside the expected range: {key}")
        summary[key] = {
            "n_games": n_games, "n_plies": n_plies,
            "accuracy_top1": top1, "accuracy_top5": top5,
            "nll_legal": nll,
        }
    expected_summary = {(model, bucket) for model in MODELS for bucket in BUCKETS}
    if set(summary) != expected_summary:
        raise ValueError(f"missing rows in bucket_summary.csv: {sorted(expected_summary - set(summary))}")

    deltas = {}
    for row in _read_csv(delta_path, DELTA_COLUMNS):
        key = (row["family"], row["bucket"])
        if key[0] not in FAMILIES or key[1] not in BUCKETS or key in deltas:
            raise ValueError(f"unknown or duplicate family/bucket: {key}")
        values = {k: _float(row, k) for k in DELTA_COLUMNS if k.startswith(("delta_", "ci_"))}
        for metric in ("accuracy_top1", "nll_legal"):
            if values[f"ci_low_{metric}"] > values[f"ci_high_{metric}"]:
                raise ValueError(f"reversed confidence interval: {key}, {metric}")
        values["n_games"] = _int(row, "n_games")
        values["n_plies"] = _int(row, "n_plies")
        deltas[key] = values
    expected_delta = {(family, bucket) for family in FAMILIES for bucket in BUCKETS}
    if set(deltas) != expected_delta:
        raise ValueError(f"missing rows in paired_deltas.csv: {sorted(expected_delta - set(deltas))}")

    for bucket in BUCKETS:
        reference = summary[(MODELS[0], bucket)]
        for model in MODELS[1:]:
            item = summary[(model, bucket)]
            if (item["n_games"], item["n_plies"]) != (reference["n_games"], reference["n_plies"]):
                raise ValueError(f"unequal samples between models in bucket {bucket}")
        for family in FAMILIES:
            item = deltas[(family, bucket)]
            if (item["n_games"], item["n_plies"]) != (reference["n_games"], reference["n_plies"]):
                raise ValueError(f"paired sample differs in bucket {bucket}")
            for metric in ("accuracy_top1", "nll_legal"):
                measured = (summary[(f"{family}_balanced", bucket)][metric]
                            - summary[(f"{family}_baseline", bucket)][metric])
                if not math.isclose(item[f"delta_{metric}"], measured, abs_tol=0.00011):
                    raise ValueError(f"delta does not match model means: {family}, {bucket}, {metric}")
    return summary, deltas


def _txt(x, y, value, *, size=13, color="#263443", anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{escape(str(value))}</text>')


def _save_svg(path: Path, elements: list[str], width=1050, height=620):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">\n'
           '<rect width="100%" height="100%" fill="white"/>\n'
           '<g font-family="Arial, Helvetica, sans-serif">\n'
           + "\n".join(elements) + "\n</g>\n</svg>\n")
    path.write_text(svg, encoding="utf-8")


def _axis(elements, ymin, ymax, *, title, ylabel, percent=False):
    left, right, top, bottom = 105, 990, 100, 495
    elements.append(_txt(525, 43, title, size=23, weight="bold"))
    elements.append(_txt(27, 300, ylabel, size=16, anchor="start"))
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = bottom - (bottom - top) * i / 5
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e4e9ed"/>')
        label = f"{value * 100:.0f}%" if percent else f"{value:.2f}"
        elements.append(_txt(left - 12, y + 4, label, anchor="end", size=12))
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#586676"/>')
    elements.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#586676"/>')
    for i, label in enumerate(BUCKET_LABELS):
        x = left + 55 + i * 155
        elements.append(_txt(x, bottom + 24, label, size=13))
    elements.append(_txt(545, 553, "Ply before the target move", size=14))
    return lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)


def plot_levels(path: Path, summary, metric: str, title: str, ylabel: str, percent=False):
    values = [summary[(model, bucket)][metric] for model in MODELS for bucket in BUCKETS]
    ymin = min(values)
    ymax = max(values)
    pad = max((ymax - ymin) * 0.2, 0.02 if percent else 0.03)
    ymin, ymax = (max(0, ymin - pad), min(1, ymax + pad)) if percent else (max(0, ymin - pad), ymax + pad)
    if ymax - ymin < 0.001:
        ymax = ymin + (0.05 if percent else 0.1)
    elements = []
    y_of = _axis(elements, ymin, ymax, title=title, ylabel=ylabel, percent=percent)
    for model in MODELS:
        points = [(160 + i * 155, y_of(summary[(model, bucket)][metric]))
                  for i, bucket in enumerate(BUCKETS)]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[model]}" stroke-width="3"/>')
        for x, y in points:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[model]}"/>')
    for i, model in enumerate(MODELS):
        x = 132 + i * 225
        elements.append(f'<line x1="{x}" y1="586" x2="{x+22}" y2="586" stroke="{COLORS[model]}" stroke-width="4"/>')
        elements.append(_txt(x + 28, 591, LABELS[model], size=12, anchor="start"))
    _save_svg(path, elements)


def plot_delta(path: Path, deltas, metric: str, title: str, ylabel: str, percent=False):
    limits = [0.0]
    for family in FAMILIES:
        for bucket in BUCKETS:
            row = deltas[(family, bucket)]
            limits.extend((row[f"ci_low_{metric}"], row[f"ci_high_{metric}"]))
    span = max(limits) - min(limits)
    pad = max(span * 0.13, 0.01 if percent else 0.02)
    ymin, ymax = min(limits) - pad, max(limits) + pad
    elements = []
    y_of = _axis(elements, ymin, ymax, title=title, ylabel=ylabel, percent=percent)
    zero = y_of(0)
    elements.append(f'<line x1="105" y1="{zero:.1f}" x2="990" y2="{zero:.1f}" stroke="#45515d" stroke-dasharray="6 5"/>')
    for family, offset in (("encoder", -18), ("decoder", 18)):
        points = []
        for i, bucket in enumerate(BUCKETS):
            row = deltas[(family, bucket)]
            x = 160 + i * 155 + offset
            y, lo, hi = (y_of(row[key]) for key in
                         (f"delta_{metric}", f"ci_low_{metric}", f"ci_high_{metric}"))
            color = COLORS[family]
            elements.append(f'<line x1="{x}" y1="{hi:.1f}" x2="{x}" y2="{lo:.1f}" stroke="{color}" stroke-width="2"/>')
            for cap in (hi, lo):
                elements.append(f'<line x1="{x-5}" y1="{cap:.1f}" x2="{x+5}" y2="{cap:.1f}" stroke="{color}" stroke-width="2"/>')
            elements.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{color}"/>')
            points.append((x, y))
        elements.append('<polyline points="' + " ".join(f"{x},{y:.1f}" for x, y in points)
                        + f'" fill="none" stroke="{COLORS[family]}" stroke-width="2"/>')
    for i, family in enumerate(FAMILIES):
        x = 360 + i * 240
        elements.append(f'<circle cx="{x}" cy="586" r="5" fill="{COLORS[family]}"/>')
        elements.append(_txt(x + 12, 591, LABELS[family], size=13, anchor="start"))
    _save_svg(path, elements)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(path: Path, summary, deltas, summary_path: Path, delta_path: Path):
    manifest_path = summary_path.parent / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else None)
    lines = [
        "# TWIC 1662–1667 checkpoint comparison", "",
        "Generated from the evaluator's aggregate CSVs. The project owner confirmed",
        "that TWIC issues 1662–1667 are absent from all four training runs. This",
        "script validates and plots the summaries; model inference and the",
        "game-cluster bootstrap are performed by the evaluator.", "",
    ]
    if manifest is not None:
        lines.extend([
            "## Evaluated sample", "",
            f"- Games scored: {manifest['accepted_games']:,} of {manifest['corpus_games']:,} corpus games.",
            f"- Decisions scored: {sum(int(summary[(MODELS[0], bucket)]['n_plies']) for bucket in BUCKETS):,}.",
            f"- Device: `{manifest['device']}`.",
        ])
        if manifest.get("sampling"):
            sample_count = manifest["sampling"]["games_per_issue"]
            lines.append(
                f"- Exploratory sample: {sample_count} hash-ranked "
                f"{'game' if sample_count == 1 else 'games'} per TWIC issue. "
                "These estimates do not cover "
                "the full corpus and have limited precision."
            )
        else:
            lines.append("- Cohort: all eligible games in TWIC 1662–1667.")
        lines.extend(["", "The builder applies the `classic` event-name heuristic, both",
                      "player Elos at least 1600, standard chess, valid nonempty",
                      "PGNs, no null moves, and complete-game deduplication.", ""])
        pgn_info = manifest.get("files", {}).get("pgn", {})
        if pgn_info.get("path"):
            audit_path = Path(pgn_info["path"]).with_name("twic_1662_1667_corpus_audit.json")
            if audit_path.exists():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("pgn_sha256") == pgn_info.get("sha256"):
                    fast = audit["explicit_180_or_300_second_games"]
                    lines.extend([
                        f"The `classic` heuristic includes {fast:,} of "
                        f"{audit['total_games']:,} games with explicit 180- or "
                        "300-second controls. Results describe a mixed-speed cohort.",
                        "",
                    ])
    lines.extend([
        "## Implemented outputs", "",
        "- `accuracy_by_ply.svg`: legal UCI top-1 accuracy for all four checkpoints.",
        "- `nll_by_ply.svg`: mean legal-move NLL of the recorded human move.",
        "- `delta_accuracy_by_ply.svg`: balanced minus baseline accuracy, with 95% CIs.",
        "- `delta_nll_by_ply.svg`: balanced minus baseline legal NLL, with 95% CIs.", "",
        "Bands use the ply index before the target move: `[0,20)`, `[20,40)`,",
        "`[40,60)`, `[60,80)`, `[80,100)`, and `[100,∞)`. They indicate numeric",
        "progression, not a chess-phase classification.", "",
        "## Metrics", "",
        "At each position, top-1 accuracy is 1 when the model's complete legal UCI",
        "move matches the recorded move, and 0 otherwise. Top-5 includes the greedy",
        "choice followed by the four highest-probability remaining legal moves.",
        "Band accuracy is the mean across decisions. Legal NLL is",
        "`-ln p(recorded_move | legal_moves)`; legal perplexity is `exp(mean NLL)`.",
        "Promotions are evaluated as complete UCI moves. Each paired difference",
        "is balanced minus baseline on the same decisions. The 95% CIs use",
        "paired resampling of complete games. Human-move agreement does not",
        "directly measure chess strength or establish decoder state drift.", "",
        "## Sample size by band", "",
        "| Ply | Games | Decisions |", "|---|---:|---:|",
    ])
    for bucket, label in zip(BUCKETS, BUCKET_LABELS):
        row = summary[(MODELS[0], bucket)]
        lines.append(f"| {label} | {row['n_games']} | {row['n_plies']} |")
    lines.extend(["", "## Paired differences", "",
                  "| Family | Ply | Δ accuracy (pp) | 95% CI (pp) | Δ NLL | 95% CI |",
                  "|---|---|---:|---:|---:|---:|"])
    for family in FAMILIES:
        for bucket, label in zip(BUCKETS, BUCKET_LABELS):
            row = deltas[(family, bucket)]
            lines.append(
                f"| {LABELS[family]} | {label} | {row['delta_accuracy_top1']*100:+.2f} | "
                f"[{row['ci_low_accuracy_top1']*100:+.2f}, {row['ci_high_accuracy_top1']*100:+.2f}] | "
                f"{row['delta_nll_legal']:+.4f} | "
                f"[{row['ci_low_nll_legal']:+.4f}, {row['ci_high_nll_legal']:+.4f}] |"
            )
    lines.extend(["", "## Summary provenance", "",
                  f"- `bucket_summary.csv`: SHA-256 `{_sha256(summary_path)}`",
                  f"- `paired_deltas.csv`: SHA-256 `{_sha256(delta_path)}`", "",
                  "The evaluator manifest records PGN, tokenizer, and checkpoint hashes,",
                  "the Git revision, filters, and runtime versions. This README is not a",
                  "substitute for that manifest.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="bucket_summary.csv")
    parser.add_argument("--deltas", type=Path, required=True, help="paired_deltas.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, deltas = load_results(args.summary, args.deltas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_levels(args.output_dir / "accuracy_by_ply.svg", summary, "accuracy_top1",
                "Human-move accuracy by ply band", "Top-1 accuracy", percent=True)
    plot_levels(args.output_dir / "nll_by_ply.svg", summary, "nll_legal",
                "Human-move legal NLL by ply band", "Legal NLL")
    plot_delta(args.output_dir / "delta_accuracy_by_ply.svg", deltas, "accuracy_top1",
               "Paired accuracy difference", "Balanced − baseline", percent=True)
    plot_delta(args.output_dir / "delta_nll_by_ply.svg", deltas, "nll_legal",
               "Paired legal NLL difference", "Balanced − baseline")
    write_readme(args.output_dir / "README.md", summary, deltas, args.summary, args.deltas)
    print(f"Charts and README written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
