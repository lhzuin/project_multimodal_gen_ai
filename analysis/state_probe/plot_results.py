"""Render dependency-free SVG figures and the generated experiment report."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path


COLORS = ("#315b8a", "#b95f3b", "#2f8f72", "#8056a3", "#ba8b2f", "#4f7484")


def _read(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _svg_line_chart(path: Path, title: str, ylabel: str, labels: list[str], series: dict[str, list[float]]):
    if not labels or not series:
        raise ValueError(f"cannot plot empty data: {title}")
    values = [value for points in series.values() for value in points if math.isfinite(value)]
    low, high = min(values), max(values)
    padding = max((high - low) * .15, .01)
    low, high = low - padding, high + padding
    width, height = 1100, 650
    left, right, top, bottom = 100, 1040, 90, 520
    elements = [f'<text x="550" y="40" text-anchor="middle" font-size="24" font-weight="bold">{escape(title)}</text>']
    for i in range(6):
        value = low + (high - low) * i / 5
        y = bottom - (bottom - top) * i / 5
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e3e8ec"/>')
        elements.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.3f}</text>')
    elements.append(f'<text x="18" y="300" font-size="15">{escape(ylabel)}</text>')
    denominator = max(1, len(labels) - 1)
    xs = [left + i * (right - left) / denominator for i in range(len(labels))]
    for x, label in zip(xs, labels):
        elements.append(f'<text x="{x:.1f}" y="548" text-anchor="middle" font-size="12">{escape(label)}</text>')
    for index, (name, points) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        ys = [bottom - (point - low) / (high - low) * (bottom - top) for point in points]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in zip(xs, ys):
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_x = 110 + (index % 3) * 330
        legend_y = 590 + (index // 3) * 25
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+25}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        elements.append(f'<text x="{legend_x+32}" y="{legend_y+5}" font-size="12">{escape(name)}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">'
           + "".join(elements) + "</g></svg>\n")
    path.write_text(svg, encoding="utf-8")


def _lookup(rows, *, source=None, model=None, condition=None, bucket=None, metric="board_nll"):
    selected = [row for row in rows
                if (source is None or row["trajectory_source"] == source)
                and (model is None or row["encoded_by"] == model)
                and (condition is None or row["condition"] == condition)
                and (bucket is None or row["bucket"] == bucket)]
    if not selected:
        return math.nan
    weights = [int(row["n_positions"]) for row in selected]
    return sum(float(row[metric]) * weight for row, weight in zip(selected, weights)) / sum(weights)


def write_report(path: Path, rows, deltas, interactions, associations, output: Path):
    manifest_path = output / "evaluate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    lines = [
        "# Decoder state drift results", "",
        "## What was measured", "",
        "Frozen linear probes reconstruct the board, turn, castling rights, and legal en-passant square from decoder hidden states. The teacher-forced condition uses the true preceding PGN moves. The on-policy condition starts after eight human plies and lets one decoder control both colors with greedy legal moves. Every generated history is replayed through both decoder checkpoints.", "",
        "The decoder recomputes its state from its token window. In this report, drift means declining state recoverability from that history or movement into positions that are harder and less familiar; it does not mean propagation of a recurrent numeric state.", "",
        "## Primary metrics", "",
        "- **Board NLL per square:** occupancy NLL on every square plus piece-identity NLL on occupied squares, divided by 64.",
        "- **Occupied-state NLL:** occupancy plus identity NLL averaged across occupied squares.",
        "- **Normalized board Hamming error:** fraction of squares whose predicted 13-class state is wrong.",
        "- **Common-reference novelty:** position absent from the Bloom index of the large pre-1662 TWIC reservoir. This is not exact training-set membership.", "",
        "Intervals in `paired_deltas.csv` resample complete games. Self-play comparisons retain complete openings as their analysis unit.", "",
        "## Evaluated rows", "",
        f"- Full-test primary teacher-forced rows: {manifest.get('teacher_rows', 'unknown')}",
        f"- Selected-sample diagnostic rows: {manifest.get('teacher_diagnostic_rows', 'unknown')}",
        f"- Self-play cross-replay rows: {manifest.get('selfplay_rows', 'unknown')}", "",
        "## How to interpret the outputs", "",
        "Representation degradation requires worse probe metrics with ply on the same human histories. A trajectory effect is supported when self-play novelty and probe error rise after branching. If both encoders struggle on positions generated by one policy, that policy selected harder states. If one encoder is worse on identical histories, the deficit is checkpoint-specific. Stable probe performance with weak policy metrics points toward the policy head, calibration, move quality, or exposure weighting.", "",
        "Probe recoverability only shows that information is available to a simple readout. It does not prove that the move head uses that information causally.", "",
        "## Output tables", "",
        "- `bucket_summary.csv`: model, trajectory, condition, and ply-band means.",
        "- `paired_deltas.csv`: balanced-minus-baseline paired teacher-history differences with 95% intervals.",
        "- `cross_replay_interactions.csv`: both representations on each generated trajectory source.",
        "- `drift_interactions.csv`: late-minus-early changes and checkpoint difference-in-differences.",
        "- `trajectory_gaps.csv`: self-play-minus-human gaps in the same absolute ply bands.",
        "- `novelty_summary.csv`: common-reference novelty by source and band.",
        "- `policy_error_association.csv`: human move metrics by probe-error quartile.", "",
        "## Figures", "",
        "The `figures/` directory contains the layer, context-window, trajectory, cross-replay, novelty, and policy-association plots. Numeric CSVs are authoritative.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _read(args.output_dir / "bucket_summary.csv")
    deltas = _read(args.output_dir / "paired_deltas.csv")
    interactions = _read(args.output_dir / "cross_replay_interactions.csv")
    associations = _read(args.output_dir / "policy_error_association.csv")
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = ("decoder_baseline", "decoder_balanced")
    ply_buckets = ("0_20", "20_40", "40_60", "60_80", "80_100", "100_plus")
    gen_buckets = ("0_10", "10_20", "20_40", "40_80", "80_120")
    layer_conditions = ("layer3_w256", "layer6_w256", "layer9_w256", "final_w256_sample")
    _svg_line_chart(figures / "layerwise_decodability.svg", "Layerwise board-state decodability",
                    "Board NLL", ["3", "6", "9", "12"],
                    {model: [_lookup(rows, source="human", model=model, condition=condition)
                             for condition in layer_conditions] for model in models})
    context_conditions = ("final_w32", "final_w64", "final_w128", "final_w256_sample")
    _svg_line_chart(figures / "context_length.svg", "Context length sensitivity", "Board NLL",
                    ["32", "64", "128", "256"],
                    {model: [_lookup(rows, source="human", model=model, condition=condition)
                             for condition in context_conditions] for model in models})
    trajectory_series = {}
    for source in ("human", *models):
        labels = ply_buckets if source == "human" else gen_buckets
        for model in models:
            trajectory_series[f"{source} / {model}"] = [
                _lookup(rows, source=source, model=model, condition="final_w256", bucket=bucket)
                for bucket in labels
            ]
    # Human has six bands and generated has five. Use the first five common display slots.
    trajectory_series = {key: value[:5] for key, value in trajectory_series.items()}
    _svg_line_chart(figures / "human_vs_selfplay_drift.svg", "Human and self-play state drift",
                    "Board NLL", list(gen_buckets), trajectory_series)
    _svg_line_chart(figures / "cross_replay.svg", "Cross-replay comparison", "Board NLL",
                    list(gen_buckets),
                    {f"{source} / {model}": [_lookup(rows, source=source, model=model,
                                                      condition="final_w256", bucket=bucket)
                                                     for bucket in gen_buckets]
                     for source in models for model in models})
    _svg_line_chart(figures / "common_reference_novelty.svg", "Common-reference novelty",
                    "Novel fraction", list(gen_buckets),
                    {source: [_lookup(rows, source=source, model="decoder_baseline",
                                      condition="final_w256", bucket=bucket, metric="novel_fraction")
                              for bucket in gen_buckets] for source in models})
    assoc_series = {}
    for model in models:
        points = []
        for quartile in range(1, 5):
            selected = [row for row in associations if row["encoded_by"] == model
                        and int(row["probe_error_quartile"]) == quartile]
            points.append(sum(float(row["policy_nll"]) * int(row["n_positions"]) for row in selected)
                          / sum(int(row["n_positions"]) for row in selected))
        assoc_series[model] = points
    _svg_line_chart(figures / "policy_error_association.svg", "Policy NLL by probe-error quartile",
                    "Legal move NLL", ["Q1", "Q2", "Q3", "Q4"], assoc_series)
    write_report(args.output_dir / "README.md", rows, deltas, interactions, associations, args.output_dir)
    print(f"State-probe figures and report written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
