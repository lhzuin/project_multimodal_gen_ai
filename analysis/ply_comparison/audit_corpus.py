"""Audit dates and time-control metadata in the frozen TWIC evaluation PGN."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import chess.pgn

from dataset.offsets import load_pgn_offsets


ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(pgn_path: Path, index_path: Path) -> dict:
    index = load_pgn_offsets(str(index_path))
    issues: dict[int, dict] = {}
    with pgn_path.open("rb") as source:
        for row in index["games"]:
            source.seek(int(row["start"]))
            raw = source.read(int(row["end"]) - int(row["start"]))
            headers = chess.pgn.read_headers(io.StringIO(raw.decode("utf-8", errors="replace")))
            if headers is None:
                raise ValueError(f"unreadable headers at game {row['idx']}")
            issue = int(headers["TWICIssue"])
            if not 1662 <= issue <= 1667:
                raise ValueError(f"unexpected TWIC issue {issue}")
            stats = issues.setdefault(issue, {
                "games": 0, "dates_valid": 0, "dates_unknown_or_invalid": 0,
                "date_min": None, "date_max": None, "time_control_present": 0,
                "years": Counter(), "time_controls": Counter(),
            })
            stats["games"] += 1
            time_control = headers.get("TimeControl")
            if time_control not in (None, "", "-", "?"):
                stats["time_control_present"] += 1
                stats["time_controls"][time_control] += 1
            raw_date = headers.get("Date", "")
            try:
                date = datetime.strptime(raw_date, "%Y.%m.%d").date().isoformat()
            except ValueError:
                stats["dates_unknown_or_invalid"] += 1
            else:
                stats["dates_valid"] += 1
                stats["date_min"] = min(stats["date_min"] or date, date)
                stats["date_max"] = max(stats["date_max"] or date, date)
                stats["years"][date[:4]] += 1
    if set(issues) != set(range(1662, 1668)):
        raise ValueError("one or more required TWIC issues are absent")
    if sum(stats["games"] for stats in issues.values()) != index["num_games"]:
        raise ValueError("audited game count differs from the offset index")
    return {
        "format": "twic_corpus_audit_v1",
        "pgn_sha256": file_sha256(pgn_path),
        "index_sha256": file_sha256(index_path),
        "total_games": index["num_games"],
        "explicit_180_or_300_second_games": sum(
            stats["time_controls"].get("180", 0) + stats["time_controls"].get("300", 0)
            for stats in issues.values()
        ),
        "issues": {str(issue): {**stats,
                                "years": dict(sorted(stats["years"].items())),
                                "time_controls_top_10": stats["time_controls"].most_common(10),
                                "time_controls": dict(sorted(stats["time_controls"].items()))}
                   for issue, stats in sorted(issues.items())},
        "note": "TWIC issue establishes source provenance; PGN Date is the game date, when valid. Event-name classification does not certify clock settings.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_classic.pgn")
    parser.add_argument("--index", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_classic_index.json")
    parser.add_argument("--output", type=Path, default=ROOT / "processed_games/eval/twic_1662_1667_corpus_audit.json")
    args = parser.parse_args(argv)
    result = audit(args.pgn, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {result['total_games']} games; wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
