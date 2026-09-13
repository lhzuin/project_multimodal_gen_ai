"""Keep a CSV-defined training queue running in tmux.

The command is intentionally a one-shot process.  systemd can invoke it from a
timer, which gives each poll a clear process boundary and avoids a second
orchestrator surviving an interrupted deployment.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {
    "id",
    "status",
    "command",
    "working_dir",
    "session_name",
}
COMPLETED = "completed"
RUNNING = "running"
PENDING = "pending"


class OrchestratorError(RuntimeError):
    """Raised when the queue cannot be safely processed."""


def _run_tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _session_exists(session_name: str) -> bool:
    return _run_tmux("has-session", "-t", session_name).returncode == 0


def _pane_state(session_name: str) -> tuple[bool, int | None]:
    result = _run_tmux(
        "display-message",
        "-p",
        "-t",
        f"{session_name}:0.0",
        "#{pane_dead}:#{pane_dead_status}",
        check=True,
    )
    dead, status = result.stdout.strip().split(":", 1)
    return dead == "1", int(status) if dead == "1" else None


def _kill_session(session_name: str) -> None:
    _run_tmux("kill-session", "-t", session_name)


def _start_session(row: dict[str, str], *, resume: bool) -> None:
    command = row["command"].strip()
    if not command:
        raise OrchestratorError(f"Training {row['id']!r} has an empty command")
    working_dir = Path(row["working_dir"]).expanduser()
    if not working_dir.is_dir():
        raise OrchestratorError(
            f"Training {row['id']!r} working_dir does not exist: {working_dir}"
        )

    argv = shlex.split(command)
    if resume:
        override = row.get("resume_override", "checkpoint_load_path=auto").strip()
        if override:
            argv.append(override)

    session = row["session_name"]
    _run_tmux("new-session", "-d", "-s", session, "-c", str(working_dir), check=True)
    # Set this before sending the command so a fast failure remains observable.
    _run_tmux("set-option", "-t", session, "remain-on-exit", "on", check=True)
    _run_tmux(
        "send-keys",
        "-t",
        f"{session}:0.0",
        shlex.join(argv),
        "Enter",
        check=True,
    )


def _validate_rows(fieldnames: Iterable[str] | None, rows: list[dict[str, str]]) -> None:
    columns = set(fieldnames or ())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise OrchestratorError(f"Queue is missing required columns: {', '.join(sorted(missing))}")
    seen_ids: set[str] = set()
    seen_sessions: set[str] = set()
    for row in rows:
        row_id = row["id"].strip()
        session = row["session_name"].strip()
        if not row_id or row_id in seen_ids:
            raise OrchestratorError(f"Queue contains a missing or duplicate id: {row_id!r}")
        if not session or session in seen_sessions:
            raise OrchestratorError(f"Queue contains a missing or duplicate session_name: {session!r}")
        if row["status"].strip().lower() not in {PENDING, RUNNING, COMPLETED}:
            raise OrchestratorError(f"Unknown status for {row_id!r}: {row['status']!r}")
        seen_ids.add(row_id)
        seen_sessions.add(session)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def process_queue(path: Path) -> str | None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    _validate_rows(fieldnames, rows)

    for row in rows:
        row["status"] = row["status"].strip().lower()
        if row["status"] == COMPLETED:
            continue

        session_name = row["session_name"].strip()
        if _session_exists(session_name):
            dead, exit_code = _pane_state(session_name)
            if not dead:
                return row["id"]
            if exit_code == 0:
                row["status"] = COMPLETED
                _kill_session(session_name)
                _write_rows(path, fieldnames, rows)
                continue
            row["status"] = RUNNING
            _kill_session(session_name)
            _write_rows(path, fieldnames, rows)
            _start_session(row, resume=True)
            return row["id"]

        _start_session(row, resume=row["status"] == RUNNING)
        row["status"] = RUNNING
        _write_rows(path, fieldnames, rows)
        return row["id"]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover and sequence tmux training jobs.")
    parser.add_argument("--csv", type=Path, required=True, help="Training queue CSV")
    args = parser.parse_args()
    process_queue(args.csv.expanduser().resolve())


if __name__ == "__main__":
    main()
