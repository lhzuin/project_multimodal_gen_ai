#!/usr/bin/env python3
"""Check indexed submission files and the compact result snapshots; read-only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = {
    ".venv", ".worktrees", "outputs", "processed_games", "checkpoints", "twic_zips",
    "logs", "wandb", "tournament_out", "distill_db", "merged_distill", "__pycache__",
}
GENERATED_SUFFIXES = {".pt", ".pth", ".ckpt", ".pgn", ".zip", ".npy", ".npz", ".bin", ".sqlite", ".pyc", ".mp4", ".mov"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-mib", type=float, default=5)
    args = parser.parse_args(argv)
    if args.max_file_mib <= 0:
        parser.error("max-file-mib must be positive")
    listing = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    files = {}
    errors = []
    for entry in listing.decode().split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, oid, stage = metadata.split()
        path = Path(name)
        if stage != "0":
            errors.append(f"Unresolved index conflict: {name}")
        if mode == "160000":
            errors.append(f"Unexpected nested Git checkout: {name}")
        if (set(path.parts) & GENERATED_DIRS or path.suffix in GENERATED_SUFFIXES
                or path.name in {".DS_Store", ".env", "p64_output.txt"}
                or "_samples" in path.name or path.name.endswith("_index.json")):
            errors.append(f"Generated/local artifact is indexed: {name}")
        files[name] = oid
    sizes = subprocess.check_output(
        ["git", "cat-file", "--batch-check=%(objectsize)"], cwd=ROOT,
        input=("\n".join(files.values()) + "\n").encode(),
    ).decode().splitlines()
    total = 0
    for name, size in zip(files, sizes, strict=True):
        size = int(size)
        total += size
        if size > args.max_file_mib * 1024**2:
            errors.append(f"File exceeds {args.max_file_mib:g} MiB: {name} ({size:,} bytes)")
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    for entry in manifest["files"]:
        relative = Path("results") / entry["path"]
        path = ROOT / relative
        if str(relative) not in files:
            errors.append(f"Curated result is not indexed: {relative}")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Curated result checksum differs: {relative}")
        if path.is_file() and path.stat().st_size != entry["size_bytes"]:
            errors.append(f"Curated result size differs: {relative}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Submission check passed: {len(files)} indexed files, {total / 1024**2:.2f} MiB; {len(manifest['files'])} result snapshots verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
