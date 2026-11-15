"""Configuration, content identities and atomic artifact writing."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

CATEGORIES = ("classic", "rapid", "blitz")
PAIRS = (("rapid", "classic"), ("blitz", "classic"), ("rapid", "blitz"))
BUCKETS = ("0_20", "20_40", "40_60", "60_80", "80_100", "100_plus")
CODE_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "time_control_comparison_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        destination.write(value)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def code_hashes() -> dict[str, str]:
    # Include local model/preprocessing implementations as well as evaluator code.
    paths = {CODE_ROOT / "analysis/time_control_comparison" / name for name in (
        "__main__.py", "common.py", "corpus.py", "inference.py", "storage.py")}
    paths.update(CODE_ROOT / "models" / name for name in (
        "chess_encoder_player.py", "chess_visual_classifier.py", "chess_vit.py", "chessformer.py"))
    paths.update(CODE_ROOT / name for name in (
        "dataset/renderer.py", "dataset/offsets.py", "utils/dataset_by_category/categories.py"))
    return {str(p.relative_to(CODE_ROOT)): sha256(p) for p in sorted(paths)}


def bucket(ply: int) -> str:
    return BUCKETS[min(ply // 20, 5)]


@dataclass(frozen=True)
class Config:
    project_root: Path
    corpus_dir: Path
    prefix: str
    output_dir: Path
    checkpoints: dict[str, Path]
    games_per_category: int = 1000
    seed: int = 42
    min_elo: int = 1600
    lower_week: int = 1662
    start_week: int = 1667
    batch_size: int = 64
    bootstrap: int = 2000
    allow_shortfall: bool = False
    exclude_pgns: tuple[Path, ...] = ()

    def selection_settings(self) -> dict:
        return {key: getattr(self, key) for key in (
            "games_per_category", "seed", "min_elo", "lower_week", "start_week", "allow_shortfall")}


def load_config(path: Path, project_root: Path | None = None) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {"corpus_dir", "prefix", "output_dir", "checkpoints", "games_per_category", "seed",
               "min_elo", "lower_week", "start_week", "batch_size", "bootstrap", "allow_shortfall", "exclude_pgns"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    root = (project_root or CODE_ROOT).resolve()
    def resolve(p):
        return (root / p).resolve()
    if set(raw["checkpoints"]) != set(CATEGORIES):
        raise ValueError("checkpoints must specify classic, rapid and blitz")
    values = dict(raw)
    for key in ("corpus_dir", "output_dir"):
        values[key] = resolve(values[key])
    values["checkpoints"] = {name: resolve(p) for name, p in raw["checkpoints"].items()}
    values["exclude_pgns"] = tuple(resolve(p) for p in raw.get("exclude_pgns", []))
    config = Config(project_root=root, **values)
    if any(getattr(config, name) < 1 for name in ("games_per_category", "batch_size", "bootstrap", "min_elo")):
        raise ValueError("game quota, batch size, bootstrap count and min Elo must be positive")
    if config.lower_week > config.start_week:
        raise ValueError("lower_week must not exceed start_week")
    if not isinstance(config.allow_shortfall, bool):
        raise ValueError("allow_shortfall must be a boolean")
    return config
